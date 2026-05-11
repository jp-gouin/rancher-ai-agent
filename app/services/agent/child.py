"""
Agent builder for creating LangGraph agents with tool execution and human-in-the-loop validation.

Provides create_child_agent which constructs agents for use under a supervisor (multi-agent)
or standalone (single-agent) setups using the langchain create_agent factory with middleware.
Each agent has an LLM-driven reasoning loop with tool execution, human validation gates,
and automatic retry on malformed tool calls.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import langgraph.types
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    SummarizationMiddleware,
    after_model,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.graph.state import Checkpointer, CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from .loader import AgentConfig
from .middleware import (
    INTERRUPT_CANCEL_MESSAGE,
    create_ui_tools_middleware,
    _dispatch_ui_tools,
    create_cancel_check_middleware,
    create_inject_request_id_middleware,
)

INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE = "tool execution cancelled because previous tool call failed"


def create_child_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Checkpointer,
    agent_config: AgentConfig,
) -> CompiledStateGraph:
    """Create and compile a child agent graph using langchain create_agent with middleware.

    The agent uses the same create_agent factory as the supervisor, with middleware
    implementing: human-in-the-loop validation, metadata injection, UI tools dispatch,
    and tool execution error handling.
    """
    planning_tools = [t for t in tools if t.name.endswith("Plan")]
    execution_tools = [t for t in tools if not t.name.endswith("Plan")]
    planning_tools_by_name = {t.name: t for t in planning_tools}

    middleware = [
        _create_tool_execution_middleware(llm, planning_tools_by_name, agent_config),
        create_cancel_check_middleware(),
        create_inject_request_id_middleware(),
        _create_inject_selected_agent_middleware(agent_config),
        create_ui_tools_middleware(llm, only_when_direct=True),
        SummarizationMiddleware(model=llm, trigger=[("messages", 30), ("tokens", 6000)]),
    ]

    return create_agent(
        llm,
        tools=execution_tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=middleware,
    )


# =============================================================================
# Middleware factories
# =============================================================================


def _create_inject_selected_agent_middleware(agent_config: AgentConfig):
    """After-model middleware: inject selected_agent into the last AIMessage."""

    @after_model
    def inject_selected_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        last_message.additional_kwargs["selected_agent"] = state.get("selected_agent", {})

        return {"messages": [last_message]}

    return inject_selected_agent


def _create_tool_execution_middleware(
    llm: BaseChatModel,
    planning_tools_by_name: dict[str, BaseTool],
    agent_config: AgentConfig,
):
    """Wrap-tool-call middleware: human validation, MCP response processing, error handling."""

    @wrap_tool_call
    async def tool_execution(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        config = get_config()
        request_id = config.get("configurable", {}).get("request_id", "")
        state = request.state
        tool_call = request.tool_call

        additional_kwargs: dict = {
            "request_id": request_id,
            "selected_agent": state.get("selected_agent", {}),
        }

        # Human validation / interrupt
        human_validation_tools = getattr(agent_config, "human_validation_tools", [])
        interrupt_message = await _should_interrupt(human_validation_tools, tool_call, planning_tools_by_name)

        if interrupt_message:
            logging.info(f"Confirmation interrupt triggered for tool '{tool_call['name']}'")

            ui_tools_list: list[dict] = []
            try:
                ui_tools_list = _build_interrupt_ui_tools(interrupt_message, state, config)
            except Exception as e:
                logging.debug(
                    f"Could not extract precomputed fields from interrupt message "
                    f"and dispatch UI tools: {e}"
                )

            response = langgraph.types.interrupt(interrupt_message)
            if response != "yes":
                additional_kwargs["interrupt_message"] = interrupt_message
                additional_kwargs["confirmation"] = False
                additional_kwargs["ui_tools"] = ui_tools_list
                return ToolMessage(
                    content=INTERRUPT_CANCEL_MESSAGE,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs,
                )

            additional_kwargs["interrupt_message"] = interrupt_message
            additional_kwargs["confirmation"] = True
            additional_kwargs["ui_tools"] = ui_tools_list

            selected_agent = state.get("selected_agent", {})
            if selected_agent:
                dispatch_custom_event(
                    "subagent_choice_event",
                    _build_agent_metadata(selected_agent.get("name"), selected_agent.get("mode")),
                )

        # Execute the tool
        try:
            logging.debug("calling tool")
            result = await handler(request)
            logging.debug("tool call finished")

            # Process result for MCP responses
            if isinstance(result, ToolMessage):
                processed_content, mcp_response = _process_tool_result(result.content, state)
                result.content = processed_content
                if mcp_response:
                    additional_kwargs["mcp_response"] = mcp_response
                result.additional_kwargs = {**result.additional_kwargs, **additional_kwargs}

            return result
        except Exception as e:
            logging.error(f"unexpected error during tool call: {e}")
            return ToolMessage(
                content=f"unexpected error during tool call: {e}",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs=additional_kwargs,
            )

    return tool_execution


# =============================================================================
# Helper functions
# =============================================================================


async def _should_interrupt(
    human_validation_tools: list[str],
    tool_call: dict,
    planning_tools_by_name: dict[str, BaseTool],
) -> str:
    """Return a confirmation prompt if *tool_call* requires human validation, else ``''``."""
    for tool_name in human_validation_tools:
        if tool_name == tool_call["name"]:
            plan_tool_name = tool_call["name"] + "Plan"
            plan_tool = planning_tools_by_name.get(plan_tool_name)
            if plan_tool is None:
                raise ValueError(
                    f"planning tool '{plan_tool_name}' not found for tool '{tool_call['name']}'"
                )
            plan_response = await plan_tool.ainvoke(tool_call["args"])

            # Normalize list response: [{"type": "text", "text": "..."}]
            if isinstance(plan_response, list) and plan_response:
                if isinstance(plan_response[0], dict) and "text" in plan_response[0]:
                    plan_response = plan_response[0]["text"]

            try:
                safe_response = json.dumps(json.loads(plan_response))
            except (json.JSONDecodeError, TypeError):
                safe_response = json.dumps(plan_response)
            return f"<confirmation-response>{safe_response}</confirmation-response>"
    return ""


def _build_interrupt_ui_tools(
    interrupt_message: str,
    state: dict,
    config: dict,
) -> list[dict]:
    """Build preprocessed UI tools from the interrupt payload and dispatch them."""
    ui_tools_list: list[dict] = []

    request_metadata = config.get("configurable", {}).get("request_metadata", {})
    ui_tools_config = request_metadata.get("ui_tools", {})
    name = ui_tools_config.get("name", "")
    if not name:
        return ui_tools_list

    data = json.loads(
        interrupt_message.strip("<confirmation-response></confirmation-response>")
    )
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    resource = data.get("resource", {})
    tool_input: dict = {
        "resourceKind": resource.get("kind"),
        "resourceName": resource.get("name"),
        "resourceNamespace": resource.get("namespace"),
    }
    ui_tool_name = "show-yaml"

    if data.get("type") == "create":
        tool_input["yaml"] = data.get("payload", {})
    else:
        ui_tool_name = "show-yaml-diff"
        tool_input["original"] = data.get("payload", {}).get("original")
        tool_input["patched"] = data.get("payload", {}).get("patched")

    tool_input = {k: v for k, v in tool_input.items() if v is not None}

    ui_tools_list = [{"toolName": ui_tool_name, "input": tool_input}]
    _dispatch_ui_tools(ui_tools_list)
    return ui_tools_list


def _build_agent_metadata(agent_name: str, selection_mode: str, extra_metadata: str = "") -> str:
    """Build a structured agent metadata string for custom events."""
    return (
        f'<agent-metadata>{{"agentName": "{agent_name}", '
        f'"selectionMode": "{selection_mode}"{extra_metadata}}}</agent-metadata>'
    )


def _process_tool_result(tool_result: str | list, state: dict) -> tuple[str, str | None]:
    """Process the raw tool result, extracting UI context and doc links if present.

    Returns:
        ``(processed_result, mcp_response)`` where *mcp_response* is ``None`` if no uiContext.
    """
    mcp_response = None
    try:
        # Handle list format: [{"type": "text", "text": "..."}]
        if isinstance(tool_result, list) and tool_result:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if "uiContext" in json_result:
            mcp_response = f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>"
            dispatch_custom_event("ui_context", mcp_response)
        llm_result = json_result.get("llm", json_result) if isinstance(json_result, dict) else json_result
        return convert_to_string_if_needed(llm_result), mcp_response
    except (json.JSONDecodeError, TypeError):
        return tool_result, mcp_response


def convert_to_string_if_needed(var):
    """Convert dicts and lists to JSON strings; pass through everything else."""
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    return var
