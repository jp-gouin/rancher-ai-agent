# Architecture

The Rancher AI Agent uses a [subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) architecture built on LangGraph. A single **supervisor agent** receives user requests and delegates work to one or more specialized **child agents** (loaded from AIAgentConfig CRDs). Each child agent is wrapped as a LangChain tool so the supervisor's LLM decides which agent(s) to invoke, coordinates their results, and synthesizes a final response.

Both supervisor and child agents are constructed with the same `create_agent` factory and share a middleware-based extension model for cross-cutting concerns.

---

## Middleware

Middleware intercepts the agent execution lifecycle at specific points — before/after the LLM call, after the full agent turn, or around individual tool calls. They are passed to `create_agent()` via the `middleware=` parameter and execute in order.

This project uses [LangChain's agent middleware system](https://python.langchain.com/docs/how_to/agent_middleware/) (`langchain.agents.middleware`).

### Middleware Types

There are two ways to define middleware: **decorator-based** (factory functions) and **class-based**.

#### Decorator-Based (Factory Functions)

Use decorators from `langchain.agents.middleware` to create middleware as factory functions. Each decorator corresponds to a lifecycle hook:

| Decorator | When it runs | Use case |
|-----------|-------------|----------|
| `@before_model` | Before the LLM is called | Inject system messages, short-circuit the LLM call |
| `@after_model` | After the LLM responds | Enrich or modify the AIMessage |
| `@after_agent` | After the full agent turn (model + tools) completes | Post-processing, dispatching events |
| `@wrap_tool_call` | Wraps each individual tool execution | Validation gates, error handling, artifact processing |

#### Class-Based

Subclass `AgentMiddleware` when your middleware needs custom state fields or multiple hooks in one unit.

### Adding a New Middleware

1. Create a new file in `app/services/agent/middleware/`.
2. Implement the middleware using the appropriate hook type. Import shared constants from `._constants` if needed.
3. Export the middleware from `__init__.py`.
4. Register the middleware in the agent's middleware list (`supervisor.py` or `child.py`) inside the `create_agent()` call.

---

## Human Validation (Human-in-the-Loop)

Human validation is the mechanism that pauses tool execution to ask the user for confirmation before performing a sensitive action (e.g. creating or modifying a Kubernetes resource).

### Child Agent Side

The `child_human_validation_middleware` (a `@wrap_tool_call` middleware in `app/services/agent/middleware/human_validation.py`) handles the gate at the child level:

1. Before executing a tool call, it checks if the tool is listed in the agent's `human_validation_tools` configuration.
2. If it is, the middleware invokes the corresponding **planning tool** (`<tool_name>Plan`) to produce a preview of the intended changes (e.g. a diff or resource manifest).
3. It then calls `langgraph.types.interrupt()` with the plan response, pausing the child graph and surfacing the confirmation prompt to the client.
4. When the graph is resumed:
   - If the user responds `"yes"`, the middleware proceeds to execute the real tool and dispatches any associated UI tools (e.g. YAML diff viewers).
   - Otherwise, it returns a cancellation `ToolMessage` (`INTERRUPT_CANCEL_MESSAGE`) and the tool call is skipped.

### Supervisor Side

Because the child agent is invoked via `ainvoke()` (not as a subgraph), a `GraphInterrupt` raised inside it is suppressed and `ainvoke()` returns normally. The supervisor handles this with a two-phase interrupt relay:

1. **Detecting a pending interrupt** — After invoking the child, the supervisor checks `aget_state()` on the child graph. If there are pending interrupts, it re-raises the interrupt at the supervisor level via `langgraph.types.interrupt(child_state.interrupts[0].value)`. This surfaces the confirmation prompt to the client through the supervisor's own graph.

2. **Resuming after user response** — On the next invocation, the supervisor's agent tool (`_invoke`) detects that the child has a pending interrupt (via `aget_state()`). It calls `langgraph.types.interrupt()` at the supervisor level to receive the user's `Command(resume=...)` value, then forwards it to the child graph with `ainvoke(Command(resume=resume_value))`.

3. **Handling cancellation** — If the resumed child returns an `INTERRUPT_CANCEL_MESSAGE`, the supervisor propagates it so that its own `cancel_human_validation_middleware` can end the graph gracefully.

### Cancellation via `cancel_human_validation_middleware`

The `cancel_human_validation_middleware` (`app/services/agent/middleware/cancel_check.py`) is a `@before_model` middleware registered on both the supervisor and child agents. After a human validation interrupt is resumed with a rejection (anything other than `"yes"`), the child returns a `ToolMessage` with `INTERRUPT_CANCEL_MESSAGE` as its content. On the next LLM turn, this middleware detects that the last message is a cancelled tool message and short-circuits the agent — it jumps directly to the `"end"` node with a cancellation reply, preventing any further LLM calls or tool executions.

This relay pattern allows human-in-the-loop confirmations to originate deep inside a child agent while the client interacts exclusively with the supervisor's interrupt surface.

---

## RBAC — Role-Based Access Control for Agents

Agent availability is gated by the requesting user's **Rancher Global Permissions**, so administrators can scope which agents each user can access without introducing a new permission model.

### Model

Each `AIAgentConfig` may declare `spec.requiredPermissions` — a list of Rancher global role names (e.g. `admin`, `restricted-admin`, `user`, or custom roles). A user may access an agent when:

- the user holds the `admin` Global Permission (**bypass** — access to all agents), **or**
- the agent has **no** `requiredPermissions` (**unrestricted** — available to all authenticated users), **or**
- the user holds **at least one** of the agent's `requiredPermissions` (**OR logic**).

### Enforcement (single server-side point)

Filtering happens in `factory.build_agent()` (`_filter_agents_for_user`) — the one place every session is built. Because the supervisor is constructed only from accessible agents:

- **UI list** (`build_chat_metadata`) only ever contains accessible agents — the client never receives restricted definitions and does no client-side filtering.
- **Prompt routing** can only route to accessible agents; the supervisor prompt instructs Liz to decline gracefully (without naming restricted agents) when a needed capability isn't available.
- **Direct agent invocation** by name cannot reach a restricted agent (it isn't in the built roster).

It **fails closed**: if identity or permissions can't be resolved, `NoAgentAvailableError` is raised and no agents are returned.

### Permission resolution — `app/services/rbac.py`

RBAC lives in its own module (sibling of `auth.py`); `loader.py` stays focused on AIAgentConfig CRD loading. A user's global permissions are the set of `globalRoleName` values on `GlobalRoleBinding` objects (`management.cattle.io/v3`) whose `userName` matches the user, read through the Kubernetes API using the agent service account (so resolution doesn't depend on the user's own RBAC). The whole feature can be disabled with `RBAC_ENABLED=false`.

`loader.load_agent_configs()` stays a pure, unfiltered loader (used by OAuth lookups that need the full roster). `rbac.load_agent_configs_for_user(user_id)` is the filtered entry point: it loads via the loader, resolves the user's permissions, and returns only accessible agents — raising `PermissionsError` (fail closed) if they can't be resolved.

> Permissions are resolved fresh on each call (no caching for now).

### UI agent list — `app/routers/agent.py`

- `GET /v1/api/agents/available` — user-scoped; returns the RBAC-filtered agents (`name`, `displayName`, `description`, `status`) the caller may access via `rbac.load_agent_configs_for_user`. This is the single source of truth for the UI's agent selector, so the UI never lists `AIAgentConfig` CRDs directly. Fails closed (503) if permissions can't be resolved.

### Configuring per-agent access

Administrators set an agent's `spec.requiredPermissions` directly on the `AIAgentConfig` CRD (the durable, auditable source of truth) through Rancher's native resource editor — no dedicated backend endpoints are needed. Changes take effect on the next agent-list fetch (no restart).

### Resolved PRD open questions

| Question | Resolution |
|---|---|
| Cache user permissions? How long? | Yes — in-memory, short TTL via `RBAC_PERMISSIONS_CACHE_TTL` (default 30s). |
| Endpoint to resolve global permissions? | `GlobalRoleBinding` resources in `management.cattle.io/v3`, read via the Kubernetes API by the agent service account. |
| Default for agents with no permissions? | Unrestricted (all authenticated users) — preserves upgrade behavior. |
| Agent groups sharing a permission set? | Individual per-agent configuration for this release; grouping is future work. |
| Routing with partial availability? | Supervisor is built only with accessible agents, so it naturally routes to the best accessible one and declines otherwise. |
