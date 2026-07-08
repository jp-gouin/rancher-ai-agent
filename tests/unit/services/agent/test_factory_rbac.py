"""
Integration-style unit tests for RBAC filtering inside build_agent.

Verifies that build_agent scopes the agent roster to what the requesting user is
authorized to access, and fails closed when nobody/nothing is accessible.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.services.agent.factory import build_agent, _filter_agents_for_user
from app.services.agent._constants import NoAgentAvailableError
from app.services.agent.loader import AgentConfig


def _agent(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        displayName=name,
        description=name,
        system_prompt="prompt",
        mcp_url="",
    )


def _websocket():
    ws = MagicMock()
    ws.cookies = {"R_SESS": "token"}
    mm = MagicMock()
    mm.get_checkpointer.return_value = MagicMock()
    ws.app.memory_manager = mm
    return ws


def _allow(*names):
    """A filter_agent_configs stand-in that keeps only agents in *names*."""
    allowed = set(names)
    return lambda user_id, configs: [c for c in configs if c.name in allowed]


@pytest.mark.asyncio
async def test_filter_agents_for_user_scopes_by_permission():
    """User only gets agents they are authorized to access."""
    configs = [_agent("rancher"), _agent("security")]
    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.get_user_id_from_websocket", AsyncMock(return_value="u-bob")), \
         patch("app.services.agent.factory.filter_agent_configs", side_effect=_allow("rancher")):
        result = await _filter_agents_for_user(configs, ws)

    assert [c.name for c in result] == ["rancher"]


@pytest.mark.asyncio
async def test_filter_agents_for_user_admin_gets_all():
    """An admin (authorized for everything) receives the full list."""
    configs = [_agent("rancher"), _agent("security")]
    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.get_user_id_from_websocket", AsyncMock(return_value="u-admin")), \
         patch("app.services.agent.factory.filter_agent_configs", side_effect=_allow("rancher", "security")):
        result = await _filter_agents_for_user(configs, ws)

    assert {c.name for c in result} == {"rancher", "security"}


@pytest.mark.asyncio
async def test_filter_agents_for_user_no_accessible_agents_raises():
    """A user with no accessible agents gets a graceful empty-state error."""
    configs = [_agent("security")]
    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.get_user_id_from_websocket", AsyncMock(return_value="u-bob")), \
         patch("app.services.agent.factory.filter_agent_configs", side_effect=_allow()):
        with pytest.raises(NoAgentAvailableError):
            await _filter_agents_for_user(configs, ws)


@pytest.mark.asyncio
async def test_filter_agents_for_user_unknown_identity_raises():
    """If the user's identity cannot be resolved, fail closed."""
    configs = [_agent("rancher")]
    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.get_user_id_from_websocket", AsyncMock(return_value=None)):
        with pytest.raises(NoAgentAvailableError):
            await _filter_agents_for_user(configs, ws)


@pytest.mark.asyncio
async def test_filter_agents_for_user_rbac_disabled_passthrough():
    """When RBAC is disabled, all configs pass through untouched."""
    configs = [_agent("rancher"), _agent("security")]
    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=False):
        result = await _filter_agents_for_user(configs, ws)

    assert result == configs


@pytest.mark.asyncio
@patch("app.services.agent.factory.create_supervisor_agent")
@patch("app.services.agent.factory.create_child_agent")
@patch("app.services.agent.factory._load_mcp_tools")
@patch("app.services.agent.factory.load_agent_configs")
async def test_build_agent_only_builds_accessible_agents(
    mock_load_configs, mock_load_tools, mock_create_child, mock_create_parent
):
    """build_agent must not construct child agents the user cannot access."""
    mock_load_configs.return_value = [_agent("rancher"), _agent("fleet"), _agent("security")]
    mock_load_tools.return_value = [MagicMock()]
    mock_create_child.return_value = MagicMock()
    mock_create_parent.return_value = MagicMock()

    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.get_user_id_from_websocket", AsyncMock(return_value="u-bob")), \
         patch("app.services.agent.factory.filter_agent_configs", side_effect=_allow("rancher", "fleet")):
        _, agents_metadata = await build_agent(MagicMock(), ws)

    built_names = {m["name"] for m in agents_metadata}
    assert built_names == {"rancher", "fleet"}
    # MCP tools were never loaded for the restricted agent.
    loaded_for = {call.args[0].name for call in mock_load_tools.call_args_list}
    assert "security" not in loaded_for
