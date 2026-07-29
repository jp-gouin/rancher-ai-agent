"""
Integration-style unit tests for RBAC filtering inside build_agent.

Verifies that build_agent loads only the agents Rancher says the user can access,
and fails closed when nothing/nobody is accessible.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.services.agent.factory import build_agent, _load_agents_for_user
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


# ---------------------------------------------------------------------------
# _load_agents_for_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_agents_scoped_to_user():
    """Returns exactly what Rancher reports as accessible."""
    ws = _websocket()
    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.accessible_agent_configs",
               AsyncMock(return_value=[_agent("rancher")])):
        result = await _load_agents_for_user(ws)
    assert [c.name for c in result] == ["rancher"]


@pytest.mark.asyncio
async def test_load_agents_rbac_error_raises():
    """If Rancher can't resolve access, fail closed."""
    from app.services.rbac import RBACError
    ws = _websocket()
    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.accessible_agent_configs",
               AsyncMock(side_effect=RBACError("down"))):
        with pytest.raises(NoAgentAvailableError):
            await _load_agents_for_user(ws)


@pytest.mark.asyncio
async def test_load_agents_no_token_raises(monkeypatch):
    """If there's no session token, fail closed."""
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)
    ws = _websocket()
    ws.cookies = {}
    with patch("app.services.agent.factory.rbac_enabled", return_value=True):
        with pytest.raises(NoAgentAvailableError):
            await _load_agents_for_user(ws)


@pytest.mark.asyncio
async def test_load_agents_rbac_disabled_uses_full_roster():
    """When RBAC is disabled, the full enabled roster is loaded from CRDs."""
    ws = _websocket()
    with patch("app.services.agent.factory.rbac_enabled", return_value=False), \
         patch("app.services.agent.factory.load_agent_configs",
               return_value=[_agent("rancher"), _agent("security")]):
        result = await _load_agents_for_user(ws)
    assert {c.name for c in result} == {"rancher", "security"}


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_agent_raises_when_no_accessible_agents():
    ws = _websocket()
    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.accessible_agent_configs", AsyncMock(return_value=[])):
        with pytest.raises(NoAgentAvailableError):
            await build_agent(MagicMock(), ws)


@pytest.mark.asyncio
@patch("app.services.agent.factory.create_supervisor_agent")
@patch("app.services.agent.factory.create_child_agent")
@patch("app.services.agent.factory._load_mcp_tools")
async def test_build_agent_only_builds_accessible_agents(
    mock_load_tools, mock_create_child, mock_create_parent
):
    """build_agent must only construct the agents Rancher returned."""
    mock_load_tools.return_value = [MagicMock()]
    mock_create_child.return_value = MagicMock()
    mock_create_parent.return_value = MagicMock()

    ws = _websocket()

    with patch("app.services.agent.factory.rbac_enabled", return_value=True), \
         patch("app.services.agent.factory.accessible_agent_configs",
               AsyncMock(return_value=[_agent("rancher"), _agent("fleet")])):
        _, agents_metadata = await build_agent(MagicMock(), ws)

    built_names = {m["name"] for m in agents_metadata}
    assert built_names == {"rancher", "fleet"}
    loaded_for = {call.args[0].name for call in mock_load_tools.call_args_list}
    assert "security" not in loaded_for
