"""
Unit tests for the user-scoped, RBAC-filtered agent listing endpoint
(``GET /v1/api/agents/available`` in ``app.routers.agent``).
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, status

from app.routers import agent as agent_router
from app.services.agent.loader import AgentConfig
from app.services.rbac import PermissionsError


def _cfg(name, ready=True):
    return AgentConfig(
        name=name,
        displayName=name.capitalize(),
        description=f"{name} desc",
        system_prompt="",
        mcp_url="",
        ready=ready,
    )


@pytest.fixture
def mock_request():
    req = MagicMock()
    req.cookies = {"R_SESS": "token"}
    return req


@pytest.mark.asyncio
async def test_available_agents_unauthorized(mock_request):
    with patch("app.routers.agent.get_user_id_from_request", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await agent_router.list_available_agents(mock_request)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_available_agents_maps_ui_shape(mock_request):
    """The endpoint maps the filtered configs to the UI Agent shape."""
    with patch("app.routers.agent.get_user_id_from_request", AsyncMock(return_value="u-bob")), \
         patch("app.routers.agent.load_agent_configs_for_user", return_value=[_cfg("rancher")]):
        resp = await agent_router.list_available_agents(mock_request)

    assert resp.status_code == status.HTTP_200_OK
    body = json.loads(resp.body)
    assert body == [{
        "name": "rancher",
        "displayName": "Rancher",
        "description": "rancher desc",
        "status": "ready",
    }]


@pytest.mark.asyncio
async def test_available_agents_fails_closed(mock_request):
    """A PermissionsError from the loader surfaces as 503."""
    with patch("app.routers.agent.get_user_id_from_request", AsyncMock(return_value="u-bob")), \
         patch("app.routers.agent.load_agent_configs_for_user", side_effect=PermissionsError("down")):
        resp = await agent_router.list_available_agents(mock_request)

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_available_agents_unready_status(mock_request):
    """An agent whose CRD is not Ready is reported with status 'error'."""
    with patch("app.routers.agent.get_user_id_from_request", AsyncMock(return_value="u-bob")), \
         patch("app.routers.agent.load_agent_configs_for_user", return_value=[_cfg("rancher", ready=False)]):
        resp = await agent_router.list_available_agents(mock_request)

    body = json.loads(resp.body)
    assert body[0]["status"] == "error"
