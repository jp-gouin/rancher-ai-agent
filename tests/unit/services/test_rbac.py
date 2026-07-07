"""
Unit tests for the RBAC helpers in app/services/rbac.py.

Covers Global Permission resolution, admin bypass, OR-logic agent filtering,
and the user-scoped ``load_agent_configs_for_user`` entry point.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.rbac import (
    ADMIN_GLOBAL_ROLE,
    PermissionsError,
    can_access_agent,
    filter_agent_configs,
    get_global_permissions,
    is_admin,
    load_agent_configs_for_user,
    AgentConfig,
)


def _agent(name: str, required: list[str] | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        displayName=name,
        description="",
        system_prompt="",
        mcp_url="",
        required_permissions=required or [],
    )


# ---------------------------------------------------------------------------
# is_admin / can_access_agent / filter_agent_configs
# ---------------------------------------------------------------------------

def test_is_admin_true():
    assert is_admin({ADMIN_GLOBAL_ROLE, "user"}) is True


def test_is_admin_false():
    assert is_admin({"user", "restricted-admin"}) is False


def test_admin_bypasses_all_restrictions():
    agent = _agent("security", required=["security-role"])
    assert can_access_agent(agent, {ADMIN_GLOBAL_ROLE}) is True


def test_unrestricted_agent_accessible_to_everyone():
    agent = _agent("rancher", required=[])
    assert can_access_agent(agent, set()) is True
    assert can_access_agent(agent, {"user"}) is True


def test_or_logic_grants_access_with_any_matching_permission():
    agent = _agent("ops", required=["cluster-owner", "restricted-admin"])
    assert can_access_agent(agent, {"restricted-admin"}) is True


def test_restricted_agent_denied_without_matching_permission():
    agent = _agent("security", required=["security-role"])
    assert can_access_agent(agent, {"user"}) is False


def test_filter_agent_configs_non_admin():
    agents = [
        _agent("rancher", required=[]),
        _agent("security", required=["security-role"]),
        _agent("ops", required=["cluster-owner", "user"]),
    ]
    result = filter_agent_configs(agents, {"user"})
    assert {a.name for a in result} == {"rancher", "ops"}


def test_filter_agent_configs_admin_gets_all():
    agents = [_agent("rancher", required=[]), _agent("security", required=["security-role"])]
    result = filter_agent_configs(agents, {ADMIN_GLOBAL_ROLE})
    assert len(result) == 2


def test_filter_agent_configs_no_permissions_only_unrestricted():
    agents = [_agent("rancher", required=[]), _agent("security", required=["security-role"])]
    result = filter_agent_configs(agents, set())
    assert [a.name for a in result] == ["rancher"]


# ---------------------------------------------------------------------------
# get_global_permissions
# ---------------------------------------------------------------------------

def _binding(user: str, role: str) -> dict:
    return {"userName": user, "globalRoleName": role}


@patch("app.services.rbac._load_k8s_config")
@patch("app.services.rbac.client")
def test_get_global_permissions_resolves_user_roles(mock_client, _mock_cfg):
    api = MagicMock()
    mock_client.CustomObjectsApi.return_value = api
    api.list_cluster_custom_object.return_value = {
        "items": [
            _binding("u-alice", "admin"),
            _binding("u-alice", "user"),
            _binding("u-bob", "user"),
            {"userName": "u-alice"},  # missing globalRoleName -> ignored
        ]
    }

    assert get_global_permissions("u-alice") == {"admin", "user"}


@patch("app.services.rbac._load_k8s_config")
@patch("app.services.rbac.client")
def test_get_global_permissions_empty_for_unbound_user(mock_client, _mock_cfg):
    api = MagicMock()
    mock_client.CustomObjectsApi.return_value = api
    api.list_cluster_custom_object.return_value = {"items": [_binding("u-bob", "user")]}

    assert get_global_permissions("u-alice") == set()


def test_get_global_permissions_empty_user_id_raises():
    with pytest.raises(PermissionsError):
        get_global_permissions("")


@patch("app.services.rbac._load_k8s_config")
@patch("app.services.rbac.client")
def test_get_global_permissions_fails_closed_on_api_error(mock_client, _mock_cfg):
    api = MagicMock()
    mock_client.CustomObjectsApi.return_value = api
    api.list_cluster_custom_object.side_effect = Exception("boom")

    with pytest.raises(PermissionsError):
        get_global_permissions("u-alice")


# ---------------------------------------------------------------------------
# load_agent_configs_for_user
# ---------------------------------------------------------------------------

@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_filters_by_permission(mock_load):
    mock_load.return_value = [_agent("rancher"), _agent("security", ["security-role"])]

    with patch("app.services.rbac.rbac_enabled", return_value=True), \
         patch("app.services.rbac.get_global_permissions", return_value={"user"}):
        result = load_agent_configs_for_user("u-bob")

    assert [a.name for a in result] == ["rancher"]


@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_admin_gets_all(mock_load):
    mock_load.return_value = [_agent("rancher"), _agent("security", ["security-role"])]

    with patch("app.services.rbac.rbac_enabled", return_value=True), \
         patch("app.services.rbac.get_global_permissions", return_value={"admin"}):
        result = load_agent_configs_for_user("u-admin")

    assert {a.name for a in result} == {"rancher", "security"}


@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_rbac_disabled_returns_all(mock_load):
    mock_load.return_value = [_agent("rancher"), _agent("security", ["security-role"])]

    with patch("app.services.rbac.rbac_enabled", return_value=False):
        result = load_agent_configs_for_user("u-bob")

    assert {a.name for a in result} == {"rancher", "security"}


@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_fails_closed(mock_load):
    mock_load.return_value = [_agent("rancher")]

    with patch("app.services.rbac.rbac_enabled", return_value=True), \
         patch("app.services.rbac.get_global_permissions", side_effect=PermissionsError("down")):
        with pytest.raises(PermissionsError):
            load_agent_configs_for_user("u-bob")
