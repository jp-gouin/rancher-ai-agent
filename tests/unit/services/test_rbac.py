"""
Unit tests for RBAC agent filtering (SubjectAccessReview-based) in app/services/rbac.py.
"""
from unittest.mock import MagicMock, patch

from app.services.rbac import (
    user_can_access_agent,
    filter_agent_configs,
    load_agent_configs_for_user,
)
from app.services.agent.loader import AgentConfig


def _agent(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        displayName=name,
        description="",
        system_prompt="",
        mcp_url="",
    )


def _sar_api(allowed_names: set[str]):
    """A mocked AuthorizationV1Api whose SAR allows only *allowed_names*.

    Real V1SubjectAccessReview/V1ResourceAttributes objects are built by the code
    under test, so ``sar.spec.resource_attributes.name`` reflects the real name.
    """
    def _create_sar(sar):
        name = sar.spec.resource_attributes.name
        resp = MagicMock()
        resp.status.allowed = name in allowed_names
        return resp

    api = MagicMock()
    api.create_subject_access_review.side_effect = _create_sar
    return api


def _patch_sar(allowed_names: set[str]):
    return patch("app.services.rbac.client.AuthorizationV1Api", return_value=_sar_api(allowed_names))


# ---------------------------------------------------------------------------
# user_can_access_agent
# ---------------------------------------------------------------------------

@patch("app.services.rbac._load_k8s_config")
def test_user_can_access_agent_allowed(_cfg):
    with _patch_sar({"rancher"}):
        assert user_can_access_agent("u-bob", "rancher") is True


@patch("app.services.rbac._load_k8s_config")
def test_user_can_access_agent_denied(_cfg):
    with _patch_sar({"rancher"}):
        assert user_can_access_agent("u-bob", "security") is False


def test_user_can_access_agent_empty_user_id():
    assert user_can_access_agent("", "rancher") is False


@patch("app.services.rbac._load_k8s_config")
def test_user_can_access_agent_fails_closed_on_error(_cfg):
    mock_client = MagicMock()
    mock_client.AuthorizationV1Api.return_value.create_subject_access_review.side_effect = Exception("boom")
    with patch("app.services.rbac.client", mock_client):
        assert user_can_access_agent("u-bob", "rancher") is False


# ---------------------------------------------------------------------------
# filter_agent_configs
# ---------------------------------------------------------------------------

@patch("app.services.rbac._load_k8s_config")
def test_filter_agent_configs_keeps_allowed(_cfg):
    agents = [_agent("rancher"), _agent("security"), _agent("fleet")]
    with _patch_sar({"rancher", "fleet"}):
        result = filter_agent_configs("u-bob", agents)
    assert {a.name for a in result} == {"rancher", "fleet"}


# ---------------------------------------------------------------------------
# load_agent_configs_for_user
# ---------------------------------------------------------------------------

@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_filters(mock_load):
    mock_load.return_value = [_agent("rancher"), _agent("security")]
    with patch("app.services.rbac.rbac_enabled", return_value=True), \
         patch("app.services.rbac._load_k8s_config"), \
         _patch_sar({"rancher"}):
        result = load_agent_configs_for_user("u-bob")
    assert [a.name for a in result] == ["rancher"]


@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_rbac_disabled_returns_all(mock_load):
    mock_load.return_value = [_agent("rancher"), _agent("security")]
    with patch("app.services.rbac.rbac_enabled", return_value=False):
        result = load_agent_configs_for_user("u-bob")
    assert {a.name for a in result} == {"rancher", "security"}


@patch("app.services.rbac.load_agent_configs")
def test_load_for_user_no_access_returns_empty(mock_load):
    """When the user can access nothing, an empty list is returned (caller fails closed)."""
    mock_load.return_value = [_agent("security")]
    with patch("app.services.rbac.rbac_enabled", return_value=True), \
         patch("app.services.rbac._load_k8s_config"), \
         _patch_sar(set()):
        result = load_agent_configs_for_user("u-bob")
    assert result == []
