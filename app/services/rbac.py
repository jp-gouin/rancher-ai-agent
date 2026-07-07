"""
RBAC — filter AI agents by the requesting user's Rancher Global Permissions.
"""

import logging
import os

from kubernetes import client
from kubernetes.client.rest import ApiException

from .agent.loader import AgentConfig, load_agent_configs, _load_k8s_config

# Global role that grants full admin access; bypasses all per-agent restrictions.
ADMIN_GLOBAL_ROLE = "admin"

# Rancher management API used to resolve global permissions.
MGMT_GROUP = "management.cattle.io"
MGMT_VERSION = "v3"
GLOBAL_ROLE_BINDINGS_PLURAL = "globalrolebindings"

# Feature flag. When disabled, filtering is skipped and every authenticated user
# receives the full agent list (legacy behavior). Evaluated once at import time.
RBAC_ENABLED = os.environ.get("RBAC_ENABLED", "true").lower() == "true"


class PermissionsError(Exception):
    """Raised when a user's Global Permissions cannot be resolved."""
    pass


def rbac_enabled() -> bool:
    """Return whether RBAC agent filtering is enabled for this deployment."""
    return RBAC_ENABLED


def _list_global_role_bindings() -> list[dict]:
    """List all GlobalRoleBinding objects via the Kubernetes API."""
    _load_k8s_config()
    api = client.CustomObjectsApi()
    response = api.list_cluster_custom_object(
        group=MGMT_GROUP,
        version=MGMT_VERSION,
        plural=GLOBAL_ROLE_BINDINGS_PLURAL,
    )
    return response.get("items", [])


def get_global_permissions(user_id: str) -> set[str]:
    """
    Resolve the set of Global Permission names held by a user.

    Args:
        user_id: The Rancher user id (e.g. ``u-abcde``) as returned by the
            Rancher users API.

    Returns:
        The set of ``globalRoleName`` values bound to the user.

    Raises:
        PermissionsError: If the bindings cannot be listed (fail closed).
    """
    if not user_id:
        raise PermissionsError("Cannot resolve permissions for an empty user id")

    try:
        bindings = _list_global_role_bindings()
    except ApiException as e:
        raise PermissionsError(
            f"Failed to list GlobalRoleBindings from the Rancher API: {e}"
        ) from e
    except Exception as e:
        raise PermissionsError(
            f"Failed to resolve global permissions for user '{user_id}': {e}"
        ) from e

    perms = {
        binding.get("globalRoleName")
        for binding in bindings
        if binding.get("userName") == user_id and binding.get("globalRoleName")
    }

    logging.debug("Resolved %d global permission(s) for user '%s'", len(perms), user_id)
    return perms


def is_admin(permissions: set[str]) -> bool:
    """Return True if the permission set includes the Administrator role."""
    return ADMIN_GLOBAL_ROLE in permissions


def can_access_agent(agent_config: AgentConfig, permissions: set[str]) -> bool:
    """
    Determine whether a user with *permissions* can access *agent_config*.

    Access is granted when any of the following holds:
    * the user is an Administrator (bypass);
    * the agent has no ``required_permissions`` (unrestricted); or
    * the user holds at least one of the agent's required permissions (OR logic).
    """
    if is_admin(permissions):
        return True
    required = agent_config.required_permissions
    if not required:
        return True
    return bool(set(required) & permissions)


def filter_agent_configs(
    agent_configs: list[AgentConfig],
    permissions: set[str],
) -> list[AgentConfig]:
    """Return only the agent configs accessible to a user with *permissions*."""
    return [cfg for cfg in agent_configs if can_access_agent(cfg, permissions)]


def load_agent_configs_for_user(user_id: str) -> list[AgentConfig]:
    """
    Load enabled agent configs filtered by the user's Rancher Global Permissions.

    When RBAC is disabled, returns all enabled configs unchanged. Otherwise
    resolves the user's global permissions and keeps only the agents they may
    access (Administrators receive the full list).

    Raises:
        PermissionsError: If RBAC is enabled and permissions cannot be resolved
            (fail closed — callers should surface an error and expose no agents).
    """
    configs = load_agent_configs()
    if not rbac_enabled():
        return configs

    permissions = get_global_permissions(user_id)
    filtered = filter_agent_configs(configs, permissions)
    logging.info(
        "RBAC: user '%s' can access %d of %d agent(s)",
        user_id, len(filtered), len(configs),
    )
    return filtered
