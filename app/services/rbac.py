"""
RBAC — filter AI agents using native Rancher/Kubernetes authorization.

Rather than re-implementing role matching, access is delegated to
Kubernetes authorization via a ``SubjectAccessReview``: an agent is visible to a
user iff that user is allowed to ``get`` the corresponding ``AIAgentConfig``
resource. Access is therefore configured with ordinary Rancher GlobalRole rules
scoped by ``resourceNames`` — no custom permission model, no admin-bypass logic
(cluster admins are allowed to get everything and naturally see all agents).

Note: Kubernetes RBAC ``resourceNames`` constrains ``get`` (not ``list``/
``watch``), so filtering must happen server-side, one agent at a time — which is
exactly what the ``/agents/available`` endpoint and ``build_agent`` do.
"""

import logging
import os

from kubernetes import client
from kubernetes.client.rest import ApiException

from .agent.loader import AgentConfig, load_agent_configs, _load_k8s_config, NAMESPACE, CRD_GROUP

AGENT_RESOURCE = "aiagentconfigs"

# Feature flag. When disabled, filtering is skipped and every authenticated user
# receives the full agent list (legacy behavior). Evaluated once at import time.
RBAC_ENABLED = os.environ.get("RBAC_ENABLED", "true").lower() == "true"


def rbac_enabled() -> bool:
    """Return whether RBAC agent filtering is enabled for this deployment."""
    return RBAC_ENABLED


def user_can_access_agent(user_id: str, agent_name: str) -> bool:
    """
    Return True if *user_id* is allowed to ``get`` the named ``AIAgentConfig``.

    The decision is delegated to Rancher/Kubernetes authorization through a
    SubjectAccessReview (run with the agent service account). Fails closed:
    any error resolving the decision yields ``False``.
    """
    if not user_id:
        return False

    try:
        _load_k8s_config()
        auth_api = client.AuthorizationV1Api()
        sar = client.V1SubjectAccessReview(
            spec=client.V1SubjectAccessReviewSpec(
                user=user_id,
                resource_attributes=client.V1ResourceAttributes(
                    verb="get",
                    group=CRD_GROUP,
                    resource=AGENT_RESOURCE,
                    name=agent_name,
                    namespace=NAMESPACE,
                ),
            )
        )
        allowed = bool(auth_api.create_subject_access_review(sar).status.allowed)
        logging.debug(
            "SAR: user '%s' get %s/%s -> %s", user_id, AGENT_RESOURCE, agent_name, allowed
        )
        return allowed
    except ApiException as e:
        logging.error("SubjectAccessReview failed for user '%s', agent '%s': %s", user_id, agent_name, e)
        return False
    except Exception as e:
        logging.error("SubjectAccessReview error for user '%s', agent '%s': %s", user_id, agent_name, e)
        return False


def filter_agent_configs(user_id: str, agent_configs: list[AgentConfig]) -> list[AgentConfig]:
    """Return only the agents *user_id* is allowed to access (via SubjectAccessReview)."""
    return [cfg for cfg in agent_configs if user_can_access_agent(user_id, cfg.name)]


def load_agent_configs_for_user(user_id: str) -> list[AgentConfig]:
    """
    Load enabled agent configs filtered to those *user_id* may access.

    When RBAC is disabled, returns all enabled configs unchanged. Otherwise
    keeps only the agents the user is authorized to ``get`` (cluster admins see
    all). Fails closed per agent: an agent is excluded if its access decision
    cannot be resolved.
    """
    configs = load_agent_configs()
    if not rbac_enabled():
        return configs

    filtered = filter_agent_configs(user_id, configs)
    logging.info(
        "RBAC: user '%s' can access %d of %d agent(s)",
        user_id, len(filtered), len(configs),
    )
    return filtered
