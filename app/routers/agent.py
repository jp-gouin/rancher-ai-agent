import logging
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from ..services.auth import get_user_id_from_request
from ..services.rbac import load_agent_configs_for_user, PermissionsError

router = APIRouter(prefix="/v1/api", tags=["agent"])

@router.get("/health")
async def health():
    """
    Liveness probe endpoint to verify the HTTP service is running.
    Returns 200 OK if the service is responding.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy"}
    )

@router.get("/readiness")
async def readiness(request: Request):
    """
    Readiness probe endpoint to verify the agent is ready.
    Checks:
    - Memory manager is initialized
    - FastAPI startup is complete
    """
    try:        
        # Check memory manager is initialized
        if not hasattr(request.app, 'memory_manager'):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Memory manager not initialized"}
            )

        # Check if startup is complete
        if not getattr(request.app.state, 'ready', False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Application startup not complete"}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": "Agent is ready"}
        )

    except Exception as e:
        logging.error(f"Readiness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )
@router.get("/agents/available")
async def list_available_agents(request: Request):
    """
    Return the agents the current user is permitted to access.

    Returns objects shaped for the UI ``Agent`` type
    (``name``, ``displayName``, ``description``, ``status``). Live health/OAuth
    status continues to arrive as an overlay via the WebSocket ``chat-metadata``.

    Fails closed: if RBAC is enabled and permissions cannot be resolved, no
    agents are returned and an error is surfaced.
    """
    user_id = await get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    try:
        configs = load_agent_configs_for_user(user_id)
    except PermissionsError as e:
        logging.error(f"Failed to resolve permissions for user '{user_id}': {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Unable to verify your permissions with Rancher"},
        )

    agents = [
        {
            "name": c.name,
            "displayName": c.displayName or c.name,
            "description": c.description,
            "status": "ready" if c.ready else "error",
        }
        for c in configs
    ]
    return JSONResponse(status_code=status.HTTP_200_OK, content=agents)
