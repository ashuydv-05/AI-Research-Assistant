from fastapi import APIRouter, Depends, Request
from loguru import logger
from src.api.dependencies import get_workflow
from src.agent.workflow import MultiAgentWorkflow
from src.api.models import HealthCheck
from src.config.clients import get_es_client, get_qdrant_client
from src.config.settings import settings


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthCheck)
async def health_check(
    workflow: MultiAgentWorkflow = Depends(get_workflow),
) -> HealthCheck:
    components = {"workflow": "healthy" if workflow else "unhealthy"}
    try:
        if workflow and hasattr(workflow, "app"):
            components["workflow_graph"] = "healthy"
        else:
            components["workflow_graph"] = "not_configured"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        components["workflow"] = "unhealthy"

    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant.collection)
        points = getattr(info, "points_count", 0) or 0
        components["qdrant"] = "healthy" if points > 0 else "empty"
        components["qdrant_points"] = str(points)
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        components["qdrant"] = "unhealthy"

    try:
        es = get_es_client()
        if es is None:
            components["elasticsearch"] = "not_configured"
        else:
            count = es.count(index=settings.elasticsearch.index).get("count", 0)
            components["elasticsearch"] = "healthy" if count > 0 else "empty"
            components["elasticsearch_documents"] = str(count)
    except Exception as e:
        logger.warning(f"Elasticsearch health check failed: {e}")
        components["elasticsearch"] = "unhealthy"

    qdrant_ok = components.get("qdrant") == "healthy"
    elasticsearch_ok = components.get("elasticsearch") == "healthy"
    workflow_ok = components.get("workflow") == "healthy"
    return HealthCheck(
        status="healthy" if qdrant_ok and elasticsearch_ok and workflow_ok else "degraded",
        version="2.0.0",
        components=components,
    )


@router.get("/ready")
async def readiness_check(request: Request) -> dict:
    try:
        workflow = request.app.state.workflow
        if workflow is None:
            return {"ready": False, "reason": "Workflow not initialized"}
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/live")
async def liveness_check() -> dict:
    return {"alive": True}
