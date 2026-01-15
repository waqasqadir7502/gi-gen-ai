from fastapi import APIRouter
from ..clients.cohere_client import cohere_client
from ..clients.qdrant_client import qdrant_client
from ..config import config

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    """
    Health check endpoint that verifies all services are accessible
    """
    try:
        # Test Cohere connection by attempting to embed a simple text
        test_embedding = cohere_client.embed(["health check"], input_type="classification")
        cohere_ok = len(test_embedding) > 0 if test_embedding else False

        # Test Qdrant connection by getting collection info
        collection_info = qdrant_client.get_collection_info()
        qdrant_ok = collection_info is not None

        # Test Neon database connection if configured
        from ..utils.db_connection import db_manager
        try:
            neon_ok = db_manager.test_connection()
        except:
            neon_ok = False  # If there's an error testing connection, treat as not OK

        # Overall status
        all_ok = cohere_ok and qdrant_ok and (neon_ok if config.DATABASE_URL else True)

        return {
            "status": "healthy" if all_ok else "degraded",
            "details": {
                "cohere_connection": "ok" if cohere_ok else "failed",
                "qdrant_connection": "ok" if qdrant_ok else "failed",
                "neon_database_connection": "ok" if neon_ok else ("not configured" if not config.DATABASE_URL else "failed"),
                "collection_name": config.COLLECTION_NAME,
                "vector_size": config.VECTOR_SIZE
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": {
                "cohere_connection": "unknown",
                "qdrant_connection": "unknown"
            }
        }

@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint
    """
    return {"status": "ready"}