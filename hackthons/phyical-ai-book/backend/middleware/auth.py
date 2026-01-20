from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

# Handle relative imports for direct execution
try:
    from ..config import config
    from ..utils.logger import log_info, log_error
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))
    from config import config
    from utils.logger import log_info, log_error

class APIKeyAuth:
    def __init__(self):
        self.backend_api_key = config.BACKEND_API_KEY

    async def authenticate(self, request: Request):
        """
        Authenticate the request using the X-API-Key header
        """
        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key != self.backend_api_key:
            log_error("Unauthorized access attempt", extra={
                "client_host": request.client.host,
                "request_path": request.url.path
            })
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )

        # Log successful authentication
        log_info("Successful API authentication", extra={
            "client_host": request.client.host,
            "request_path": request.url.path
        })

        return True

# Create a singleton instance
auth_handler = APIKeyAuth()

# HTTP Bearer scheme for documentation
security_scheme = HTTPBearer()

async def verify_api_key(request: Request):
    """
    Dependency to verify API key for protected endpoints
    """
    return await auth_handler.authenticate(request)