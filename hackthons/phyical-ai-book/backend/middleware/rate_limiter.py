import time
from typing import Dict, Optional
from fastapi import HTTPException, Request
from datetime import datetime, timedelta

# Handle relative imports for direct execution
try:
    from ..utils.logger import log_warning, log_info
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))
    from utils.logger import log_warning, log_info

class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}  # IP -> list of timestamps

    def is_allowed(self, request: Request) -> bool:
        """
        Check if a request from the given IP is allowed based on rate limits
        """
        client_ip = self._get_client_ip(request)

        now = time.time()

        # Clean old requests (older than 1 minute)
        if client_ip in self.requests:
            self.requests[client_ip] = [
                timestamp for timestamp in self.requests[client_ip]
                if now - timestamp < 60  # 60 seconds = 1 minute
            ]
        else:
            self.requests[client_ip] = []

        # Check if the client has exceeded the rate limit
        current_requests = len(self.requests[client_ip])
        is_allowed = current_requests < self.requests_per_minute

        if is_allowed:
            # Record this request
            self.requests[client_ip].append(now)

        return is_allowed

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from the request, considering proxies
        """
        # Check for forwarded-for header first (common in proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # The first IP in the list is usually the real client IP
            return forwarded_for.split(",")[0].strip()

        # Check for real-ip header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    def get_remaining_requests(self, request: Request) -> int:
        """
        Get the number of remaining requests for the client
        """
        client_ip = self._get_client_ip(request)

        if client_ip not in self.requests:
            return self.requests_per_minute

        now = time.time()
        # Clean old requests
        self.requests[client_ip] = [
            timestamp for timestamp in self.requests[client_ip]
            if now - timestamp < 60
        ]

        return max(0, self.requests_per_minute - len(self.requests[client_ip]))

# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=10)

async def check_rate_limit(request: Request):
    """
    FastAPI dependency to check rate limits
    """
    if not rate_limiter.is_allowed(request):
        remaining = rate_limiter.get_remaining_requests(request)
        retry_after = 60  # Reset after 1 minute

        log_warning("Rate limit exceeded", extra={
            "client_ip": rate_limiter._get_client_ip(request),
            "requests_per_minute": rate_limiter.requests_per_minute
        })

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit is {rate_limiter.requests_per_minute} per minute.",
                "retry_after_seconds": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )

    return True