import time
import logging
from fastapi import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("bottrainer")


async def log_requests(request: Request, call_next):
    """
    Middleware that logs every API request.
    Tracks method, path, status code, and response time.
    """
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration_ms:.0f}ms)"
    )

    return response