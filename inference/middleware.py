import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            '%s %s %d %.2fms',
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        response.headers['X-Process-Time-Ms'] = f'{latency_ms:.2f}'
        return response
