import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("app.requests")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Structured request logging middleware.

    Logs each request with method, path, status code, and duration
    using Python's logging module with a JSON-compatible format.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            '{"method": "%s", "path": "%s", "status": %d, "duration_ms": %.2f, "client": "%s"}',
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "unknown",
        )

        return response
