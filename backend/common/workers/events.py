"""
Event emitter for worker-side observability.

Tasks publish status-change events to Redis so the
``GET /jobs/{id}/events`` SSE endpoint (Session 1.4) can push to browser
clients without polling. Abstraction kept narrow — the emitter takes a
channel name, event name, and dict payload, and that's it.

Design choices

- Redis is optional at import time. A :class:`NullEventEmitter` swallows
  events so code paths that don't care about observability (tests,
  workers running without Redis) keep working.
- :class:`RedisPubSubEmitter` lazily imports ``redis`` so the import
  graph doesn't require it when the feature's disabled.
- Envelope format:

      {
        "event":      "job.status.running",
        "ts":         "2026-04-18T12:34:56.789Z",
        "payload":    {...},
        "emitter_version": "1",
      }

  The Session 1.4 SSE polling endpoint doesn't consume these yet; when
  Session 10 (observability) wires the real push path, it subscribes
  to ``orion:events:job:{id}``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)

EMITTER_VERSION = "1"


def event_channel(job_id: Any) -> str:
    """Redis pubsub channel name for a job's event stream."""
    return f"orion:events:job:{job_id}"


def _envelope(event: str, payload: Dict[str, Any]) -> str:
    """Standard envelope; returns a JSON-serializable string."""
    return json.dumps(
        {
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "emitter_version": EMITTER_VERSION,
        },
        default=str,
    )


# ---------------------------------------------------------------------------
# Protocol and implementations
# ---------------------------------------------------------------------------


class EventEmitter(Protocol):
    """Any object that can emit a job event."""

    def emit(self, *, channel: str, event: str, payload: Dict[str, Any]) -> int:
        """Publish *event* + *payload* on *channel*. Returns 1 on success."""
        ...


class NullEventEmitter:
    """Swallow emits; useful for tests and Redis-less dev environments."""

    def __init__(self) -> None:
        self.events: list[Dict[str, Any]] = []

    def emit(self, *, channel: str, event: str, payload: Dict[str, Any]) -> int:
        self.events.append({"channel": channel, "event": event, "payload": payload})
        logger.debug("null-emit channel=%s event=%s", channel, event)
        return 1


class RedisPubSubEmitter:
    """
    Real emitter. Uses ``redis.Redis.publish`` synchronously.

    Parameters
    ----------
    url
        Connection URL; defaults to ``settings.redis_url`` at first use.
    client
        Optional pre-built ``redis.Redis`` instance for tests.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._url = url
        self._client = client
        self._connection_failed = False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._connection_failed:
            raise ConnectionError("Redis unreachable; not retrying within this emitter")
        import redis  # lazy

        url = self._url
        if url is None:
            from src.api.config import settings

            url = settings.redis_url
        try:
            self._client = redis.Redis.from_url(url, socket_timeout=1.0)
            self._client.ping()
        except Exception:
            self._connection_failed = True
            raise
        return self._client

    def emit(self, *, channel: str, event: str, payload: Dict[str, Any]) -> int:
        body = _envelope(event, payload)
        try:
            client = self._get_client()
            return int(client.publish(channel, body))
        except Exception as exc:  # noqa: BLE001 — emitter must not kill a task
            logger.warning(
                "pubsub emit dropped channel=%s event=%s err=%s",
                channel, event, exc,
            )
            return 0
