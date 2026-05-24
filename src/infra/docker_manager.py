"""Docker Compose Manager — application lifecycle for infrastructure services.

Handles:
- Auto-start: `docker compose up -d` on application boot
- Health check: redis-cli PING retry loop until healthy
- Optional shutdown: `docker compose down` on application exit

v4.5.0 §3.2.2: Redis with AOF persistence, appendfsync everysec.
"""
from __future__ import annotations

import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDIS_HEALTH_PING_INTERVAL_SEC: float = 2.0
REDIS_HEALTH_PING_RETRIES: int = 30
REDIS_HEALTH_TOTAL_TIMEOUT_SEC: float = 60.0
DOCKER_COMPOSE_FILE: str = "docker-compose.yml"


# ---------------------------------------------------------------------------
# DockerManager
# ---------------------------------------------------------------------------


class DockerManager:
    """Manages Docker Compose infrastructure lifecycle.

    Parameters
    ----------
    compose_file: str | Path
        Path to docker-compose.yml.
    project_dir: str | Path
        Working directory for docker compose commands.
    redis_host: str
        Redis host for health-check ping (default 'localhost').
    redis_port: int
        Redis port for health-check ping (default 6379).
    """

    def __init__(
        self,
        compose_file: str | Path = DOCKER_COMPOSE_FILE,
        project_dir: str | Path = ".",
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ) -> None:
        self._compose_file = str(compose_file)
        self._project_dir = str(project_dir)
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._started = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Start infrastructure services via docker compose up -d.

        Returns True if all services are healthy, False otherwise.
        """
        trace_id = str(uuid.uuid4())
        logger.info(
            "[%s] DockerManager: starting services via compose file %s",
            trace_id,
            self._compose_file,
        )

        try:
            self._compose_up(trace_id)
        except Exception as exc:
            logger.warning(
                "[%s] DockerManager: compose up failed: %s",
                trace_id,
                exc,
            )
            return False

        healthy = self._wait_for_redis(trace_id)
        if healthy:
            self._started = True
            logger.info(
                "[%s] DockerManager: all services healthy", trace_id
            )
        else:
            logger.warning(
                "[%s] DockerManager: Redis health check failed after "
                "%.1fs — services may not be fully ready",
                trace_id,
                REDIS_HEALTH_TOTAL_TIMEOUT_SEC,
            )

        return healthy

    def stop(self) -> bool:
        """Stop infrastructure services via docker compose down.

        Returns True if successful.
        """
        trace_id = str(uuid.uuid4())
        logger.info(
            "[%s] DockerManager: stopping services", trace_id
        )

        try:
            self._compose_down(trace_id)
            self._started = False
            logger.info(
                "[%s] DockerManager: services stopped", trace_id
            )
            return True
        except Exception as exc:
            logger.warning(
                "[%s] DockerManager: compose down failed: %s",
                trace_id,
                exc,
            )
            return False

    def is_redis_healthy(self) -> bool:
        """Ping Redis to check if it is responsive."""
        try:
            import redis
            client = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            result = client.ping()
            # redis-py may return Awaitable in typed stubs — force bool evaluation
            return bool(result)
        except ImportError:
            return self._ping_redis_via_cli()
        except Exception as exc:
            logger.debug("DockerManager: Redis ping failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Docker Compose helpers
    # ------------------------------------------------------------------ #

    def _compose_up(self, trace_id: str) -> None:
        """Run docker compose up -d."""
        cmd = [
            "docker", "compose",
            "-f", self._compose_file,
            "up", "-d",
            "--wait",
        ]
        logger.debug("[%s] Running: %s", trace_id, " ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=self._project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"docker compose up returned {result.returncode}: {stderr}"
            )

    def _compose_down(self, trace_id: str) -> None:
        """Run docker compose down."""
        cmd = [
            "docker", "compose",
            "-f", self._compose_file,
            "down",
            "--remove-orphans",
        ]
        logger.debug("[%s] Running: %s", trace_id, " ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=self._project_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"docker compose down returned {result.returncode}: {stderr}"
            )

    # ------------------------------------------------------------------ #
    # Redis health check
    # ------------------------------------------------------------------ #

    def _wait_for_redis(self, trace_id: str) -> bool:
        """Poll Redis until healthy or timeout.

        Returns True if Redis becomes healthy, False on timeout.
        """
        deadline = time.monotonic() + REDIS_HEALTH_TOTAL_TIMEOUT_SEC

        for attempt in range(1, REDIS_HEALTH_PING_RETRIES + 1):
            if time.monotonic() > deadline:
                logger.warning(
                    "[%s] DockerManager: Redis health check timed out "
                    "after %d attempts (%.1fs)",
                    trace_id,
                    attempt,
                    REDIS_HEALTH_TOTAL_TIMEOUT_SEC,
                )
                return False

            if self.is_redis_healthy():
                logger.info(
                    "[%s] DockerManager: Redis healthy after %d "
                    "attempt(s) (%.1fs)",
                    trace_id,
                    attempt,
                    time.monotonic() - (deadline - REDIS_HEALTH_TOTAL_TIMEOUT_SEC),
                )
                return True

            logger.debug(
                "[%s] DockerManager: Redis not ready (attempt %d/%d), "
                "retrying in %.1fs",
                trace_id,
                attempt,
                REDIS_HEALTH_PING_RETRIES,
                REDIS_HEALTH_PING_INTERVAL_SEC,
            )
            time.sleep(REDIS_HEALTH_PING_INTERVAL_SEC)

        return False

    def _ping_redis_via_cli(self) -> bool:
        """Fallback: ping Redis via redis-cli when redis-py is not available."""
        try:
            result = subprocess.run(
                [
                    "redis-cli",
                    "-h", self._redis_host,
                    "-p", str(self._redis_port),
                    "PING",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "PONG" in result.stdout
        except Exception as exc:
            logger.debug(
                "DockerManager: redis-cli ping failed: %s", exc
            )
            return False


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_docker_manager: DockerManager | None = None


def get_docker_manager(
    compose_file: str | Path = DOCKER_COMPOSE_FILE,
    project_dir: str | Path = ".",
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> DockerManager:
    """Return a singleton DockerManager, creating it on first call."""
    global _docker_manager
    if _docker_manager is None:
        _docker_manager = DockerManager(
            compose_file=compose_file,
            project_dir=project_dir,
            redis_host=redis_host,
            redis_port=redis_port,
        )
    return _docker_manager
