"""Per-instance heartbeat monitor for vLLM health tracking."""

import sys
import time
import urllib.request
import urllib.error

from .registry import ServiceRegistry, ServiceStatus


def run_heartbeat(
    service_id: str,
    host: str,
    port: int,
    redis_host: str,
    redis_port: int,
    interval: int = 30,
) -> None:
    """Poll a vLLM instance's /health endpoint and update the registry.

    This function runs an infinite loop and is meant to be invoked as a
    standalone subprocess via ``python -m aegis.heartbeat``.
    """
    registry = ServiceRegistry(redis_host=redis_host, redis_port=redis_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{host}:{port}/health"

    last_status: ServiceStatus | None = None

    while True:
        try:
            with opener.open(url, timeout=10) as resp:
                status = ServiceStatus.HEALTHY if resp.status == 200 else ServiceStatus.UNHEALTHY
        except (urllib.error.URLError, OSError):
            status = ServiceStatus.UNHEALTHY

        if status != last_status:
            print(
                f"[heartbeat] {service_id}: {last_status.value if last_status else 'init'}"
                f" -> {status.value}",
                file=sys.stderr,
            )
            last_status = status

        registry.update_health(service_id, status)
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage: python -m aegis.heartbeat SERVICE_ID HOST PORT REDIS_HOST REDIS_PORT [INTERVAL]",
            file=sys.stderr,
        )
        sys.exit(1)

    _service_id = sys.argv[1]
    _host = sys.argv[2]
    _port = int(sys.argv[3])
    _redis_host = sys.argv[4]
    _redis_port = int(sys.argv[5])
    _interval = int(sys.argv[6]) if len(sys.argv) > 6 else 30

    run_heartbeat(_service_id, _host, _port, _redis_host, _redis_port, _interval)
