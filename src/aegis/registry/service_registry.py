#!/usr/bin/env python3
"""
In-process Service Registry and Health Tracking

This module provides:
- InMemoryRegistry: a dict-backed registry for use within the heartbeat process
- RegistryHTTPHandler: HTTP request handler exposing read-only query endpoints
- start_registry_server: launches a ThreadingHTTPServer in a daemon thread
- ServiceRegistryClient: thin HTTP client matching the query API shape
"""

import json
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class ServiceInfo:
    """Service information dataclass"""
    service_id: str
    host: str
    port: int
    service_type: str
    status: str = ServiceStatus.HEALTHY.value
    last_seen: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = time.time()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInfo':
        """Create from dictionary."""
        data['last_seen'] = float(data['last_seen'])
        data['port'] = int(data['port'])
        return cls(**data)


# ---------------------------------------------------------------------------
# In-memory registry (lives inside the heartbeat process)
# ---------------------------------------------------------------------------

class InMemoryRegistry:
    """Thread-safe, dict-backed service registry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._services: Dict[str, ServiceInfo] = {}
        self._active: set[str] = set()
        self._types: Dict[str, set[str]] = {}  # type -> set of service_ids

    def register_service(self, service_info: ServiceInfo) -> bool:
        with self._lock:
            sid = service_info.service_id
            self._services[sid] = service_info
            self._active.add(sid)
            self._types.setdefault(service_info.service_type, set()).add(sid)
        return True

    def deregister_service(self, service_id: str) -> bool:
        with self._lock:
            info = self._services.pop(service_id, None)
            if info is None:
                return False
            self._active.discard(service_id)
            type_set = self._types.get(info.service_type)
            if type_set:
                type_set.discard(service_id)
            return True

    def update_health(self, service_id: str, status: ServiceStatus,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        with self._lock:
            info = self._services.get(service_id)
            if info is None:
                return False
            info.status = status.value
            info.last_seen = time.time()
            if metadata:
                info.metadata.update(metadata)
        return True

    def heartbeat(self, service_id: str) -> bool:
        with self._lock:
            info = self._services.get(service_id)
            if info is None:
                return False
            info.last_seen = time.time()
        return True

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        with self._lock:
            return self._services.get(service_id)

    def list_services(self, service_type: Optional[str] = None,
                      status_filter: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        with self._lock:
            if service_type:
                ids = self._types.get(service_type, set())
            else:
                ids = self._active
            results = []
            for sid in ids:
                info = self._services.get(sid)
                if info is None:
                    continue
                if status_filter is not None and info.status != status_filter.value:
                    continue
                results.append(info)
        return results

    def get_healthy_services(self, service_type: Optional[str] = None,
                             timeout_seconds: int = 30) -> List[ServiceInfo]:
        services = self.list_services(service_type=service_type)
        now = time.time()
        return [
            s for s in services
            if s.status == ServiceStatus.HEALTHY.value
            and (now - s.last_seen) < timeout_seconds
        ]

    def get_service_count(self, service_type: Optional[str] = None) -> int:
        with self._lock:
            if service_type:
                return len(self._types.get(service_type, set()))
            return len(self._active)

    def mark_unhealthy_services(self, timeout_seconds: int = 30) -> int:
        services = self.list_services()
        now = time.time()
        count = 0
        for s in services:
            if s.status != ServiceStatus.HEALTHY.value:
                continue
            if (now - s.last_seen) > timeout_seconds:
                if self.update_health(s.service_id, ServiceStatus.UNHEALTHY):
                    count += 1
        return count


# ---------------------------------------------------------------------------
# HTTP handler (read-only query API served from the heartbeat process)
# ---------------------------------------------------------------------------

def _make_handler(registry: InMemoryRegistry):
    """Create a handler class bound to the given registry instance."""

    class RegistryHTTPHandler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            # Silence default stderr logging
            pass

        def _json_response(self, data: Any, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/")
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/services":
                stype = qs.get("type", [None])[0]
                status_str = qs.get("status", [None])[0]
                status_filter = ServiceStatus(status_str) if status_str else None
                services = registry.list_services(
                    service_type=stype, status_filter=status_filter,
                )
                self._json_response([s.to_dict() for s in services])

            elif path == "/services/healthy":
                stype = qs.get("type", [None])[0]
                timeout = int(qs.get("timeout", [30])[0])
                services = registry.get_healthy_services(
                    service_type=stype, timeout_seconds=timeout,
                )
                self._json_response([s.to_dict() for s in services])

            elif path == "/services/count":
                stype = qs.get("type", [None])[0]
                count = registry.get_service_count(service_type=stype)
                self._json_response({"count": count})

            elif path.startswith("/services/"):
                service_id = path[len("/services/"):]
                info = registry.get_service(service_id)
                if info:
                    self._json_response(info.to_dict())
                else:
                    self._json_response({"error": "not found"}, status=404)

            else:
                self._json_response({"error": "not found"}, status=404)

    return RegistryHTTPHandler


def start_registry_server(
    registry: InMemoryRegistry,
    host: str = "0.0.0.0",
    port: int = 8471,
) -> ThreadingHTTPServer:
    """Start a ThreadingHTTPServer in a daemon thread and return the server."""
    handler = _make_handler(registry)
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# HTTP client (used by external callers to query the registry)
# ---------------------------------------------------------------------------

class ServiceRegistryClient:
    """Thin HTTP client that queries the registry HTTP API."""

    def __init__(self, host: str = "localhost", port: int = 8471):
        self._base = f"http://{host}:{port}"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(self, path: str) -> Any:
        url = f"{self._base}{path}"
        with self._opener.open(url, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        try:
            data = self._get(f"/services/{service_id}")
            if "error" in data:
                return None
            return ServiceInfo.from_dict(data)
        except (urllib.error.URLError, OSError):
            return None

    def list_services(self, service_type: Optional[str] = None,
                      status_filter: Optional[str] = None) -> List[ServiceInfo]:
        params = {}
        if service_type:
            params["type"] = service_type
        if status_filter:
            params["status"] = status_filter
        qs = urllib.parse.urlencode(params)
        path = f"/services?{qs}" if qs else "/services"
        try:
            data = self._get(path)
            return [ServiceInfo.from_dict(d) for d in data]
        except (urllib.error.URLError, OSError):
            return []

    def get_healthy_services(self, service_type: Optional[str] = None,
                             timeout_seconds: int = 30) -> List[ServiceInfo]:
        params: dict[str, str] = {}
        if service_type:
            params["type"] = service_type
        params["timeout"] = str(timeout_seconds)
        qs = urllib.parse.urlencode(params)
        path = f"/services/healthy?{qs}"
        try:
            data = self._get(path)
            return [ServiceInfo.from_dict(d) for d in data]
        except (urllib.error.URLError, OSError):
            return []

    def get_service_count(self, service_type: Optional[str] = None) -> int:
        params = {}
        if service_type:
            params["type"] = service_type
        qs = urllib.parse.urlencode(params)
        path = f"/services/count?{qs}" if qs else "/services/count"
        try:
            data = self._get(path)
            return data.get("count", 0)
        except (urllib.error.URLError, OSError):
            return 0
