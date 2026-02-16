"""
In-process Service Registry

This package provides:
1. InMemoryRegistry — dict-backed registry for the heartbeat process
2. ServiceRegistryClient — HTTP client for querying the registry
3. start_registry_server — launches an HTTP query API in a daemon thread
"""

from .service_registry import (
    InMemoryRegistry,
    ServiceRegistryClient,
    ServiceInfo,
    ServiceStatus,
    start_registry_server,
)

__version__ = '0.1.0'
__all__ = [
    'InMemoryRegistry',
    'ServiceRegistryClient',
    'ServiceInfo',
    'ServiceStatus',
    'start_registry_server',
]
