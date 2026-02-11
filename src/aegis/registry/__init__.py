"""
Redis-based Service Registry

This package provides tools for:
1. Service registry & health tracking (using hashes + sets)
2. Centralized job queue with lists/streams (coming soon)
3. Result collection & async API (coming soon)
"""

from .service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceStatus
)

__version__ = '0.1.0'
__all__ = [
    'ServiceRegistry',
    'ServiceInfo',
    'ServiceStatus'
]
