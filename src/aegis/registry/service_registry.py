#!/usr/bin/env python3
"""
Service Registry and Health Tracking using Redis

This module provides a service registry system using Redis hashes and sets
to track service metadata, health status, and enable service discovery.

Data Structure:
- Hash: service:{service_id} -> {host, port, status, last_seen, metadata}
- Set: services:active -> set of active service_ids
- Set: services:type:{type} -> set of service_ids of a specific type
"""

import redis
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
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
        """Convert to dictionary"""
        data = asdict(self)
        # Convert metadata dict to JSON string for Redis storage
        data['metadata'] = json.dumps(data['metadata'])
        data['last_seen'] = str(data['last_seen'])
        data['port'] = str(data['port'])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInfo':
        """Create from dictionary"""
        if isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        data['last_seen'] = float(data['last_seen'])
        data['port'] = int(data['port'])
        return cls(**data)


class ServiceRegistry:
    """
    Service Registry for managing service registration, health tracking,
    and service discovery using Redis.
    """

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, redis_password: Optional[str] = None,
                 key_prefix: str = ''):
        """
        Initialize ServiceRegistry

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            key_prefix: Prefix for all Redis keys
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        self.key_prefix = key_prefix

    def _key(self, key: str) -> str:
        """Generate prefixed key"""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def register_service(self, service_info: ServiceInfo) -> bool:
        """
        Register a new service or update existing one

        Args:
            service_info: ServiceInfo object with service details

        Returns:
            bool: True if successful
        """
        try:
            service_key = self._key(f"service:{service_info.service_id}")
            active_set = self._key("services:active")
            type_set = self._key(f"services:type:{service_info.service_type}")

            # Store service info in hash
            self.redis_client.hset(service_key, mapping=service_info.to_dict())

            # Add to active services set
            self.redis_client.sadd(active_set, service_info.service_id)

            # Add to type-specific set
            self.redis_client.sadd(type_set, service_info.service_id)

            return True
        except Exception as e:
            print(f"Error registering service: {e}")
            return False

    def deregister_service(self, service_id: str) -> bool:
        """
        Deregister a service

        Args:
            service_id: Service identifier

        Returns:
            bool: True if successful
        """
        try:
            # Get service info first to know its type
            service_info = self.get_service(service_id)
            if not service_info:
                return False

            service_key = self._key(f"service:{service_id}")
            active_set = self._key("services:active")
            type_set = self._key(f"services:type:{service_info.service_type}")

            # Remove from all sets
            self.redis_client.srem(active_set, service_id)
            self.redis_client.srem(type_set, service_id)

            # Delete service hash
            self.redis_client.delete(service_key)

            return True
        except Exception as e:
            print(f"Error deregistering service: {e}")
            return False

    def update_health(self, service_id: str, status: ServiceStatus,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update service health status

        Args:
            service_id: Service identifier
            status: New health status
            metadata: Optional additional metadata

        Returns:
            bool: True if successful
        """
        try:
            service_key = self._key(f"service:{service_id}")

            # Check if service exists
            if not self.redis_client.exists(service_key):
                print(f"Service {service_id} not found")
                return False

            # Update status and last_seen
            updates = {
                'status': status.value,
                'last_seen': str(time.time())
            }

            # Update metadata if provided
            if metadata:
                current_metadata = self.redis_client.hget(service_key, 'metadata')
                if current_metadata:
                    current_metadata = json.loads(current_metadata)
                    current_metadata.update(metadata)
                else:
                    current_metadata = metadata
                updates['metadata'] = json.dumps(current_metadata)

            self.redis_client.hset(service_key, mapping=updates)
            return True
        except Exception as e:
            print(f"Error updating health: {e}")
            return False

    def heartbeat(self, service_id: str) -> bool:
        """
        Record a heartbeat for a service (updates last_seen timestamp)

        Args:
            service_id: Service identifier

        Returns:
            bool: True if successful
        """
        try:
            service_key = self._key(f"service:{service_id}")

            if not self.redis_client.exists(service_key):
                return False

            self.redis_client.hset(service_key, 'last_seen', str(time.time()))
            return True
        except Exception as e:
            print(f"Error recording heartbeat: {e}")
            return False

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        Get service information

        Args:
            service_id: Service identifier

        Returns:
            ServiceInfo object or None if not found
        """
        try:
            service_key = self._key(f"service:{service_id}")
            data = self.redis_client.hgetall(service_key)

            if not data:
                return None

            return ServiceInfo.from_dict(data)
        except Exception as e:
            print(f"Error getting service: {e}")
            return None

    def check_health(self, service_id: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        """
        Check health of a registered service by comparing last heartbeat with current time

        Args:
            service_id: Service identifier
            timeout_seconds: Consider service unhealthy if no heartbeat in this many seconds

        Returns:
            Dictionary with health check results:
            - service_id: The service identifier
            - exists: Whether the service is registered
            - status: The service's reported status (healthy/unhealthy/etc.)
            - is_responsive: Whether the service has sent a heartbeat within timeout
            - last_seen: Unix timestamp of last heartbeat
            - seconds_since_heartbeat: Seconds since last heartbeat
            - timeout_seconds: The timeout threshold used
        """
        current_time = time.time()

        service = self.get_service(service_id)

        if not service:
            return {
                'service_id': service_id,
                'exists': False,
                'status': None,
                'is_responsive': False,
                'last_seen': None,
                'seconds_since_heartbeat': None,
                'timeout_seconds': timeout_seconds
            }

        seconds_since_heartbeat = current_time - service.last_seen
        is_responsive = seconds_since_heartbeat < timeout_seconds

        return {
            'service_id': service_id,
            'exists': True,
            'status': service.status,
            'is_responsive': is_responsive,
            'last_seen': service.last_seen,
            'seconds_since_heartbeat': round(seconds_since_heartbeat, 2),
            'timeout_seconds': timeout_seconds
        }

    def list_services(self, service_type: Optional[str] = None,
                     status_filter: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """
        List all registered services

        Args:
            service_type: Filter by service type (optional)
            status_filter: Filter by status (optional)

        Returns:
            List of ServiceInfo objects
        """
        try:
            # Get service IDs from appropriate set
            if service_type:
                type_set = self._key(f"services:type:{service_type}")
                service_ids = self.redis_client.smembers(type_set)
            else:
                active_set = self._key("services:active")
                service_ids = self.redis_client.smembers(active_set)

            services = []
            for service_id in service_ids:
                service_info = self.get_service(service_id)
                if service_info:
                    # Apply status filter if specified
                    if status_filter is None or service_info.status == status_filter.value:
                        services.append(service_info)

            return services
        except Exception as e:
            print(f"Error listing services: {e}")
            return []

    def get_healthy_services(self, service_type: Optional[str] = None,
                            timeout_seconds: int = 30) -> List[ServiceInfo]:
        """
        Get all healthy services (with recent heartbeat)

        Args:
            service_type: Filter by service type (optional)
            timeout_seconds: Consider service unhealthy if no heartbeat in this many seconds

        Returns:
            List of healthy ServiceInfo objects
        """
        services = self.list_services(service_type=service_type)
        current_time = time.time()
        healthy_services = []

        for service in services:
            time_since_seen = current_time - service.last_seen
            if (service.status == ServiceStatus.HEALTHY.value and
                time_since_seen < timeout_seconds):
                healthy_services.append(service)

        return healthy_services

    def cleanup_unhealthy_services(self) -> int:
        """
        Remove services marked as unhealthy

        This removes services that have been marked as unhealthy by mark_unhealthy_services()
        or other health monitoring processes. Only services with status="unhealthy" are removed.

        Returns:
            Number of services removed
        """
        try:
            services = self.list_services(status_filter=ServiceStatus.UNHEALTHY)
            removed_count = 0

            for service in services:
                if self.deregister_service(service.service_id):
                    removed_count += 1

            return removed_count
        except Exception as e:
            print(f"Error cleaning up unhealthy services: {e}")
            return 0

    def mark_unhealthy_services(self, timeout_seconds: int = 30) -> int:
        """
        Mark services as unhealthy if they haven't sent a heartbeat in a while

        Unlike cleanup_unhealthy_services which removes services, this only updates
        their status to unhealthy, preserving the service registration.

        Args:
            timeout_seconds: Mark unhealthy if no heartbeat in this many seconds (default: 30)

        Returns:
            Number of services marked as unhealthy
        """
        try:
            services = self.list_services()
            current_time = time.time()
            marked_count = 0

            for service in services:
                # Only mark currently healthy services
                if service.status != ServiceStatus.HEALTHY.value:
                    continue

                time_since_seen = current_time - service.last_seen
                if time_since_seen > timeout_seconds:
                    if self.update_health(service.service_id, ServiceStatus.UNHEALTHY):
                        marked_count += 1

            return marked_count
        except Exception as e:
            print(f"Error marking unhealthy services: {e}")
            return 0

    def get_service_count(self, service_type: Optional[str] = None) -> int:
        """
        Get count of registered services

        Args:
            service_type: Filter by service type (optional)

        Returns:
            Number of services
        """
        try:
            if service_type:
                type_set = self._key(f"services:type:{service_type}")
                return self.redis_client.scard(type_set)
            else:
                active_set = self._key("services:active")
                return self.redis_client.scard(active_set)
        except Exception as e:
            print(f"Error getting service count: {e}")
            return 0

    def get_service_types(self) -> List[str]:
        """
        Get list of all service types

        Returns:
            List of service type strings
        """
        try:
            pattern = self._key("services:type:*")
            keys = self.redis_client.keys(pattern)
            prefix = self._key("services:type:")
            return [key.replace(prefix, '') for key in keys]
        except Exception as e:
            print(f"Error getting service types: {e}")
            return []

    def clear_all(self) -> bool:
        """
        Clear all service registry data (USE WITH CAUTION)

        Returns:
            bool: True if successful
        """
        try:
            # Get all service registry keys
            patterns = [
                self._key("service:*"),
                self._key("services:*")
            ]

            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)

            return True
        except Exception as e:
            print(f"Error clearing registry: {e}")
            return False
