#!/usr/bin/env python3
"""
AEC Redis Queue Adapter - Redis-backed education queue for cross-service deployments.

Replaces file-based education-queue.json with Redis lists.
Same interface as education.py queue functions for drop-in replacement.

Key pattern: aec:queue:{capsule_id}

Usage:
    from adapters.redis_queue import RedisQueue

    queue = RedisQueue()
    queue.queue_failure("my-capsule", query, response, aec_result)
    pending = queue.get_pending("my-capsule")
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any


class RedisQueueError(Exception):
    """Raised when Redis is unavailable or operations fail."""
    pass


class RedisQueue:
    """Redis-backed education queue with same interface as file-based queue."""

    def __init__(self, redis_url: str | None = None):
        """
        Initialize Redis connection.

        Args:
            redis_url: Redis connection URL (default: REDIS_URL env var or redis://localhost:6379)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = None

    def _get_client(self):
        """Get Redis client, raising clear error if unavailable."""
        if self._client is not None:
            return self._client

        try:
            import redis
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self._client.ping()
            return self._client
        except ImportError:
            raise RedisQueueError("Redis unavailable — install redis package: pip install redis")
        except Exception as e:
            raise RedisQueueError(f"Redis unavailable — use file backend: {str(e)}")

    def _key(self, capsule_id: str) -> str:
        """Generate Redis key for capsule queue."""
        return f"aec:queue:{capsule_id}"

    def _status_key(self, capsule_id: str) -> str:
        """Generate Redis key for status index."""
        return f"aec:status:{capsule_id}"

    def queue_failure(
        self,
        capsule_id: str,
        query: str,
        response: str,
        aec_result: dict,
    ) -> str:
        """
        Add AEC failure to education queue.

        Args:
            capsule_id: Identifier for the capsule/agent
            query: Original query that triggered the response
            response: LLM response that failed verification
            aec_result: Full AEC verification result dict

        Returns:
            Record ID for tracking
        """
        client = self._get_client()

        record_id = str(uuid.uuid4())[:8]
        record = {
            "id": record_id,
            "capsule_id": capsule_id,
            "query": query,
            "response": response,
            "aec_result": aec_result,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Add to queue list
        client.lpush(self._key(capsule_id), json.dumps(record))

        # Index by status
        client.hset(self._status_key(capsule_id), record_id, "pending")

        return record_id

    def get_pending(self, capsule_id: str) -> list[dict]:
        """
        Get all pending education records for a capsule.

        Args:
            capsule_id: Identifier for the capsule/agent

        Returns:
            List of pending record dicts
        """
        client = self._get_client()

        records = []
        raw_items = client.lrange(self._key(capsule_id), 0, -1)

        for item in raw_items:
            try:
                record = json.loads(item)
                if record.get("status") == "pending":
                    records.append(record)
            except json.JSONDecodeError:
                continue

        return records

    def update_status(
        self,
        capsule_id: str,
        record_id: str,
        status: str,
        metadata: dict | None = None,
    ) -> bool:
        """
        Update status of an education record.

        Args:
            capsule_id: Identifier for the capsule/agent
            record_id: Record ID to update
            status: New status (pending, researching, validated, integrated, failed, rejected_contradiction)
            metadata: Optional additional metadata to merge

        Returns:
            True if record was found and updated, False otherwise
        """
        client = self._get_client()

        # Get all items and find the one to update
        key = self._key(capsule_id)
        raw_items = client.lrange(key, 0, -1)

        for i, item in enumerate(raw_items):
            try:
                record = json.loads(item)
                if record.get("id") == record_id:
                    # Update record
                    record["status"] = status
                    record["updated_at"] = datetime.now().isoformat()
                    if metadata:
                        record.update(metadata)

                    # Replace in list
                    client.lset(key, i, json.dumps(record))

                    # Update status index
                    client.hset(self._status_key(capsule_id), record_id, status)

                    return True
            except json.JSONDecodeError:
                continue

        return False

    def get_queue(self, capsule_id: str) -> list[dict]:
        """
        Get all education records for a capsule (all statuses).

        Args:
            capsule_id: Identifier for the capsule/agent

        Returns:
            List of all record dicts
        """
        client = self._get_client()

        records = []
        raw_items = client.lrange(self._key(capsule_id), 0, -1)

        for item in raw_items:
            try:
                records.append(json.loads(item))
            except json.JSONDecodeError:
                continue

        return records

    def queue_stats(self, capsule_id: str) -> dict:
        """
        Get queue statistics for a capsule.

        Args:
            capsule_id: Identifier for the capsule/agent

        Returns:
            Dict with counts by status
        """
        client = self._get_client()

        stats = {
            "total": 0,
            "pending": 0,
            "researching": 0,
            "validated": 0,
            "integrated": 0,
            "failed": 0,
            "rejected_contradiction": 0,
        }

        # Get all status values
        statuses = client.hgetall(self._status_key(capsule_id))
        stats["total"] = len(statuses)

        for record_id, status in statuses.items():
            if status in stats:
                stats[status] += 1

        return stats

    def clear_queue(self, capsule_id: str) -> int:
        """
        Clear all education records for a capsule.

        Args:
            capsule_id: Identifier for the capsule/agent

        Returns:
            Number of records cleared
        """
        client = self._get_client()

        count = client.llen(self._key(capsule_id))
        client.delete(self._key(capsule_id))
        client.delete(self._status_key(capsule_id))

        return count


# Convenience functions matching education.py interface
_default_queue: RedisQueue | None = None


def _get_queue() -> RedisQueue:
    """Get or create default queue instance."""
    global _default_queue
    if _default_queue is None:
        _default_queue = RedisQueue()
    return _default_queue


def queue_failure(capsule_id: str, query: str, response: str, aec_result: dict) -> str:
    """Add AEC failure to education queue."""
    return _get_queue().queue_failure(capsule_id, query, response, aec_result)


def get_pending(capsule_id: str) -> list[dict]:
    """Get all pending education records for a capsule."""
    return _get_queue().get_pending(capsule_id)


def update_status(capsule_id: str, record_id: str, status: str, metadata: dict | None = None) -> bool:
    """Update status of an education record."""
    return _get_queue().update_status(capsule_id, record_id, status, metadata)


def get_queue(capsule_id: str) -> list[dict]:
    """Get all education records for a capsule."""
    return _get_queue().get_queue(capsule_id)


def queue_stats(capsule_id: str) -> dict:
    """Get queue statistics for a capsule."""
    return _get_queue().queue_stats(capsule_id)


if __name__ == "__main__":
    # Test the interface
    print("Testing RedisQueue interface...")

    try:
        queue = RedisQueue()
        print(f"Redis URL: {queue.redis_url}")

        # Test queue_failure
        record_id = queue.queue_failure(
            capsule_id="test-capsule",
            query="What is the capital of France?",
            response="The capital of France is Berlin.",
            aec_result={"score": 0.0, "passed": False, "gaps": []},
        )
        print(f"Created record: {record_id}")

        # Test get_pending
        pending = queue.get_pending("test-capsule")
        print(f"Pending records: {len(pending)}")

        # Test queue_stats
        stats = queue.queue_stats("test-capsule")
        print(f"Queue stats: {stats}")

        # Test update_status
        updated = queue.update_status("test-capsule", record_id, "researching")
        print(f"Updated status: {updated}")

        # Clean up
        cleared = queue.clear_queue("test-capsule")
        print(f"Cleared {cleared} records")

        print("\nAll interface tests passed!")

    except RedisQueueError as e:
        print(f"\nRedis not available: {e}")
        print("This is expected if Redis is not running locally.")

    # Interface compatibility check
    print("\n--- Interface Compatibility Check ---")
    required_functions = ["queue_failure", "get_pending", "update_status", "get_queue", "queue_stats"]
    for fn in required_functions:
        if hasattr(RedisQueue, fn):
            print(f"  {fn}: OK")
        else:
            print(f"  {fn}: MISSING")
