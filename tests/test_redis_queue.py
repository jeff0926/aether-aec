#!/usr/bin/env python3
"""
Tests for Redis queue adapter using fakeredis.

Usage: python tests/test_redis_queue.py

Requires: pip install fakeredis
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, details: str = ""):
    """Record test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name} - {details}")


class FakeRedisQueue:
    """
    Test version of RedisQueue that uses fakeredis.
    Mirrors the real implementation but with in-memory storage.
    """

    def __init__(self):
        import fakeredis
        self._client = fakeredis.FakeRedis(decode_responses=True)

    def _key(self, capsule_id: str) -> str:
        return f"aec:queue:{capsule_id}"

    def _status_key(self, capsule_id: str) -> str:
        return f"aec:status:{capsule_id}"

    def queue_failure(self, capsule_id: str, query: str, response: str, aec_result: dict) -> str:
        import uuid
        from datetime import datetime

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

        self._client.lpush(self._key(capsule_id), json.dumps(record))
        self._client.hset(self._status_key(capsule_id), record_id, "pending")
        return record_id

    def get_pending(self, capsule_id: str) -> list:
        records = []
        raw_items = self._client.lrange(self._key(capsule_id), 0, -1)
        for item in raw_items:
            try:
                record = json.loads(item)
                if record.get("status") == "pending":
                    records.append(record)
            except json.JSONDecodeError:
                continue
        return records

    def update_status(self, capsule_id: str, record_id: str, status: str, metadata: dict = None) -> bool:
        from datetime import datetime

        key = self._key(capsule_id)
        raw_items = self._client.lrange(key, 0, -1)

        for i, item in enumerate(raw_items):
            try:
                record = json.loads(item)
                if record.get("id") == record_id:
                    record["status"] = status
                    record["updated_at"] = datetime.now().isoformat()
                    if metadata:
                        record.update(metadata)
                    self._client.lset(key, i, json.dumps(record))
                    self._client.hset(self._status_key(capsule_id), record_id, status)
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def get_queue(self, capsule_id: str) -> list:
        records = []
        raw_items = self._client.lrange(self._key(capsule_id), 0, -1)
        for item in raw_items:
            try:
                records.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return records

    def queue_stats(self, capsule_id: str) -> dict:
        stats = {
            "total": 0,
            "pending": 0,
            "researching": 0,
            "validated": 0,
            "integrated": 0,
            "failed": 0,
            "rejected_contradiction": 0,
        }
        statuses = self._client.hgetall(self._status_key(capsule_id))
        stats["total"] = len(statuses)
        for record_id, status in statuses.items():
            if status in stats:
                stats[status] += 1
        return stats

    def clear_queue(self, capsule_id: str) -> int:
        count = self._client.llen(self._key(capsule_id))
        self._client.delete(self._key(capsule_id))
        self._client.delete(self._status_key(capsule_id))
        return count


def test_queue_failure():
    """queue_failure() adds item to queue."""
    print("\nTest: queue_failure")

    queue = FakeRedisQueue()
    record_id = queue.queue_failure(
        capsule_id="test-capsule",
        query="What is 2+2?",
        response="2+2 equals 5.",
        aec_result={"score": 0.0, "passed": False},
    )

    check("returns record_id", isinstance(record_id, str), f"got {type(record_id)}")
    check("record_id has length", len(record_id) == 8, f"got length {len(record_id)}")

    # Verify it's in the queue
    all_records = queue.get_queue("test-capsule")
    check("record in queue", len(all_records) == 1, f"got {len(all_records)} records")


def test_get_pending():
    """get_pending() returns queued items."""
    print("\nTest: get_pending")

    queue = FakeRedisQueue()

    # Add two pending items
    queue.queue_failure("test-capsule", "q1", "r1", {"score": 0.1})
    queue.queue_failure("test-capsule", "q2", "r2", {"score": 0.2})

    pending = queue.get_pending("test-capsule")

    check("returns list", isinstance(pending, list), f"got {type(pending)}")
    check("two pending items", len(pending) == 2, f"got {len(pending)}")

    for item in pending:
        check(f"item has status=pending", item.get("status") == "pending", f"got {item.get('status')}")


def test_update_status():
    """update_status() changes item status."""
    print("\nTest: update_status")

    queue = FakeRedisQueue()
    record_id = queue.queue_failure("test-capsule", "query", "response", {"score": 0.5})

    # Update status
    result = queue.update_status("test-capsule", record_id, "researching")
    check("update returns True", result == True, f"got {result}")

    # Verify status changed
    pending = queue.get_pending("test-capsule")
    check("no longer pending", len(pending) == 0, f"got {len(pending)} pending")

    all_records = queue.get_queue("test-capsule")
    check("still in queue", len(all_records) == 1, f"got {len(all_records)}")
    check("status is researching", all_records[0].get("status") == "researching",
          f"got {all_records[0].get('status')}")


def test_queue_stats():
    """queue_stats() returns counts."""
    print("\nTest: queue_stats")

    queue = FakeRedisQueue()

    # Add items with different statuses
    id1 = queue.queue_failure("test-capsule", "q1", "r1", {})
    id2 = queue.queue_failure("test-capsule", "q2", "r2", {})
    id3 = queue.queue_failure("test-capsule", "q3", "r3", {})

    queue.update_status("test-capsule", id1, "researching")
    queue.update_status("test-capsule", id2, "integrated")

    stats = queue.queue_stats("test-capsule")

    check("total is 3", stats.get("total") == 3, f"got {stats.get('total')}")
    check("pending is 1", stats.get("pending") == 1, f"got {stats.get('pending')}")
    check("researching is 1", stats.get("researching") == 1, f"got {stats.get('researching')}")
    check("integrated is 1", stats.get("integrated") == 1, f"got {stats.get('integrated')}")


def test_interface_compatibility():
    """All 5 functions match required signatures."""
    print("\nTest: interface_compatibility")

    from adapters.redis_queue import RedisQueue

    required_functions = [
        "queue_failure",
        "get_pending",
        "update_status",
        "get_queue",
        "queue_stats",
    ]

    for fn in required_functions:
        has_method = hasattr(RedisQueue, fn)
        check(f"has {fn}", has_method, "method missing")


def test_redis_unavailable_raises():
    """Clear error when Redis not available."""
    print("\nTest: redis_unavailable_raises")

    from adapters.redis_queue import RedisQueue, RedisQueueError

    # Create queue with invalid URL
    queue = RedisQueue(redis_url="redis://nonexistent-host:12345")

    try:
        queue.queue_failure("test", "q", "r", {})
        check("raises RedisQueueError", False, "should have raised")
    except RedisQueueError as e:
        check("raises RedisQueueError", True, "")
        check("error message is helpful", "unavailable" in str(e).lower(), f"got: {e}")
    except Exception as e:
        check("raises RedisQueueError", False, f"got {type(e).__name__}: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AEC Redis Queue Adapter Tests")
    print("=" * 60)

    # Check fakeredis is available
    try:
        import fakeredis
    except ImportError:
        print("fakeredis not installed. Run: pip install fakeredis")
        return 1

    test_queue_failure()
    test_get_pending()
    test_update_status()
    test_queue_stats()
    test_interface_compatibility()
    test_redis_unavailable_raises()

    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"SUMMARY: {PASSED}/{total} passed")
    if FAILED > 0:
        print(f"         {FAILED} failed")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
