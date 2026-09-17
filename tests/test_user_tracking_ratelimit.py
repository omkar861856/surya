import time
try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def raises(exc_type):
            class _RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_val, exc_tb, traceback):
                    return isinstance(exc_tb, exc_type)
            return _RaisesContext()
    pytest = _MockPytest()

from surya.scripts.user_tracker import (
    UserTracker,
    OperationTracker,
    OutageRateLimiter,
)


def test_user_tracker_lifecycle():
    """Verify session creation, activity recording, and active session listing."""
    tracker = UserTracker(session_ttl_seconds=2)

    # 1. Register new user
    sess1 = tracker.record_activity("sess-100", "192.168.1.10", "Mozilla/5.0")
    assert sess1.session_id == "sess-100"
    assert sess1.ip_address == "192.168.1.10"
    assert sess1.total_requests == 1

    # 2. Activity increment
    sess1_updated = tracker.record_activity("sess-100", "192.168.1.10")
    assert sess1_updated.total_requests == 2

    # 3. Add second session
    tracker.record_activity("sess-200", "192.168.1.20", "Chrome/120")
    active = tracker.get_active_sessions(active_window_seconds=10)
    assert len(active) == 2

    # 4. Completed ops & rate limit hits
    tracker.record_completed_op("sess-100")
    assert sess1.total_completed_ops == 1

    tracker.record_rate_limit_hit("sess-100")
    assert sess1.rate_limit_hits == 1


def test_operation_tracker_lifecycle():
    """Verify in-flight operation tracking, stage transitions, and completion history."""
    op_tracker = OperationTracker(history_limit=10)

    # 1. Start operation
    rec = op_tracker.start_operation(
        session_id="sess-abc",
        ip_address="10.0.0.1",
        filename="test_prescription.jpg",
        pipeline_type="Unified OCR + AI",
        initial_stage="Preprocessing",
    )
    assert rec.status == "RUNNING"
    assert rec.stage == "Preprocessing"
    assert op_tracker.get_active_count() == 1
    assert op_tracker.get_active_count_for_user("sess-abc", "10.0.0.1") == 1

    # 2. Stage updates
    op_tracker.update_stage(rec.op_id, "Surya OCR (Port 8000)", blocks_found=25)
    active = op_tracker.get_active_operations()
    assert len(active) == 1
    assert active[0].stage == "Surya OCR (Port 8000)"
    assert active[0].blocks_found == 25

    # 3. Complete operation
    completed = op_tracker.complete_operation(rec.op_id, status="COMPLETED", medicines_found=4)
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.medicines_found == 4
    assert op_tracker.get_active_count() == 0

    # 4. History check
    history = op_tracker.get_recent_history()
    assert len(history) == 1
    assert history[0].op_id == rec.op_id


def test_operation_context_manager():
    """Verify context manager handles success and failure gracefully."""
    op_tracker = OperationTracker(history_limit=10)

    # Success case
    with op_tracker.track_operation("sess-xyz", "10.0.0.2", "doc.pdf") as op:
        assert op_tracker.get_active_count() == 1
        op_tracker.update_stage(op.op_id, "Processing")

    assert op_tracker.get_active_count() == 0
    history = op_tracker.get_recent_history()
    assert len(history) == 1
    assert history[0].status == "COMPLETED"

    # Exception case
    with pytest.raises(ValueError):
        with op_tracker.track_operation("sess-err", "10.0.0.3", "bad.png") as op:
            raise ValueError("Corrupt file")

    assert op_tracker.get_active_count() == 0
    history = op_tracker.get_recent_history()
    assert history[0].status == "FAILED"
    assert "Corrupt file" in (history[0].error_message or "")


def test_rate_limiter_per_ip_limit():
    """Verify per-IP sliding window throttle."""
    limiter = OutageRateLimiter(requests_per_minute=3, window_seconds=2)

    # First 3 requests from IP 1.2.3.4 allowed
    allowed, msg, retry = limiter.check_request("1.2.3.4", "sess-1")
    assert allowed is True

    allowed, msg, retry = limiter.check_request("1.2.3.4", "sess-1")
    assert allowed is True

    allowed, msg, retry = limiter.check_request("1.2.3.4", "sess-1")
    assert allowed is True

    # 4th request from same IP within window must be rejected
    allowed, msg, retry = limiter.check_request("1.2.3.4", "sess-1")
    assert allowed is False
    assert "Rate Limit Exceeded" in msg
    assert retry > 0
    assert limiter.total_blocked_requests == 1

    # Request from a different IP must be allowed
    allowed_other, _, _ = limiter.check_request("5.6.7.8", "sess-2")
    assert allowed_other is True


def test_rate_limiter_concurrency_and_global_capacity():
    """Verify per-user concurrency and global GPU capacity limits."""
    op_tracker = OperationTracker()
    limiter = OutageRateLimiter(
        requests_per_minute=100,
        max_concurrent_per_user=2,
        global_concurrency_limit=3,
    )

    # User 1 starts 2 jobs (max per user = 2)
    op1 = op_tracker.start_operation("sess-u1", "10.0.0.1", "doc1.pdf")
    op2 = op_tracker.start_operation("sess-u1", "10.0.0.1", "doc2.pdf")

    # User 1 attempts 3rd job -> should be blocked by per-user concurrency limit
    allowed, msg, retry = limiter.check_request("10.0.0.1", "sess-u1", op_tracker)
    assert allowed is False
    assert "User Concurrency Limit Reached" in msg

    # User 2 attempts 1st job -> allowed (total active = 2 < global limit 3)
    allowed, msg, retry = limiter.check_request("10.0.0.2", "sess-u2", op_tracker)
    assert allowed is True
    op3 = op_tracker.start_operation("sess-u2", "10.0.0.2", "doc3.pdf")

    # Total active jobs is now 3 (matching global_concurrency_limit = 3)
    # User 3 attempts a job -> should be blocked by Global System GPU Capacity
    allowed, msg, retry = limiter.check_request("10.0.0.3", "sess-u3", op_tracker)
    assert allowed is False
    assert "System at Capacity" in msg

    # Complete one job
    op_tracker.complete_operation(op1.op_id)
    assert op_tracker.get_active_count() == 2

    # User 3 attempts again -> now allowed!
    allowed, msg, retry = limiter.check_request("10.0.0.3", "sess-u3", op_tracker)
    assert allowed is True
