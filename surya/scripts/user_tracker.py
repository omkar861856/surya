"""User & In-Flight Operation Tracker and Outage Rate Limiter.
Tracks active user sessions, ongoing document processing pipelines, and enforces
per-IP rate limits and global GPU concurrency caps to prevent system outages and 504 Bad Gateways.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import datetime
import os
import threading
import time
import uuid
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple


@dataclasses.dataclass
class UserSession:
    session_id: str
    ip_address: str
    user_agent: str
    first_seen_at: float
    last_active_at: float
    total_requests: int = 1
    total_completed_ops: int = 0
    rate_limit_hits: int = 0
    is_blocked: bool = False

    @property
    def formatted_first_seen(self) -> str:
        return datetime.datetime.fromtimestamp(self.first_seen_at).strftime("%H:%M:%S")

    @property
    def formatted_last_active(self) -> str:
        return datetime.datetime.fromtimestamp(self.last_active_at).strftime("%H:%M:%S")

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time.time() - self.last_active_at)


@dataclasses.dataclass
class OperationRecord:
    op_id: str
    session_id: str
    ip_address: str
    filename: str
    pipeline_type: str
    stage: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "RUNNING"  # RUNNING, COMPLETED, FAILED, RATE_LIMITED
    error_message: Optional[str] = None
    blocks_found: int = 0
    medicines_found: int = 0

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)

    @property
    def formatted_start_time(self) -> str:
        return datetime.datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")


class UserTracker:
    """Thread-safe registry for connected users and sessions."""

    def __init__(self, session_ttl_seconds: int = 900):
        self._lock = threading.Lock()
        self._sessions: Dict[str, UserSession] = {}
        self._session_ttl = session_ttl_seconds

    def record_activity(self, session_id: str, ip_address: str, user_agent: str = "Unknown") -> UserSession:
        now = time.time()
        with self._lock:
            if session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.last_active_at = now
                sess.ip_address = ip_address or sess.ip_address
                sess.total_requests += 1
                return sess
            else:
                sess = UserSession(
                    session_id=session_id,
                    ip_address=ip_address or "127.0.0.1",
                    user_agent=user_agent[:120] if user_agent else "Unknown",
                    first_seen_at=now,
                    last_active_at=now,
                    total_requests=1,
                )
                self._sessions[session_id] = sess
                self._prune_stale_locked(now)
                return sess

    def record_completed_op(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].total_completed_ops += 1

    def record_rate_limit_hit(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].rate_limit_hits += 1

    def _prune_stale_locked(self, current_time: float):
        cutoff = current_time - self._session_ttl
        stale_keys = [sid for sid, sess in self._sessions.items() if sess.last_active_at < cutoff]
        for sid in stale_keys:
            del self._sessions[sid]

    def get_active_sessions(self, active_window_seconds: int = 300) -> List[UserSession]:
        now = time.time()
        with self._lock:
            self._prune_stale_locked(now)
            cutoff = now - active_window_seconds
            active = [s for s in self._sessions.values() if s.last_active_at >= cutoff]
            # Return sorted by last active descending
            return sorted(active, key=lambda s: s.last_active_at, reverse=True)

    def get_total_tracked_sessions_count(self) -> int:
        with self._lock:
            return len(self._sessions)


class OperationTracker:
    """Thread-safe tracker for in-flight operations and historical records."""

    def __init__(self, history_limit: int = 60):
        self._lock = threading.Lock()
        self._active_ops: Dict[str, OperationRecord] = {}
        self._history: Deque[OperationRecord] = collections.deque(maxlen=history_limit)

    def start_operation(
        self,
        session_id: str,
        ip_address: str,
        filename: str,
        pipeline_type: str = "Unified OCR + AI",
        initial_stage: str = "Queued",
    ) -> OperationRecord:
        op_id = f"op-{uuid.uuid4().hex[:8]}"
        record = OperationRecord(
            op_id=op_id,
            session_id=session_id,
            ip_address=ip_address,
            filename=filename,
            pipeline_type=pipeline_type,
            stage=initial_stage,
            start_time=time.time(),
            status="RUNNING",
        )
        with self._lock:
            self._active_ops[op_id] = record
        return record

    def update_stage(self, op_id: str, new_stage: str, blocks_found: int = 0, medicines_found: int = 0):
        with self._lock:
            if op_id in self._active_ops:
                op = self._active_ops[op_id]
                op.stage = new_stage
                if blocks_found > 0:
                    op.blocks_found = blocks_found
                if medicines_found > 0:
                    op.medicines_found = medicines_found

    def complete_operation(
        self,
        op_id: str,
        status: str = "COMPLETED",
        error_message: Optional[str] = None,
        blocks_found: int = 0,
        medicines_found: int = 0,
    ) -> Optional[OperationRecord]:
        now = time.time()
        with self._lock:
            if op_id in self._active_ops:
                op = self._active_ops.pop(op_id)
                op.end_time = now
                op.status = status
                op.error_message = error_message
                if blocks_found > 0:
                    op.blocks_found = blocks_found
                if medicines_found > 0:
                    op.medicines_found = medicines_found
                op.stage = "✅ Completed" if status == "COMPLETED" else f"❌ {status}"
                self._history.appendleft(op)
                return op
            return None

    def get_active_operations(self) -> List[OperationRecord]:
        with self._lock:
            ops = list(self._active_ops.values())
            return sorted(ops, key=lambda o: o.start_time)

    def get_active_count(self) -> int:
        with self._lock:
            return len(self._active_ops)

    def get_active_count_for_user(self, session_id: str, ip_address: str) -> int:
        with self._lock:
            return sum(
                1 for o in self._active_ops.values()
                if o.session_id == session_id or (ip_address and o.ip_address == ip_address)
            )

    def get_recent_history(self, limit: int = 30) -> List[OperationRecord]:
        with self._lock:
            return list(self._history)[:limit]

    @contextlib.contextmanager
    def track_operation(
        self,
        session_id: str,
        ip_address: str,
        filename: str,
        pipeline_type: str = "Unified OCR + AI",
    ) -> Generator[OperationRecord, None, None]:
        record = self.start_operation(session_id, ip_address, filename, pipeline_type)
        try:
            yield record
            self.complete_operation(record.op_id, status="COMPLETED")
        except Exception as exc:
            self.complete_operation(record.op_id, status="FAILED", error_message=str(exc))
            raise


class OutageRateLimiter:
    """Multi-tier Rate Limiter & Outage Shield:
    1. Per-IP Sliding Window rate limiter (preventing high-frequency flood).
    2. Per-User concurrent operation cap (preventing single user slot hoarding).
    3. Global system GPU concurrency gate (capping total parallel GPU operations
       to match hardware capacity and avoid 504 Bad Gateways/OOM).
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        max_concurrent_per_user: int = 2,
        global_concurrency_limit: int = 4,
        window_seconds: int = 60,
    ):
        self._lock = threading.Lock()
        self.requests_per_minute = requests_per_minute
        self.max_concurrent_per_user = max_concurrent_per_user
        self.global_concurrency_limit = global_concurrency_limit
        self.window_seconds = window_seconds

        # Per-IP timestamps deque: IP -> Deque[float]
        self._ip_request_timestamps: Dict[str, Deque[float]] = collections.defaultdict(
            lambda: collections.deque()
        )
        # Statistics
        self.total_checked_requests: int = 0
        self.total_blocked_requests: int = 0
        self.total_ip_rate_limit_hits: int = 0
        self.total_concurrency_limit_hits: int = 0
        self.total_global_capacity_hits: int = 0

    def check_request(
        self,
        client_ip: str,
        session_id: str,
        operation_tracker: Optional[OperationTracker] = None,
    ) -> Tuple[bool, str, int]:
        """Check if request is allowed.
        Returns:
            (is_allowed, reason_message, retry_after_seconds)
        """
        now = time.time()
        with self._lock:
            self.total_checked_requests += 1
            ip_key = client_ip or "127.0.0.1"

            # 1. Check Global System GPU Capacity
            if operation_tracker:
                current_active = operation_tracker.get_active_count()
                if current_active >= self.global_concurrency_limit:
                    self.total_blocked_requests += 1
                    self.total_global_capacity_hits += 1
                    return (
                        False,
                        f"⚡ System at Capacity: {current_active}/{self.global_concurrency_limit} GPU slots in use. "
                        "To prevent server outages and gateway timeouts, please wait a moment.",
                        5,
                    )

            # 2. Check Per-User Concurrency
            if operation_tracker:
                user_active = operation_tracker.get_active_count_for_user(session_id, client_ip)
                if user_active >= self.max_concurrent_per_user:
                    self.total_blocked_requests += 1
                    self.total_concurrency_limit_hits += 1
                    return (
                        False,
                        f"⏳ User Concurrency Limit Reached: You have {user_active} active document tasks. "
                        f"Max allowed is {self.max_concurrent_per_user} per user.",
                        4,
                    )

            # 3. Check Per-IP Sliding Window Rate
            window_cutoff = now - self.window_seconds
            ts_queue = self._ip_request_timestamps[ip_key]

            # Prune old timestamps
            while ts_queue and ts_queue[0] < window_cutoff:
                ts_queue.popleft()

            if len(ts_queue) >= self.requests_per_minute:
                self.total_blocked_requests += 1
                self.total_ip_rate_limit_hits += 1
                oldest_in_window = ts_queue[0]
                retry_after = max(1, int(oldest_in_window + self.window_seconds - now))
                return (
                    False,
                    f"🛑 Rate Limit Exceeded: Max {self.requests_per_minute} requests/min per IP. "
                    f"Please retry in {retry_after}s.",
                    retry_after,
                )

            # Record request timestamp
            ts_queue.append(now)
            return (True, "OK", 0)

    def update_limits(
        self,
        requests_per_minute: Optional[int] = None,
        max_concurrent_per_user: Optional[int] = None,
        global_concurrency_limit: Optional[int] = None,
    ):
        with self._lock:
            if requests_per_minute is not None:
                self.requests_per_minute = max(1, requests_per_minute)
            if max_concurrent_per_user is not None:
                self.max_concurrent_per_user = max(1, max_concurrent_per_user)
            if global_concurrency_limit is not None:
                self.global_concurrency_limit = max(1, global_concurrency_limit)


# Global Singleton Instances
user_tracker = UserTracker()
operation_tracker = OperationTracker()
outage_limiter = OutageRateLimiter()


def get_client_ip_and_agent() -> Tuple[str, str]:
    """Helper to extract real client IP and User Agent in Streamlit."""
    client_ip = "127.0.0.1"
    user_agent = "Browser Session"

    try:
        import streamlit as st

        # Streamlit 1.35+ provides st.context.headers
        if hasattr(st, "context") and hasattr(st.context, "headers") and st.context.headers:
            headers = st.context.headers
            # Caddy / Reverse-proxy header chain
            forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            elif headers.get("x-real-ip"):
                client_ip = headers.get("x-real-ip").strip()
            elif headers.get("remote-addr"):
                client_ip = headers.get("remote-addr").strip()

            user_agent = headers.get("user-agent") or headers.get("User-Agent") or user_agent

        # Fallback to websocket headers if available
        if client_ip == "127.0.0.1":
            try:
                from streamlit.web.server.websocket_headers import _get_websocket_headers
                ws_headers = _get_websocket_headers()
                if ws_headers:
                    forwarded = ws_headers.get("x-forwarded-for") or ws_headers.get("X-Forwarded-For")
                    if forwarded:
                        client_ip = forwarded.split(",")[0].strip()
                    elif ws_headers.get("x-real-ip"):
                        client_ip = ws_headers.get("x-real-ip").strip()
            except Exception:
                pass
    except Exception:
        pass

    return client_ip, user_agent


def get_or_create_session_id() -> str:
    """Returns or sets a persistent unique session ID in Streamlit session state."""
    try:
        import streamlit as st
        if "session_tracking_id" not in st.session_state:
            st.session_state["session_tracking_id"] = f"sess-{uuid.uuid4().hex[:10]}"
        return st.session_state["session_tracking_id"]
    except Exception:
        return f"sess-{uuid.uuid4().hex[:10]}"
