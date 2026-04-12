"""Lossless context manager with SQLite persistence and DAG-based summarization.

Architecture:
  - Every message is persisted to SQLite immediately (append-only log)
  - DAG of summaries: Level-0 = raw messages, Level-1 = summarize(100 msgs),
    Level-2 = summarize(Level-1 nodes), etc.
  - Assembly: fresh tail (last N messages) + highest-level summary available
  - grep/expand for retrieving old content

This is a COMPOSITION layer on top of ContextCompressor — it wraps the existing
“有损” compressor and adds a无损 persistence layer beneath it.
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    msg_idx     INTEGER NOT NULL,
    tool_call_id TEXT,
    parent_id   INTEGER,
    FOREIGN KEY (parent_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    level           INTEGER NOT NULL,
    source_ids      TEXT NOT NULL,   -- JSON list of message IDs
    summary_content TEXT NOT NULL,
    token_count     INTEGER,
    created_at      REAL NOT NULL,
    UNIQUE(session_id, level, source_ids)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    last_updated    REAL NOT NULL,
    model           TEXT,
    parent_session  TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, msg_idx);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, level);
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    timestamp: float
    msg_idx: int
    tool_call_id: Optional[str]
    parent_id: Optional[int]

@dataclass
class SummaryRecord:
    id: int
    session_id: str
    level: int
    source_ids: List[int]
    summary_content: str
    token_count: Optional[int]
    created_at: float

# ---------------------------------------------------------------------------
# LosslessContextManager
# ---------------------------------------------------------------------------

class LosslessContextManager:
    """SQLite-backed lossless context manager with DAG summarization.

    Composable with ContextCompressor — exposes the same ``compress()`` interface
    so the runner can drop it in without changing call-sites.
    """

    # Summary aggregation parameters
    DEFAULT_AGGREGATE_SIZE = 100   # messages per Level-1 summary node
    MAX_SUMMARY_LEVEL = 4          # beyond this, re-summarize the summaries

    # Rough chars-per-token for SQLite storage estimation
    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_id: Optional[str] = None,
        model: str = "unknown",
        parent_session_id: Optional[str] = None,
        # Wrapped compressor (ContextCompressor instance)
        wrapped_compressor=None,
        # Summary generation
        summary_fn=None,   # callable(messages: list[dict]) -> str
        aggregate_size: int = DEFAULT_AGGREGATE_SIZE,
        tail_messages: int = 50,   # fresh tail size for assembly
        quiet: bool = False,
    ):
        self.db_path = db_path or self._default_db_path()
        self.session_id = session_id or self._new_session_id()
        self.model = model
        self.parent_session_id = parent_session_id
        self.wrapped_compressor = wrapped_compressor
        self.summary_fn = summary_fn
        self.aggregate_size = aggregate_size
        self.tail_messages = tail_messages
        self.quiet = quiet

        self._lock = threading.RLock()
        self._init_db()
        # Initialize _next_msg_idx from DB so we don't re-use IDs after fork/restart
        self._next_msg_idx = self._next_msg_idx_unlocked()

        if not quiet:
            logger.info(
                "LosslessContextManager initialized: db=%s session=%s model=%s",
                self.db_path, self.session_id, model,
            )

    # -------------------------------------------------------------------------
    # DB lifecycle
    # -------------------------------------------------------------------------

    def _default_db_path(self) -> str:
        base = os.environ.get("HERMES_STATE_DIR", str(Path.home() / ".hermes" / "state"))
        os.makedirs(base, exist_ok=True)
        return str(Path(base) / "lossless_context.db")

    def _new_session_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _init_db(self):
        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, last_updated, model, parent_session) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.session_id, time.time(), time.time(), self.model, self.parent_session_id),
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # -------------------------------------------------------------------------
    # Message persistence (append-only log)
    # -------------------------------------------------------------------------

    def append_message(
        self,
        role: str,
        content: str,
        tool_call_id: Optional[str] = None,
        parent_id: Optional[int] = None,
        msg_idx: Optional[int] = None,
    ) -> int:
        """Persist a single message to SQLite. Returns the message ID."""
        if msg_idx is None:
            msg_idx = self._next_msg_idx()

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO messages
                   (session_id, role, content, timestamp, msg_idx, tool_call_id, parent_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, role, content, time.time(), msg_idx, tool_call_id, parent_id),
            )
            self._conn.execute(
                "UPDATE sessions SET last_updated = ? WHERE session_id = ?",
                (time.time(), self.session_id),
            )
            self._conn.commit()
            return cursor.lastrowid

    def append_messages_batch(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Persist a list of messages in one transaction. Returns message IDs."""
        if not messages:
            return []
        ids = []
        with self._lock:
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content") or ""
                tool_call_id = msg.get("tool_call_id") or msg.get("tool_call", {}).get("id") if msg.get("role") == "tool" else None
                # Link tool results to their call
                parent_id = None
                if role == "tool":
                    # Try to find the matching assistant message by tool_call_id
                    tc_id = tool_call_id
                    if tc_id:
                        parent_id = self._find_parent_by_tool_id(tc_id)
                msg_idx = self._next_msg_idx_unlocked() + i
                cursor = self._conn.execute(
                    """INSERT INTO messages
                       (session_id, role, content, timestamp, msg_idx, tool_call_id, parent_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.session_id, role, content, time.time(), msg_idx, tool_call_id, parent_id),
                )
                ids.append(cursor.lastrowid)
            self._conn.execute(
                "UPDATE sessions SET last_updated = ? WHERE session_id = ?",
                (time.time(), self.session_id),
            )
            self._conn.commit()
        return ids

    def _next_msg_idx(self) -> int:
        with self._lock:
            return self._next_msg_idx_unlocked()

    def _next_msg_idx_unlocked(self) -> int:
        cursor = self._conn.execute(
            "SELECT COALESCE(MAX(msg_idx), -1) + 1 FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        return cursor.fetchone()[0]

    def _find_parent_by_tool_id(self, tool_call_id: str) -> Optional[int]:
        cursor = self._conn.execute(
            """SELECT m.id FROM messages m
               WHERE m.session_id = ? AND m.role = 'assistant'
               AND m.content LIKE '%' || ? || '%'
               ORDER BY m.msg_idx DESC LIMIT 1""",
            (self.session_id, tool_call_id),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # -------------------------------------------------------------------------
    # Message retrieval
    # -------------------------------------------------------------------------

    def get_messages(
        self,
        session_id: Optional[str] = None,
        since_idx: int = 0,
        limit: int = 1000,
    ) -> List[MessageRecord]:
        """Fetch raw messages for a session, ordered by msg_idx."""
        sid = session_id or self.session_id
        with self._lock:
            cursor = self._conn.execute(
                """SELECT id, session_id, role, content, timestamp, msg_idx, tool_call_id, parent_id
                   FROM messages
                   WHERE session_id = ? AND msg_idx >= ?
                   ORDER BY msg_idx
                   LIMIT ?""",
                (sid, since_idx, limit),
            )
            rows = cursor.fetchall()
        return [
            MessageRecord(
                id=r[0], session_id=r[1], role=r[2], content=r[3],
                timestamp=r[4], msg_idx=r[5], tool_call_id=r[6], parent_id=r[7],
            )
            for r in rows
        ]

    def get_all_messages(self, session_id: Optional[str] = None) -> List[MessageRecord]:
        return self.get_messages(session_id=session_id, since_idx=0, limit=1_000_000)

    # -------------------------------------------------------------------------
    # DAG summarization
    # -------------------------------------------------------------------------

    def _build_summary(
        self,
        source_ids: List[int],
        level: int,
    ) -> Optional[SummaryRecord]:
        """Generate a summary for a group of message IDs at a given DAG level."""
        if not self.summary_fn:
            logger.debug("No summary_fn provided — skipping DAG summarization")
            return None

        # Fetch source messages
        with self._lock:
            placeholders = ",".join(["?"] * len(source_ids))
            cursor = self._conn.execute(
                f"""SELECT role, content FROM messages
                    WHERE id IN ({placeholders})
                    ORDER BY msg_idx""",
                source_ids,
            )
            rows = cursor.fetchall()

        messages = [{"role": r[0], "content": r[1]} for r in rows]
        if not messages:
            return None

        try:
            summary_text = self.summary_fn(messages)
        except Exception as e:
            logger.warning("Summary generation failed at level %d: %s", level, e)
            return None

        token_count = len(summary_text) // self._CHARS_PER_TOKEN

        with self._lock:
            cursor = self._conn.execute(
                """INSERT OR IGNORE INTO summaries
                   (session_id, level, source_ids, summary_content, token_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, level, json.dumps(source_ids), summary_text, token_count, time.time()),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                # Already exists — fetch existing
                cursor = self._conn.execute(
                    """SELECT id, session_id, level, source_ids, summary_content, token_count, created_at
                       FROM summaries WHERE session_id = ? AND level = ? AND source_ids = ?""",
                    (self.session_id, level, json.dumps(source_ids)),
                )
                row = cursor.fetchone()
            else:
                row = (cursor.lastrowid, self.session_id, level, json.dumps(source_ids),
                       summary_text, token_count, time.time())

        return SummaryRecord(
            id=row[0], session_id=row[1], level=row[2],
            source_ids=json.loads(row[3]), summary_content=row[4],
            token_count=row[5], created_at=row[6],
        )

    def _get_latest_summary_level(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COALESCE(MAX(level), 0) FROM summaries WHERE session_id = ?",
                (self.session_id,),
            )
            return cursor.fetchone()[0]

    def _get_summary_nodes(self, level: int) -> List[SummaryRecord]:
        with self._lock:
            cursor = self._conn.execute(
                """SELECT id, session_id, level, source_ids, summary_content, token_count, created_at
                   FROM summaries WHERE session_id = ? AND level = ?
                   ORDER BY created_at""",
                (self.session_id, level),
            )
            rows = cursor.fetchall()
        return [
            SummaryRecord(
                id=r[0], session_id=r[1], level=r[2],
                source_ids=json.loads(r[3]), summary_content=r[4],
                token_count=r[5], created_at=r[6],
            )
            for r in rows
        ]

    def _aggregate_into_next_level(self, source_level: int) -> Optional[SummaryRecord]:
        """Take all summary nodes at source_level and create a summary of them."""
        nodes = self._get_summary_nodes(source_level)
        if not nodes:
            return None

        # Collect all underlying message IDs from source nodes
        all_source_ids = []
        for node in nodes:
            all_source_ids.extend(node.source_ids)

        # Group adjacent nodes into chunks of aggregate_size
        chunk_size = max(1, self.aggregate_size // max(1, len(nodes)))
        chunks = []
        for i in range(0, len(nodes), chunk_size):
            chunk_nodes = nodes[i : i + chunk_size]
            chunk_ids = []
            for n in chunk_nodes:
                chunk_ids.extend(n.source_ids)
            chunks.append((chunk_nodes[0].id, chunk_ids))

        # Build one summary per chunk at level = source_level + 1
        new_level = source_level + 1
        results = []
        for (first_node_id, chunk_ids) in chunks:
            summary = self._build_summary(chunk_ids, new_level)
            if summary:
                results.append(summary)

        # Return the first summary node created (representative)
        return results[0] if results else None

    def _ensure_dag_coverage(self):
        """Build DAG summary layers up to MAX_SUMMARY_LEVEL if needed."""
        current_max = self._get_latest_summary_level()
        if current_max >= self.MAX_SUMMARY_LEVEL:
            return

        for level in range(current_max + 1):
            nodes = self._get_summary_nodes(level)
            if level == 0 and len(nodes) > self.aggregate_size:
                # Level-0 messages need to be summarized into L1 nodes
                self._build_level0_summaries()
            elif level < self.MAX_SUMMARY_LEVEL and len(nodes) >= 2:
                # Multiple nodes at same level → aggregate into next level
                self._aggregate_into_next_level(level)

    def _build_level0_summaries(self):
        """Aggregate raw messages into Level-1 summary nodes."""
        with self._lock:
            cursor = self._conn.execute(
                """SELECT id, msg_idx FROM messages
                   WHERE session_id = ? AND id NOT IN (SELECT value FROM json_each((SELECT GROUP_CONCAT(DISTINCT value) FROM summaries, json_each(summaries.source_ids) WHERE session_id = ?)))
                   ORDER BY msg_idx""",
                (self.session_id, self.session_id),
            )
            # Simpler approach: get all un-aggregated messages
            cursor = self._conn.execute(
                """SELECT id FROM messages m
                   WHERE m.session_id = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM summaries s, json_each(s.source_ids) AS sid
                       WHERE s.session_id = m.session_id AND sid.value = m.id
                   )
                   ORDER BY m.msg_idx""",
                (self.session_id,),
            )
            unaggregated = [r[0] for r in cursor.fetchall()]

        if not unaggregated:
            return

        # Chunk into groups of aggregate_size
        for i in range(0, len(unaggregated), self.aggregate_size):
            chunk = unaggregated[i : i + self.aggregate_size]
            self._build_summary(chunk, level=1)

    # -------------------------------------------------------------------------
    # Persist-only pass (called on every API turn, before compression)
    # -------------------------------------------------------------------------

    def persist(self, messages: List[Dict[str, Any]]):
        """Persist messages to SQLite and build DAG summaries as needed.

        Call this on every API turn to keep the lossless log current.
        This does NOT return assembled context — use assemble() for that.
        """
        if not messages:
            return
        self.append_messages_batch(messages)
        self._ensure_dag_coverage()

    # -------------------------------------------------------------------------
    # Context assembly (the core compress() interface)
    # -------------------------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Persist messages, then assemble: fresh tail + highest-level DAG summary.

        This is the drop-in replacement for ContextCompressor.compress().
        Returns a compressed message list suitable for the LLM API.
        """
        # Phase 1: Persist all incoming messages
        self.persist(messages)

        # Phase 2: Assemble context
        return self._assemble_context()

    def _assemble_context(self) -> List[Dict[str, Any]]:
        """Build the context: system (if any) + fresh tail + highest-level summary."""
        # Get system message (role=system, lowest msg_idx)
        with self._lock:
            cursor = self._conn.execute(
                """SELECT content FROM messages
                   WHERE session_id = ? AND role = 'system'
                   ORDER BY msg_idx LIMIT 1""",
                (self.session_id,),
            )
            system_row = cursor.fetchone()
            system_msg = {"role": "system", "content": system_row[0]} if system_row else None

        # Fresh tail: last N messages
        recent = self.get_messages(session_id=self.session_id, since_idx=0, limit=10_000)
        if len(recent) <= self.tail_messages:
            tail_msgs = recent
        else:
            tail_msgs = recent[-self.tail_messages :]

        # Highest-level summary
        latest_level = self._get_latest_summary_level()
        summary_nodes = self._get_summary_nodes(latest_level) if latest_level > 0 else []

        # Build context list
        ctx: List[Dict[str, Any]] = []

        if system_msg:
            ctx.append(system_msg)

        # Summary prefix message
        if summary_nodes:
            # Use the most recent summary at the highest level
            latest_summary = summary_nodes[-1]
            summary_content = (
                f"[CONTEXT COMPACTION] Earlier turns stored in lossless SQLite log. "
                f"Latest summary (level {latest_level}):\n\n{latest_summary.summary_content}"
            )
            # Pick a role that won't break alternation
            if tail_msgs:
                tail_role = tail_msgs[0].role
                summary_role = "user" if tail_role in ("assistant", "tool") else "assistant"
            else:
                summary_role = "assistant"
            ctx.append({"role": summary_role, "content": summary_content})

        # Fresh tail messages
        for rec in tail_msgs:
            msg = {"role": rec.role, "content": rec.content}
            if rec.tool_call_id:
                msg["tool_call_id"] = rec.tool_call_id
            ctx.append(msg)

        return ctx

    # -------------------------------------------------------------------------
    # Query / retrieval API
    # -------------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        session_id: Optional[str] = None,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search message content using a simple substring or regex pattern.

        Returns matching message records with session_id, role, content, msg_idx.
        """
        sid = session_id or self.session_id
        mode = "LIKE" if not case_sensitive else "LIKE BINARY"
        with self._lock:
            cursor = self._conn.execute(
                f"""SELECT session_id, role, content, msg_idx, timestamp
                   FROM messages
                   WHERE session_id = ? AND content {mode} ?
                   ORDER BY msg_idx DESC
                   LIMIT ?""",
                (sid, f"%{pattern}%" if case_sensitive else f"%{pattern}%", limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "session_id": r[0],
                "role": r[1],
                "content": r[2],
                "msg_idx": r[3],
                "timestamp": r[4],
            }
            for r in rows
        ]

    def expand(
        self,
        msg_idx: int,
        session_id: Optional[str] = None,
        radius: int = 5,
    ) -> List[Dict[str, Any]]:
        """Expand around a message index: return that message ± radius context messages.

        Returns raw messages (role + content) around the given index.
        """
        sid = session_id or self.session_id
        with self._lock:
            cursor = self._conn.execute(
                """SELECT role, content, msg_idx FROM messages
                   WHERE session_id = ? AND msg_idx BETWEEN ? AND ?
                   ORDER BY msg_idx""",
                (sid, max(0, msg_idx - radius), msg_idx + radius),
            )
            rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1], "msg_idx": r[2]} for r in rows]

    def get_session_history(self) -> List[Tuple[str, float, float, Optional[str]]]:
        """Return (session_id, created_at, last_updated, model) for all sessions."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT session_id, created_at, last_updated, model FROM sessions ORDER BY last_updated DESC"
            )
            return cursor.fetchall()

    def get_context_stats(self) -> Dict[str, Any]:
        """Return storage statistics for the current session."""
        with self._lock:
            msg_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]
            summary_count = self._conn.execute(
                "SELECT COUNT(*) FROM summaries WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]
            max_level = self._conn.execute(
                "SELECT COALESCE(MAX(level), 0) FROM summaries WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]
        return {
            "session_id": self.session_id,
            "total_messages": msg_count,
            "total_summaries": summary_count,
            "max_summary_level": max_level,
            "db_path": self.db_path,
        }

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------

    def fork_session(self, new_session_id: Optional[str] = None) -> "LosslessContextManager":
        """Fork a new session that inherits from the current one.

        Args:
            new_session_id: Optional. If None, a new ID is auto-generated.
        """
        new_sid = new_session_id or self._new_session_id()
        return LosslessContextManager(
            db_path=self.db_path,
            session_id=new_sid,
            model=self.model,
            parent_session_id=self.session_id,
            wrapped_compressor=self.wrapped_compressor,
            summary_fn=self.summary_fn,
            aggregate_size=self.aggregate_size,
            tail_messages=self.tail_messages,
            quiet=self.quiet,
        )

    # -------------------------------------------------------------------------
    # ContextCompressor compatibility shim
    # -------------------------------------------------------------------------

    @property
    def compression_count(self) -> int:
        return self._get_latest_summary_level()

    def update_from_response(self, usage: Dict[str, Any]):
        pass  # No-op for lossless manager

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False  # Lossless manager always accepts context — no threshold
