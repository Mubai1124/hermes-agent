"""Lossless Context Engine — persists every message to SQLite and provides history search.

This engine wraps the standard ContextCompressor so compression behavior is
unchanged, but before each compress() call we write the full raw message list
to the lossless SQLite database.  A ``lossless_grep`` tool lets the agent
search historical messages.

Database schema (created automatically at ~/.hermes/state/lossless_context.db):
    messages   — one row per raw message (role, content, timestamp, msg_idx, tool_call_id, parent_id)
    sessions   — one row per session (session_id, created_at, last_updated, model)
    summaries  — DAG summary nodes keyed by (session_id, level, source_ids)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.model_metadata import get_model_context_length

logger = logging.getLogger(__name__)

_DB_PATH = os.path.expanduser("~/.hermes/state/lossless_context.db")

# Schema version — bump if schema changes
_SCHEMA_VERSION = 1

# Token rough estimate for summary tracking
_ESTIMATED_TOKENS_PER_CHAR = 0.25


def _get_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _ensure_db(db_path: str) -> None:
    """Create schema if the DB doesn't exist yet."""
    if os.path.exists(db_path):
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id     TEXT PRIMARY KEY,
            created_at     REAL NOT NULL,
            last_updated   REAL NOT NULL,
            model          TEXT,
            parent_session TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            timestamp    REAL NOT NULL,
            msg_idx      INTEGER NOT NULL,
            tool_call_id TEXT,
            parent_id    INTEGER,
            FOREIGN KEY (parent_id) REFERENCES messages(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_parent  ON messages(parent_id);

        CREATE TABLE IF NOT EXISTS summaries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            level           INTEGER NOT NULL,
            source_ids      TEXT NOT NULL,
            summary_content TEXT NOT NULL,
            token_count     INTEGER,
            created_at      REAL NOT NULL,
            UNIQUE(session_id, level, source_ids)
        );

        PRAGMA user_version = 1;
    """)
    conn.close()
    logger.info("Created lossless context DB at %s", db_path)


class LosslessContextEngine(ContextEngine):
    """Context engine that writes every message to SQLite before compression.

    The agent can call ``lossless_grep`` to search historical messages.
    Compression delegates to the standard ContextCompressor so the algorithm
    is unchanged.
    """

    name = "lossless"

    def __init__(
        self,
        model: str = "",
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        hermes_home: str | None = None,
    ):
        _home = Path(hermes_home) if hermes_home else _get_hermes_home()
        self._db_path = str(_home / "state" / "lossless_context.db")
        _ensure_db(self._db_path)

        # Delegate compression to the standard compressor
        self._delegate = ContextCompressor(
            model=model,
            threshold_percent=threshold_percent,
            protect_first_n=protect_first_n,
            protect_last_n=protect_last_n,
            summary_target_ratio=summary_target_ratio,
            quiet_mode=quiet_mode,
            summary_model_override=summary_model_override,
            base_url=base_url,
            api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )

        self._session_id: str = ""
        self._msg_idx: int = 0          # monotonic index within session
        self._parent_id: int | None = None  # tracks in-thread replies
        self._quiet = quiet_mode

    # ─── Delegated properties ────────────────────────────────────────────────

    @property
    def last_prompt_tokens(self) -> int:
        return self._delegate.last_prompt_tokens

    @last_prompt_tokens.setter
    def last_prompt_tokens(self, value: int) -> None:
        self._delegate.last_prompt_tokens = value

    @property
    def last_completion_tokens(self) -> int:
        return self._delegate.last_completion_tokens

    @last_completion_tokens.setter
    def last_completion_tokens(self, value: int) -> None:
        self._delegate.last_completion_tokens = value

    @property
    def last_total_tokens(self) -> int:
        return self._delegate.last_total_tokens

    @last_total_tokens.setter
    def last_total_tokens(self, value: int) -> None:
        self._delegate.last_total_tokens = value

    @property
    def threshold_tokens(self) -> int:
        return self._delegate.threshold_tokens

    @property
    def context_length(self) -> int:
        return self._delegate.context_length

    @property
    def compression_count(self) -> int:
        return self._delegate.compression_count

    @property
    def threshold_percent(self) -> float:
        return self._delegate.threshold_percent

    @property
    def protect_first_n(self) -> int:
        return self._delegate.protect_first_n

    @property
    def protect_last_n(self) -> int:
        return self._delegate.protect_last_n

    # ─── ContextEngine interface ─────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._delegate.update_from_response(usage)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return self._delegate.should_compress(prompt_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        return self._delegate.should_compress_preflight(messages)

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._msg_idx = 0
        self._parent_id = None

        conn = sqlite3.connect(self._db_path, timeout=30)
        try:
            now = time.time()
            model = kwargs.get("model", "")
            conn.execute("""
                INSERT OR IGNORE INTO sessions (session_id, created_at, last_updated, model)
                VALUES (?, ?, ?, ?)
            """, (session_id, now, now, model))
            conn.commit()
        finally:
            conn.close()

        if not self._quiet:
            logger.info("LosslessContextEngine session start: %s", session_id)

    def on_session_reset(self) -> None:
        self._delegate.on_session_reset()
        self._msg_idx = 0
        self._parent_id = None
        if not self._quiet:
            logger.info("LosslessContextEngine session reset")

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Flush any remaining messages and close the session."""
        # Write any unwritten messages one last time
        if messages and self._session_id == session_id:
            self._write_messages_batch(messages, final=True)
        self._session_id = ""
        self._msg_idx = 0
        self._parent_id = None
        if not self._quiet:
            logger.info("LosslessContextEngine session end: %s", session_id)

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self._delegate.update_model(model, context_length, base_url, api_key, provider)

    # ─── Core compress (called by run_agent) ─────────────────────────────────

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Write raw messages to SQLite, then delegate compression."""
        if not self._session_id:
            # No active session — just delegate
            return self._delegate.compress(messages, current_tokens)

        # Persist all messages before compression
        self._write_messages_batch(messages)

        # Delegate to the standard compressor (unchanged behavior)
        return self._delegate.compress(messages, current_tokens, focus_topic)

    # ─── lossless_grep tool ──────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [{
            "name": "lossless_grep",
            "description": (
                "Search historical messages in the lossless context database. "
                "Returns matching messages with session ID and timestamp. "
                "Use this when the user asks about something from earlier in the "
                "conversation or a past session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords or phrases to match against message content.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: restrict search to a specific session ID.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10, max 50).",
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional: filter by message role (user, assistant, tool).",
                    },
                },
                "required": ["query"],
            },
        }]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name == "lossless_grep":
            return self._lossless_grep(
                query=args.get("query", ""),
                session_id=args.get("session_id"),
                limit=min(int(args.get("limit", 10)), 50),
                role=args.get("role"),
            )
        return json.dumps({"error": f"Unknown tool: {name}"})

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _write_messages_batch(
        self,
        messages: List[Dict[str, Any]],
        final: bool = False,
    ) -> None:
        """Write a batch of messages to the SQLite DB.

        We track _msg_idx and _parent_id to maintain the parent_id chain
        within a turn (user → assistant → tool-results → assistant → ...).
        """
        if not messages or not self._session_id:
            return

        conn = sqlite3.connect(self._db_path, timeout=30)
        try:
            now = time.time()
            rows = []
            for msg in messages:
                role = msg.get("role", "user")
                # Serialise content: may be str or list of content parts
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                elif not isinstance(content, str):
                    content = str(content)

                tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId") or None
                # Parent_id: the assistant message that spawned this turn is the
                # immediate parent.  We infer it from message ordering:
                # tool/assistant after a user starts a new turn; consecutive
                # assistants belong to the same turn.
                if role == "user":
                    self._parent_id = None  # reset per user turn
                elif role == "assistant":
                    # This assistant message is a reply to the current parent.
                    # Its parent is the previous parent (the tool call or user).
                    pass  # parent_id stays as set
                elif role in ("tool", "tool_result"):
                    parent_id = self._parent_id
                else:
                    parent_id = self._parent_id

                rows.append((
                    self._session_id,
                    role,
                    content,
                    now,
                    self._msg_idx,
                    tool_call_id,
                    self._parent_id,
                ))
                self._msg_idx += 1

                # After writing an assistant message, it becomes the parent for
                # any subsequent tool results in the same turn
                if role == "assistant":
                    # We'll set parent_id = last_insert_rowid after inserting
                    pass

            # Insert all rows and update parent_id for assistant rows
            cur = conn.cursor()
            cur.execute("BEGIN")
            for i, row in enumerate(rows):
                role = row[1]
                cur.execute("""
                    INSERT INTO messages
                        (session_id, role, content, timestamp, msg_idx, tool_call_id, parent_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)
                if role == "assistant":
                    # This assistant message is now the parent for the next turn
                    self._parent_id = cur.lastrowid
            cur.execute("""
                UPDATE sessions SET last_updated = ? WHERE session_id = ?
            """, (now, self._session_id))
            cur.execute("COMMIT")
        finally:
            conn.close()

    def _lossless_grep(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
        role: str | None = None,
    ) -> str:
        """Search historical messages."""
        if not query.strip():
            return json.dumps({"error": "query cannot be empty"}, ensure_ascii=False)

        conn = sqlite3.connect(self._db_path, timeout=30)
        try:
            clause_session = "AND session_id = ?" if session_id else ""
            clause_role = "AND role = ?" if role else ""
            params: list = []
            if session_id:
                params.append(session_id)
            if role:
                params.append(role)
            params.append(f"%{query}%")
            params.append(limit)

            sql = f"""
                SELECT id, session_id, role, content, timestamp, msg_idx
                FROM messages
                WHERE content LIKE ? {clause_session} {clause_role}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for id_, sess, r, content, ts, idx in rows:
                # Reconstruct plain text from serialised content
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        text = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in parsed
                        )
                    else:
                        text = str(parsed)
                except Exception:
                    text = content
                results.append({
                    "id": id_,
                    "session_id": sess,
                    "role": r,
                    "content": text[:500],  # truncate for display
                    "timestamp": ts,
                    "msg_idx": idx,
                })

            return json.dumps({
                "query": query,
                "session_id": session_id,
                "role": role,
                "count": len(results),
                "results": results,
            }, ensure_ascii=False)
        finally:
            conn.close()

    # Expose delegate state for display
    def get_status(self) -> Dict[str, Any]:
        return self._delegate.get_status()
