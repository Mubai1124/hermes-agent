#!/usr/bin/env python3
"""Enhanced lossless grep — searches historical messages with session browsing and LLM summarization.

Features:
  - Keyword search (LIKE) or recent-sessions browsing (no query)
  - Session-level dedup ( delegation chains)
  - Excludes current session
  - LLM summarization of matching sessions
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = str(Path.home() / ".hermes" / "state" / "lossless_context.db")


def _get_db_path() -> str:
    return DB_PATH


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_to_parent(session_id: str, cur: sqlite3.Cursor) -> str:
    """Walk delegation/parent chain to find the root session ID."""
    visited = set()
    while session_id and session_id not in visited:
        visited.add(session_id)
        row = cur.execute(
            "SELECT parent_session FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row or not row[0]:
            break
        session_id = row[0]
    return session_id


def _get_session_title(session_id: str, cur: sqlite3.Cursor) -> str:
    """Extract a human-readable title from the first user message."""
    row = cur.execute(
        "SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id ASC LIMIT 1",
        (session_id,),
    ).fetchone()
    if not row:
        return session_id
    try:
        parsed = json.loads(row[0])
        if isinstance(parsed, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parsed)
        else:
            text = str(parsed)
    except Exception:
        text = row[0]
    return (text[:80] + "...") if len(text) > 80 else text


def _parse_content(raw: str) -> str:
    """Deserialize serialised message content to plain text."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parsed)
        return str(parsed)
    except Exception:
        return raw


def _summarize_text(text: str, query: str) -> str:
    """Summarise *text* about *query* using the auxiliary LLM."""
    try:
        from agent.auxiliary_client import call_llm

        messages = [
            {
                "role": "user",
                "content": (
                    f'A user searched for "{query}". Summarise the relevant parts of this conversation\n'
                    f"in 2-4 sentences. Focus on facts, decisions, and results — not process.\n\n"
                    f"{text[:6000]}"
                ),
            }
        ]
        result = call_llm(messages=messages, task="session_search")
        if result is None:
            return "[Summarisation unavailable]"
        content = getattr(result, "choices", [None])[0]
        content = getattr(content, "message", None) if content else None
        text = getattr(content, "content", None) if content else None
        return text.strip() if text else "[Summarisation unavailable]"
    except Exception as e:
        logger.warning("Summarisation failed: %s", e)
        return f"[Summarisation unavailable — {e}]"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _list_recent_sessions(
    limit: int,
    current_session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent sessions ordered by last_updated, excluding the current lineage."""
    conn = _open_db()
    try:
        cur = conn.cursor()

        # Resolve current lineage root
        current_root = None
        if current_session_id:
            current_root = _resolve_to_parent(current_session_id, cur)

        rows = cur.execute(
            "SELECT session_id, created_at, last_updated, model "
            "FROM sessions ORDER BY last_updated DESC LIMIT ?",
            (limit * 3,),  # over-fetch to account for exclusions
        ).fetchall()

        results = []
        for row in rows:
            sid = row["session_id"]
            if current_root:
                parent = _resolve_to_parent(sid, cur)
                if parent == current_root:
                    continue
            title = _get_session_title(sid, cur)
            results.append(
                {
                    "session_id": sid,
                    "title": title,
                    "last_updated": row["last_updated"],
                    "model": row["model"],
                }
            )
            if len(results) >= limit:
                break
        return results
    finally:
        conn.close()


def _search_sessions(
    query: str,
    limit: int,
    current_session_id: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full-text search across messages, grouped by resolved session."""
    conn = _open_db()
    try:
        cur = conn.cursor()

        # Resolve current lineage
        current_root = None
        if current_session_id:
            current_root = _resolve_to_parent(current_session_id, cur)

        clause_role = "AND role = ?" if role else ""
        params: List[Any] = []
        if role:
            params.append(role)
        params.append(f"%{query}%")
        params.append(limit * 3)

        rows = cur.execute(
            f"""
            SELECT id, session_id, role, content, timestamp, msg_idx
            FROM messages
            WHERE content LIKE ? {clause_role}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        # Group by resolved parent session
        seen: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            sid = row["session_id"]
            resolved = _resolve_to_parent(sid, cur)

            # Exclude current session lineage
            if current_root and resolved == current_root:
                continue

            if resolved in seen:
                continue

            text = _parse_content(row["content"])
            seen[resolved] = {
                "session_id": resolved,
                "matched_role": row["role"],
                "matched_content": text[:500],
                "timestamp": row["timestamp"],
            }
            if len(seen) >= limit:
                break

        return list(seen.values())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lossless_grep(
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 10,
    role: Optional[str] = None,
    # Internal — injected by registry dispatcher
    _current_session_id: Optional[str] = None,
    _summarise: bool = True,
) -> Dict[str, Any]:
    """
    Search or browse Hermes lossless conversation storage.

    Mode:
      - browse (query empty): list most recent sessions
      - search (query present): full-text search across messages

    Args:
        query: Keyword to search. Omit to browse recent sessions.
        session_id: Optional session ID to filter to a single session.
        limit: Max sessions to return (default 10).
        role: Optional role filter ('user', 'assistant', 'system', 'tool').
        _current_session_id: Internally injected — excludes current session lineage.
        _summarise: Whether to LLM-summarise matching sessions (default True).

    Returns:
        Dict with query, count, mode, and results list.
    """
    if _current_session_id:
        # Normalise: use the same field name throughout
        current_session_id = _current_session_id
    else:
        current_session_id = session_id

    mode = "browse" if not (query and query.strip()) else "search"

    try:
        if mode == "browse":
            sessions = _list_recent_sessions(limit=limit, current_session_id=current_session_id)
            return {
                "mode": mode,
                "count": len(sessions),
                "results": sessions,
            }

        # Search mode
        raw_hits = _search_sessions(
            query=query.strip(),
            limit=limit,
            current_session_id=current_session_id,
            role=role,
        )

        results = []
        for hit in raw_hits:
            sid = hit["session_id"]
            conn = _open_db()
            try:
                cur = conn.cursor()
                # Full conversation for summarisation
                msgs = cur.execute(
                    "SELECT role, content FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                ).fetchall()
                full_text = "\n".join(f"[{r['role']}] {_parse_content(r['content'])}" for r in msgs)
            finally:
                conn.close()

            summary = _summarize_text(full_text, query) if _summarise else None

            results.append(
                {
                    "session_id": sid,
                    "matched_role": hit["matched_role"],
                    "matched_content": hit["matched_content"],
                    "timestamp": hit["timestamp"],
                    "summary": summary,
                }
            )

        return {
            "query": query,
            "mode": mode,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.exception("lossless_grep failed")
        return {"error": str(e), "mode": mode}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "name": "lossless_grep",
    "description": (
        "Search or browse Hermes lossless SQLite storage. "
        "Use when context has been compacted, or when the user asks what was worked on before, "
        "'remember when', 'last time', 'as I mentioned', or asks about past sessions. "
        "WITHOUT A QUERY: returns recent sessions (no LLM call — instant). "
        "WITH A QUERY: full-text search + LLM summarisation of each matching session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to search. Omit entirely to browse recent sessions.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session ID to filter search to one session.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to return (default 10, max 20).",
                "default": 10,
            },
            "role": {
                "type": "string",
                "description": "Optional role filter: user, assistant, system, or tool.",
            },
            "_summarise": {
                "type": "boolean",
                "description": "Whether to LLM-summarise matching sessions (default True). Set False for raw results only.",
                "default": True,
            },
        },
        "required": [],
    },
}


def handle_lossless_grep(arguments: Dict[str, Any], **kwargs) -> str:
    """Handler for the lossless_grep tool — receives all kwargs from dispatch."""
    # session_id injected by handle_function_call → dispatch → here
    current_sid = kwargs.get("session_id") or arguments.get("_current_session_id")
    result = lossless_grep(
        query=arguments.get("query"),
        session_id=arguments.get("session_id"),
        limit=arguments.get("limit", 10),
        role=arguments.get("role"),
        _current_session_id=current_sid,
        _summarise=arguments.get("_summarise", True),
    )
    return json.dumps(result, ensure_ascii=False)


def register(registry):
    registry.register(
        name="lossless_grep",
        toolset="hermes",
        schema=TOOL_SCHEMA,
        handler=handle_lossless_grep,
    )
