#!/usr/bin/env python3
"""
Session Search Tool - Delegates to lossless_grep for Hermes long-term recall.

Keeps the session_search tool name/schema for backwards compatibility,
but all actual search + summarization logic is handled by lossless_grep.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Delegate to lossless_grep for recent sessions browsing."""
    from tools.lossless_grep_tool import lossless_grep
    result = lossless_grep(
        query=None,
        limit=limit,
        _current_session_id=current_session_id,
        _summarise=False,  # No LLM call for recent browsing
    )
    return json.dumps({
        "success": True,
        "mode": "recent",
        "results": result.get("results", []),
        "count": result.get("count", 0),
        "message": f"Showing {result.get('count', 0)} most recent sessions.",
    }, ensure_ascii=False)


def session_search(
    query: str,
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
) -> str:
    """
    Search past sessions and return focused summaries.
    Delegates entirely to lossless_grep for search + LLM summarization.
    """
    from tools.lossless_grep_tool import lossless_grep

    result = lossless_grep(
        query=query or None,
        limit=min(limit, 5),
        role=role_filter,
        _current_session_id=current_session_id,
        _summarise=True,
    )
    return json.dumps({
        "success": True,
        "query": query,
        "mode": result.get("mode"),
        "results": result.get("results", []),
        "count": result.get("count", 0),
        "sessions_searched": result.get("count", 0),
    }, ensure_ascii=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database and an auxiliary text model."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- "
        "every past session is searchable, and this tool summarizes what happened.\n\n"
        "TWO MODES:\n"
        "1. Recent sessions (no query): Call with no arguments to see what was worked on recently. "
        "Returns titles, previews, and timestamps. Zero LLM cost, instant. "
        "Start here when the user asks what were we working on or what did we do recently.\n"
        "2. Keyword search (with query): Search for specific topics across all past sessions. "
        "Returns LLM-generated summaries of matching sessions.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search when it is actually cross-session -- it's fast and cheap. "
        "Better to search and confirm than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), "
        "phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel. Returns summaries of the top matching sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead (returns titles, previews, timestamps with no LLM cost).",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to summarize (default: 3, max: 5).",
                "default": 3,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id")),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
