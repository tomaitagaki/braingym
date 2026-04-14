"""
Query tracking for NeuroTribe.

Appends events to a local JSONL file (one JSON object per line).
Each event captures: timestamp, event type, user query, model response metadata,
file info, brain metrics, and session ID.
"""

import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

TRACKING_DIR = Path(__file__).parent / "tracking_logs"
TRACKING_DIR.mkdir(exist_ok=True)

LOG_FILE = TRACKING_DIR / "queries.jsonl"


def get_session_id():
    """Get or create a session ID for this browser session."""
    import streamlit as st
    if "tracking_session_id" not in st.session_state:
        st.session_state["tracking_session_id"] = str(uuid.uuid4())[:12]
    return st.session_state["tracking_session_id"]


def _write_event(event: dict):
    """Append an event to the JSONL log."""
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    event["session_id"] = get_session_id()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def track_query(query: str, has_brain_data: bool, response_len: int,
                viz_requested: list = None, papers_retrieved: int = 0):
    """Track a user chat query."""
    _write_event({
        "type": "query",
        "query": query,
        "has_brain_data": has_brain_data,
        "response_length": response_len,
        "viz_requested": viz_requested or [],
        "papers_retrieved": papers_retrieved,
    })


def track_upload(filename: str, modality: str, duration_seconds: float = None,
                 encoding_time: float = None, n_timepoints: int = None,
                 attention_mean: float = None, engagement_mean: float = None):
    """Track a file upload + encoding event."""
    _write_event({
        "type": "upload",
        "filename": filename,
        "modality": modality,
        "duration_seconds": duration_seconds,
        "encoding_time": encoding_time,
        "n_timepoints": n_timepoints,
        "attention_mean": attention_mean,
        "engagement_mean": engagement_mean,
    })


def track_suggestion_click(suggestion: str):
    """Track when a user clicks a suggested prompt."""
    _write_event({
        "type": "suggestion_click",
        "suggestion": suggestion,
    })


def track_new_session():
    """Track when a user starts a new session."""
    _write_event({"type": "new_session"})


def get_stats() -> dict:
    """Quick stats from the log file."""
    if not LOG_FILE.exists():
        return {"total_events": 0}

    events = []
    for line in LOG_FILE.read_text().strip().split("\n"):
        if line:
            events.append(json.loads(line))

    queries = [e for e in events if e["type"] == "query"]
    uploads = [e for e in events if e["type"] == "upload"]
    sessions = set(e.get("session_id", "") for e in events)

    return {
        "total_events": len(events),
        "total_queries": len(queries),
        "total_uploads": len(uploads),
        "unique_sessions": len(sessions),
        "top_queries": _top_values([q["query"] for q in queries], 10),
        "modality_breakdown": _top_values([u["modality"] for u in uploads], 5),
        "viz_requests": sum(1 for q in queries if q.get("viz_requested")),
    }


def _top_values(items: list, n: int) -> list:
    from collections import Counter
    return Counter(items).most_common(n)
