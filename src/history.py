"""
Search History Utilities
"""

import json
import os
from datetime import datetime
from typing import List, Dict

HISTORY_PATH = os.path.expanduser(os.getenv("ARXIV_ASSISTANT_HISTORY_PATH", "~/.arxiv_assistant/search_history.json"))
HISTORY_LIMIT = 100


def _ensure_parent_dir():
    parent = os.path.dirname(HISTORY_PATH)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_history(limit: int = HISTORY_LIMIT) -> List[Dict]:
    try:
        if not os.path.exists(HISTORY_PATH):
            return []
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-limit:]
    except Exception:
        return []
    return []


def append_history(entry: Dict, limit: int = HISTORY_LIMIT) -> None:
    if not entry:
        return
    entry = dict(entry)
    entry["timestamp"] = entry.get("timestamp") or datetime.utcnow().isoformat()
    
    _ensure_parent_dir()
    history = load_history(limit=limit)
    history.append(entry)
    history = history[-limit:]
    
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=True, indent=2)
    except Exception:
        pass


def clear_history() -> None:
    try:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
    except Exception:
        pass
