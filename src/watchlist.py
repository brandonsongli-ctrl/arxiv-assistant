"""
Watchlist and digest generation for keyword/author monitoring.
"""

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from src import scraper, database, dedupe


WATCHLIST_PATH = os.getenv(
    "ARXIV_ASSISTANT_WATCHLIST_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "watchlist", "watchlist.json"),
)
DIGEST_DIR = os.getenv(
    "ARXIV_ASSISTANT_DIGEST_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "watchlist", "digests"),
)


def _utc_now() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


def _from_iso(v: str) -> datetime:
    if not v:
        return _utc_now()
    try:
        return datetime.fromisoformat(str(v).replace("Z", ""))
    except Exception:
        return _utc_now()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _load_store() -> Dict:
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("watches", [])
                data.setdefault("latest_digest", "")
                return data
        except Exception:
            pass
    return {"watches": [], "latest_digest": ""}


def _save_store(data: Dict) -> None:
    _ensure_parent(WATCHLIST_PATH)
    tmp = WATCHLIST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
    os.replace(tmp, WATCHLIST_PATH)


def list_watches() -> List[Dict]:
    store = _load_store()
    return list(store.get("watches", []))


def add_watch(query: str, watch_type: str = "keyword", source: str = "arxiv", frequency: str = "daily") -> str:
    query = str(query or "").strip()
    if not query:
        raise ValueError("query is required")
    watch_type = watch_type if watch_type in {"keyword", "author"} else "keyword"
    source = source if source in {"arxiv", "semantic_scholar"} else "arxiv"
    frequency = frequency if frequency in {"daily", "weekly"} else "daily"

    store = _load_store()
    now = _utc_now()
    watch_id = str(uuid.uuid4())
    store["watches"].append(
        {
            "id": watch_id,
            "query": query,
            "watch_type": watch_type,
            "source": source,
            "frequency": frequency,
            "enabled": True,
            "created_at": _iso(now),
            "last_run": "",
            "next_run": _iso(now),
        }
    )
    _save_store(store)
    return watch_id


def remove_watch(watch_id: str) -> bool:
    store = _load_store()
    before = len(store.get("watches", []))
    store["watches"] = [w for w in store.get("watches", []) if w.get("id") != watch_id]
    after = len(store.get("watches", []))
    if after < before:
        _save_store(store)
        return True
    return False


def toggle_watch(watch_id: str, enabled: bool) -> bool:
    store = _load_store()
    changed = False
    for w in store.get("watches", []):
        if w.get("id") == watch_id:
            w["enabled"] = bool(enabled)
            changed = True
            break
    if changed:
        _save_store(store)
    return changed


def _next_run(now: datetime, frequency: str) -> datetime:
    return now + (timedelta(days=1) if frequency == "daily" else timedelta(days=7))


def _search_watch(watch: Dict, max_results: int = 10) -> List[Dict]:
    query = str(watch.get("query", "")).strip()
    source = watch.get("source", "arxiv")
    watch_type = watch.get("watch_type", "keyword")
    if watch_type == "author" and source == "arxiv":
        # arXiv author search syntax
        query = f'au:"{query}"'

    if source == "semantic_scholar":
        return scraper.search_semantic_scholar(query, max_results=max_results)
    return scraper.search_arxiv(query, max_results=max_results, sort_by="last_updated")


def _filter_new_results(results: List[Dict]) -> List[Dict]:
    existing = database.get_all_papers()
    existing_titles = {dedupe.normalize_title(p.get("title", "")) for p in existing if p.get("title")}
    existing_dois = {
        str(p.get("doi", "")).lower()
        for p in existing
        if p.get("doi") not in ["Unknown", "", None, "None"]
    }
    out = []
    for r in results:
        doi = str(r.get("doi", "") or "").lower().strip()
        tnorm = dedupe.normalize_title(r.get("title", ""))
        if doi and doi in existing_dois:
            continue
        if tnorm and tnorm in existing_titles:
            continue
        out.append(r)
    return out


def _write_digest(lines: List[str]) -> str:
    _ensure_parent(os.path.join(DIGEST_DIR, "placeholder"))
    if not os.path.exists(DIGEST_DIR):
        os.makedirs(DIGEST_DIR, exist_ok=True)
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DIGEST_DIR, f"digest_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def run_due_watches(
    max_results_per_watch: int = 8,
    enqueue_new: bool = False,
    enqueue_fn: Optional[Callable[[Dict], str]] = None,
    run_enrichment: bool = True,
) -> Dict:
    store = _load_store()
    watches = store.get("watches", [])
    now = _utc_now()
    due = []
    for w in watches:
        if not w.get("enabled", True):
            continue
        next_run = _from_iso(w.get("next_run"))
        if next_run <= now:
            due.append(w)

    if not due:
        return {"ran": 0, "new_papers": 0, "digest_path": "", "details": []}

    digest_lines = [f"# Watchlist Digest ({_iso(now)})", ""]
    all_new = 0
    all_queued = 0
    details = []

    for w in due:
        query = w.get("query", "")
        source = w.get("source", "arxiv")
        digest_lines.append(f"## {query} [{source}]")
        try:
            raw = _search_watch(w, max_results=max_results_per_watch)
            raw = scraper.dedupe_results(raw)
            new_items = _filter_new_results(raw)
            queued_count = 0
            if not new_items:
                digest_lines.append("- No new papers.")
            else:
                for p in new_items[: max_results_per_watch]:
                    title = p.get("title", "Unknown")
                    year = p.get("published", "Unknown")
                    venue = p.get("venue", "")
                    doi = p.get("doi", "")
                    suffix = []
                    if year:
                        suffix.append(str(year))
                    if venue:
                        suffix.append(str(venue))
                    if doi:
                        suffix.append(f"DOI:{doi}")
                    digest_lines.append(f"- {title} | {' | '.join(suffix)}")
                    if enqueue_new and enqueue_fn is not None:
                        try:
                            enqueue_fn(p, run_enrichment=run_enrichment)
                            queued_count += 1
                        except Exception:
                            pass
                all_new += len(new_items)
            all_queued += queued_count
            if queued_count > 0:
                digest_lines.append(f"- Queued {queued_count} new papers for background ingest.")
            details.append(
                {
                    "watch_id": w.get("id"),
                    "query": query,
                    "new_count": len(new_items),
                    "queued_count": queued_count,
                }
            )
        except Exception as e:
            digest_lines.append(f"- Error: {str(e)}")
            details.append({"watch_id": w.get("id"), "query": query, "new_count": 0, "queued_count": 0, "error": str(e)})
        digest_lines.append("")

        # schedule next
        w["last_run"] = _iso(now)
        w["next_run"] = _iso(_next_run(now, w.get("frequency", "daily")))

    digest_path = _write_digest(digest_lines)
    store["latest_digest"] = digest_path
    _save_store(store)

    return {
        "ran": len(due),
        "new_papers": all_new,
        "queued_tasks": all_queued,
        "digest_path": digest_path,
        "details": details,
    }


def get_latest_digest_path() -> str:
    store = _load_store()
    return str(store.get("latest_digest", "") or "")


def due_watch_count() -> int:
    now = _utc_now()
    count = 0
    for w in list_watches():
        if not w.get("enabled", True):
            continue
        if _from_iso(w.get("next_run")) <= now:
            count += 1
    return count
