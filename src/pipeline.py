"""
Incremental post-ingest maintenance pipeline.

After a new paper is ingested, this module can:
1) repair missing metadata,
2) merge near-duplicate records around the new paper,
3) rebuild embeddings for affected papers.
"""

import os
from typing import Dict, List, Optional, Set

from src import database, dedupe, enrich, scraper


DEFAULT_PIPELINE_DEDUPE_THRESHOLD = float(os.getenv("ARXIV_ASSISTANT_PIPELINE_DEDUPE_THRESHOLD", "0.84"))
DEFAULT_PIPELINE_REBUILD = os.getenv("ARXIV_ASSISTANT_PIPELINE_REBUILD_VECTORS", "1") == "1"


def _norm_title(title: str) -> str:
    return str(title or "").strip().lower()


def _is_unknown(value) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0 or value == ["Unknown"]
    return str(value).strip().lower() in {"", "unknown", "none", "null"}


def _needs_metadata_fix(paper: Dict) -> bool:
    return (
        _is_unknown(paper.get("authors"))
        or _is_unknown(paper.get("published"))
        or _is_unknown(paper.get("doi"))
        or _is_unknown(paper.get("venue"))
    )


def _get_paper_by_title(title: str, papers: Optional[List[Dict]] = None) -> Optional[Dict]:
    if not title:
        return None
    target = _norm_title(title)
    hay = papers if papers is not None else database.get_all_papers()
    for p in hay:
        if _norm_title(p.get("title")) == target:
            return p
    return None


def _paper_identity(paper: Dict) -> str:
    return str(paper.get("canonical_id") or paper.get("entry_id") or paper.get("title") or "")


def _find_duplicate_group_for_title(title: str, min_score: float = 0.84) -> List[Dict]:
    papers = database.get_all_papers()
    target = _get_paper_by_title(title, papers=papers)
    if not target:
        return []

    group = [target]
    target_id = _paper_identity(target)
    for p in papers:
        if _paper_identity(p) == target_id:
            continue
        score = dedupe.score_duplicate_pair(target, p)
        if float(score.get("combined_score", 0.0)) >= float(min_score):
            group.append(p)

    uniq = []
    seen = set()
    for p in group:
        pid = _paper_identity(p)
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(p)
    return uniq if len(uniq) > 1 else []


def run_incremental_indexing(
    new_title: str,
    run_metadata_fix: bool = True,
    dedupe_threshold: float = None,
    rebuild_vectors: bool = None,
) -> Dict:
    """
    Run post-ingest maintenance pipeline centered on one newly ingested paper.
    """
    dedupe_threshold = float(
        DEFAULT_PIPELINE_DEDUPE_THRESHOLD if dedupe_threshold is None else dedupe_threshold
    )
    if rebuild_vectors is None:
        rebuild_vectors = DEFAULT_PIPELINE_REBUILD

    result = {
        "input_title": str(new_title or ""),
        "metadata_updates": 0,
        "dedupe_groups": 0,
        "dedupe_removed": 0,
        "reindexed_chunks": 0,
        "reindexed_titles": [],
        "errors": [],
    }

    base_title = str(new_title or "").strip()
    if not base_title:
        result["errors"].append("empty_title")
        return result

    affected_titles: Set[str] = {base_title}

    # Step 1: metadata fix (targeted)
    try:
        current = _get_paper_by_title(base_title)
        if run_metadata_fix and current and _needs_metadata_fix(current):
            resolved = scraper.resolve_paper_metadata(current.get("title", base_title))
            if resolved.get("found"):
                updated = enrich.update_paper_metadata(current.get("title", base_title), resolved)
                result["metadata_updates"] = int(updated)
                resolved_title = str(resolved.get("title", "")).strip()
                if resolved_title:
                    affected_titles.add(resolved_title)
    except Exception as e:
        result["errors"].append(f"metadata_fix_failed:{e}")

    # Step 2: dedupe around affected titles
    processed_signatures = set()
    for seed_title in list(affected_titles):
        try:
            group = _find_duplicate_group_for_title(seed_title, min_score=dedupe_threshold)
            if len(group) < 2:
                continue
            signature = tuple(sorted({dedupe.normalize_title(p.get("title", "")) for p in group}))
            if signature in processed_signatures:
                continue
            processed_signatures.add(signature)

            merged = dedupe.merge_duplicates_group(group)
            result["dedupe_groups"] += 1
            removed_count = len([x for x in merged.get("removed", []) if x[1]])
            result["dedupe_removed"] += removed_count

            keep = merged.get("keep") or {}
            keep_title = str(keep.get("title", "")).strip()
            if keep_title:
                affected_titles.add(keep_title)
        except Exception as e:
            result["errors"].append(f"dedupe_failed:{e}")

    # Step 3: rebuild vectors for affected titles still in DB
    if rebuild_vectors:
        for title in sorted(affected_titles):
            try:
                if not _get_paper_by_title(title):
                    continue
                count = int(database.reindex_chunks_by_title(title))
                if count > 0:
                    result["reindexed_chunks"] += count
                    result["reindexed_titles"].append(title)
            except Exception as e:
                result["errors"].append(f"reindex_failed:{title}:{e}")

    return result
