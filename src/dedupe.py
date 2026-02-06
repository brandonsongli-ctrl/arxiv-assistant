"""
Deduplication module with exact and fuzzy matching plus merge helpers.
"""

from collections import defaultdict
from difflib import SequenceMatcher
import re
from typing import Dict, List, Set

from src import database


def normalize_title(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", title.lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _normalize_authors(authors) -> Set[str]:
    if not authors:
        return set()
    if isinstance(authors, list):
        raw = authors
    else:
        raw = re.split(r",| and ", str(authors))
    last_names = set()
    for a in raw:
        a = str(a).strip()
        if not a:
            continue
        if "," in a:
            last = a.split(",")[0].strip()
        else:
            parts = a.split()
            last = parts[-1] if parts else ""
        last = re.sub(r"[^a-zA-Z0-9]", "", last.lower())
        if last:
            last_names.add(last)
    return last_names


def _extract_year(published) -> int:
    if not published:
        return -1
    m = re.search(r"(\d{4})", str(published))
    return int(m.group(1)) if m else -1


def _title_similarity(t1: str, t2: str) -> float:
    norm1 = normalize_title(t1)
    norm2 = normalize_title(t2)
    if not norm1 or not norm2:
        return 0.0
    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    tokens1 = set(_tokenize(t1))
    tokens2 = set(_tokenize(t2))
    jacc = (len(tokens1 & tokens2) / len(tokens1 | tokens2)) if (tokens1 and tokens2) else 0.0
    return max(seq_ratio, jacc)


def _author_similarity(a1, a2) -> float:
    s1 = _normalize_authors(a1)
    s2 = _normalize_authors(a2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _year_similarity(y1: int, y2: int) -> float:
    if y1 <= 0 or y2 <= 0:
        return 0.0
    diff = abs(y1 - y2)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    if diff == 2:
        return 0.4
    return 0.0


def score_duplicate_pair(p1: Dict, p2: Dict) -> Dict:
    title_sim = _title_similarity(p1.get("title", ""), p2.get("title", ""))
    author_sim = _author_similarity(p1.get("authors"), p2.get("authors"))
    year_sim = _year_similarity(_extract_year(p1.get("published")), _extract_year(p2.get("published")))

    # Strong signal boost for exact DOI/canonical matches
    doi1 = str(p1.get("doi", "") or "").lower().strip()
    doi2 = str(p2.get("doi", "") or "").lower().strip()
    cid1 = str(p1.get("canonical_id", "") or "").lower().strip()
    cid2 = str(p2.get("canonical_id", "") or "").lower().strip()
    exact_boost = 0.0
    reasons = []
    if doi1 and doi2 and doi1 not in {"unknown", "none"} and doi1 == doi2:
        exact_boost = max(exact_boost, 0.25)
        reasons.append("exact_doi")
    if cid1 and cid2 and cid1 == cid2:
        exact_boost = max(exact_boost, 0.30)
        reasons.append("exact_canonical_id")

    combined = (0.65 * title_sim) + (0.25 * author_sim) + (0.10 * year_sim) + exact_boost
    combined = min(1.0, combined)
    if title_sim >= 0.92:
        reasons.append("very_high_title_similarity")
    if author_sim >= 0.6:
        reasons.append("author_overlap")
    if year_sim >= 0.7:
        reasons.append("year_close")

    return {
        "title_similarity": round(title_sim, 4),
        "author_similarity": round(author_sim, 4),
        "year_similarity": round(year_sim, 4),
        "combined_score": round(combined, 4),
        "reasons": reasons,
    }


def _looks_like_candidate(p1: Dict, p2: Dict) -> bool:
    t1 = normalize_title(p1.get("title", ""))
    t2 = normalize_title(p2.get("title", ""))
    if not t1 or not t2:
        return False
    if t1[:1] != t2[:1]:
        doi1 = str(p1.get("doi", "") or "").lower().strip()
        doi2 = str(p2.get("doi", "") or "").lower().strip()
        if not (doi1 and doi2 and doi1 == doi2):
            return False
    return True


def find_potential_duplicates(min_score: float = 0.84, max_pairs: int = 5000) -> List[Dict]:
    papers = database.get_all_papers()
    candidates: List[Dict] = []
    pair_count = 0
    n = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            if pair_count >= max_pairs:
                break
            p1 = papers[i]
            p2 = papers[j]
            if not _looks_like_candidate(p1, p2):
                continue
            score = score_duplicate_pair(p1, p2)
            pair_count += 1
            if score["combined_score"] < min_score:
                continue
            candidates.append({
                "paper_a": p1,
                "paper_b": p2,
                **score,
            })
        if pair_count >= max_pairs:
            break
    candidates.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return candidates


def group_duplicate_candidates(candidates: List[Dict]) -> List[List[Dict]]:
    if not candidates:
        return []

    paper_map: Dict[str, Dict] = {}
    keys = set()
    for c in candidates:
        a = c["paper_a"]
        b = c["paper_b"]
        ka = str(a.get("canonical_id") or a.get("entry_id") or a.get("title"))
        kb = str(b.get("canonical_id") or b.get("entry_id") or b.get("title"))
        keys.add(ka)
        keys.add(kb)
        paper_map[ka] = a
        paper_map[kb] = b

    keys = list(keys)
    idx_map = {k: i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for c in candidates:
        a = c["paper_a"]
        b = c["paper_b"]
        ka = str(a.get("canonical_id") or a.get("entry_id") or a.get("title"))
        kb = str(b.get("canonical_id") or b.get("entry_id") or b.get("title"))
        union(idx_map[ka], idx_map[kb])

    grouped = defaultdict(list)
    for k in keys:
        grouped[find(idx_map[k])].append(paper_map[k])

    return [g for g in grouped.values() if len(g) > 1]


def find_duplicates(min_score: float = 0.84) -> List[List[Dict]]:
    candidates = find_potential_duplicates(min_score=min_score)
    return group_duplicate_candidates(candidates)


def _score_paper_completeness(p: Dict) -> int:
    score = 0
    if p.get("doi") and p.get("doi") not in ["Unknown", "", None, "None"]:
        score += 10
    if p.get("published") and p.get("published") not in ["Unknown", "", None]:
        score += 5
    if p.get("authors") and p.get("authors") not in ["Unknown", ["Unknown"], "", None]:
        score += 5
    if p.get("summary") and p.get("summary") not in ["Unknown", "", None]:
        score += 3
    if p.get("venue") and p.get("venue") not in ["Unknown", "", None]:
        score += 2
    if p.get("is_favorite"):
        score += 1
    if p.get("in_reading_list"):
        score += 1
    return score


def propose_merged_metadata(group: List[Dict]) -> Dict:
    if not group:
        return {}
    sorted_group = sorted(group, key=_score_paper_completeness, reverse=True)
    best = dict(sorted_group[0])

    tags = set()
    favorite = False
    reading = False
    for p in group:
        for t in re.split(r"[;,]", str(p.get("tags", "") or "")):
            t = t.strip()
            if t:
                tags.add(t)
        if bool(p.get("is_favorite")):
            favorite = True
        if bool(p.get("in_reading_list")):
            reading = True
        if (not best.get("doi") or str(best.get("doi")).lower() in {"unknown", "none", ""}) and p.get("doi"):
            best["doi"] = p.get("doi")
        if (not best.get("venue") or str(best.get("venue")).lower() in {"unknown", "none", ""}) and p.get("venue"):
            best["venue"] = p.get("venue")
        if (not best.get("summary") or str(best.get("summary")).lower() in {"unknown", "none", ""}) and p.get("summary"):
            best["summary"] = p.get("summary")

    best["tags"] = ", ".join(sorted(tags))
    best["is_favorite"] = favorite
    best["in_reading_list"] = reading
    return best


def _delete_single_paper(paper: Dict) -> bool:
    entry_id = paper.get("entry_id")
    canonical_id = paper.get("canonical_id")
    title = paper.get("title")
    if entry_id:
        ok = database.delete_paper_by_metadata("entry_id", entry_id)
        if ok:
            return True
    if canonical_id:
        ok = database.delete_paper_by_metadata("canonical_id", canonical_id)
        if ok:
            return True
    if title:
        return database.delete_paper_by_title(title)
    return False


def merge_duplicates_group(group: List[Dict], keep_strategy: str = "best_metadata") -> Dict:
    if not group:
        return {"keep": None, "removed": []}

    sorted_group = sorted(group, key=_score_paper_completeness, reverse=True)
    keep = sorted_group[0]
    if keep_strategy == "latest":
        sorted_group = sorted(group, key=lambda p: str(p.get("entry_id", "")), reverse=True)
        keep = sorted_group[0]

    merged_metadata = propose_merged_metadata(group)

    # Update keeper first
    if keep.get("entry_id"):
        database.update_paper_metadata_by_field("entry_id", keep.get("entry_id"), merged_metadata)
    else:
        database.update_paper_metadata_by_title(keep.get("title", ""), merged_metadata)

    removed = []
    for p in group:
        if p is keep:
            continue
        ok = _delete_single_paper(p)
        removed.append((p.get("title", "Unknown"), ok))

    return {"keep": keep, "removed": removed, "merged_metadata": merged_metadata}


def merge_duplicates(duplicate_group: List[Dict], keep_strategy: str = "best_metadata"):
    result = merge_duplicates_group(duplicate_group, keep_strategy=keep_strategy)
    return result.get("keep"), result.get("removed", [])

