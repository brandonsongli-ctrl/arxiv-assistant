"""
Hybrid Retrieval Module

Supports vector, BM25, or hybrid retrieval with optional reranking.
"""

import os
import re
from typing import Dict, List
from src import database

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# Retrieval configuration
DEFAULT_MODE = os.getenv("ARXIV_ASSISTANT_RETRIEVAL_MODE", "hybrid").strip().lower()
HYBRID_ALPHA = float(os.getenv("ARXIV_ASSISTANT_HYBRID_ALPHA", "0.7"))
RERANKER_MODEL = os.getenv("ARXIV_ASSISTANT_RERANKER_MODEL", "").strip()
RERANKER_TOP_K = int(os.getenv("ARXIV_ASSISTANT_RERANKER_TOP_K", "20"))
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("ARXIV_ASSISTANT_HYBRID_CANDIDATE_MULTIPLIER", "3"))

# Runtime overrides (UI)
_override_mode = None
_override_alpha = None
_override_reranker = None
_override_reranker_top_k = None
_override_candidate_multiplier = None


def set_overrides(mode: str = None, alpha: float = None, reranker_model: str = None,
                  reranker_top_k: int = None, candidate_multiplier: int = None) -> None:
    global _override_mode, _override_alpha, _override_reranker, _override_reranker_top_k, _override_candidate_multiplier, _reranker
    _override_mode = mode
    _override_alpha = alpha
    if reranker_model is not None and reranker_model != _override_reranker:
        _reranker = None
    _override_reranker = reranker_model
    _override_reranker_top_k = reranker_top_k
    _override_candidate_multiplier = candidate_multiplier


def _get_mode() -> str:
    return (_override_mode or DEFAULT_MODE).strip().lower()


def _get_alpha() -> float:
    if _override_alpha is not None:
        return float(_override_alpha)
    return HYBRID_ALPHA


def _get_reranker_model() -> str:
    return (_override_reranker if _override_reranker is not None else RERANKER_MODEL).strip()


def _get_reranker_top_k() -> int:
    if _override_reranker_top_k is not None:
        return int(_override_reranker_top_k)
    return RERANKER_TOP_K


def _get_candidate_multiplier() -> int:
    if _override_candidate_multiplier is not None:
        return int(_override_candidate_multiplier)
    return HYBRID_CANDIDATE_MULTIPLIER

_bm25_index = None
_bm25_docs: List[str] = []
_bm25_metas: List[Dict] = []
_bm25_ids: List[str] = []
_bm25_count: int = -1

_reranker = None


def _looks_like_identifier(text: str) -> bool:
    t = str(text or "").strip().lower()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", t):
        return True
    if t.startswith("10.") and "/" in t:
        return True
    if "arxiv:" in t or "doi:" in t:
        return True
    return False


def auto_tune(query_text: str, n_results: int = 5) -> Dict:
    """
    Choose retrieval mode/top_k based on query intent.
    """
    q = str(query_text or "").strip()
    q_lower = q.lower()
    tokens = _tokenize(q)
    token_count = len(tokens)

    mode = "hybrid"
    candidate_multiplier = _get_candidate_multiplier()
    tuned_n_results = n_results
    reason = []

    if _looks_like_identifier(q):
        mode = "bm25"
        tuned_n_results = max(3, min(8, n_results))
        reason.append("identifier_query")
    elif token_count <= 2:
        mode = "bm25"
        tuned_n_results = max(5, n_results)
        reason.append("short_keyword_query")
    elif ("?" in q) or any(x in q_lower for x in ["why", "how", "compare", "difference", "tradeoff"]):
        mode = "hybrid"
        candidate_multiplier = max(candidate_multiplier, 4)
        tuned_n_results = max(n_results, 8)
        reason.append("question_or_reasoning_query")
    elif token_count >= 12:
        mode = "vector"
        tuned_n_results = max(n_results, 8)
        reason.append("long_semantic_query")
    else:
        mode = "hybrid"
        tuned_n_results = max(n_results, 6)
        reason.append("default_hybrid")

    return {
        "mode": mode,
        "n_results": tuned_n_results,
        "candidate_multiplier": candidate_multiplier,
        "reason": ",".join(reason) if reason else "default",
    }


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _ensure_bm25_index() -> bool:
    global _bm25_index, _bm25_docs, _bm25_metas, _bm25_ids, _bm25_count
    
    if BM25Okapi is None:
        return False
        
    current_count = database.get_chunk_count()
    if _bm25_index is None or _bm25_count != current_count:
        result = database.get_all_chunks()
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids = result.get("ids") or []
        
        if not docs:
            return False
            
        tokenized = [_tokenize(d) for d in docs]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_docs = docs
        _bm25_metas = metas
        _bm25_ids = ids if ids else [str(i) for i in range(len(docs))]
        _bm25_count = current_count
    
    return True


def _get_reranker():
    global _reranker
    model_name = _get_reranker_model()
    if not model_name:
        return None
    if _reranker is not None:
        return _reranker
        
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(model_name)
        return _reranker
    except Exception:
        return None


def _format_results(hits: List[Dict]) -> Dict:
    return {
        "documents": [[h["document"] for h in hits]] if hits else [[]],
        "metadatas": [[h["metadata"] for h in hits]] if hits else [[]],
        "distances": [[h["distance"] for h in hits]] if hits else [[]],
        "ids": [[h["id"] for h in hits]] if hits else [[]],
    }


def _vector_hits(query_text: str, n_results: int) -> List[Dict]:
    results = database.query_similar(query_text, n_results=n_results)
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    
    hits = []
    for i, doc in enumerate(docs):
        distance = distances[i] if i < len(distances) else 0.0
        vector_score = 1.0 / (1.0 + distance)
        hits.append({
            "id": ids[i] if i < len(ids) else str(i),
            "document": doc,
            "metadata": metas[i] if i < len(metas) else {},
            "vector_score": vector_score,
        })
    return hits


def _bm25_hits(query_text: str, n_results: int) -> List[Dict]:
    if not _ensure_bm25_index():
        return []
        
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
        
    scores = _bm25_index.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_results]
    
    hits = []
    for idx, score in ranked:
        hits.append({
            "id": _bm25_ids[idx],
            "document": _bm25_docs[idx],
            "metadata": _bm25_metas[idx],
            "bm25_score": float(score),
        })
    return hits


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [s / max_score for s in scores]


def _apply_rerank(query_text: str, hits: List[Dict]) -> List[Dict]:
    reranker = _get_reranker()
    if reranker is None or not hits:
        return hits
        
    top_k = min(_get_reranker_top_k(), len(hits))
    top_hits = hits[:top_k]
    
    pairs = [(query_text, h["document"][:1000]) for h in top_hits]
    
    try:
        scores = reranker.predict(pairs)
    except Exception:
        return hits
        
    scores = [float(s) for s in scores]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        normalized = [1.0 for _ in scores]
    else:
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        
    for hit, score in zip(top_hits, normalized):
        hit["final_score"] = score
        
    for hit in hits[top_k:]:
        if "final_score" not in hit:
            hit["final_score"] = hit.get("combined_score", 0.0)
            
    reranked = sorted(hits, key=lambda h: h.get("final_score", 0.0), reverse=True)
    return reranked


def query(query_text: str, n_results: int = 5, mode: str = None) -> Dict:
    """
    Unified retrieval entry point. Returns Chroma-like result structure.
    mode: "vector", "bm25", or "hybrid" (default).
    """
    if mode is None:
        mode = _get_mode()
        
    mode = mode.lower()

    local_candidate_multiplier = _get_candidate_multiplier()
    if mode == "auto":
        tuned = auto_tune(query_text, n_results=n_results)
        mode = tuned.get("mode", "hybrid")
        n_results = int(tuned.get("n_results", n_results))
        local_candidate_multiplier = int(tuned.get("candidate_multiplier", local_candidate_multiplier))
    
    if mode == "vector":
        return database.query_similar(query_text, n_results=n_results)
        
    if mode == "bm25":
        hits = _bm25_hits(query_text, n_results=n_results)
        bm25_scores = [h.get("bm25_score", 0.0) for h in hits]
        bm25_norm = _normalize_scores(bm25_scores)
        for h, score in zip(hits, bm25_norm):
            h["combined_score"] = score
            h["distance"] = (1.0 - score) / max(score, 1e-6)
        return _format_results(hits)
    
    # Hybrid fallback
    candidate_k = max(n_results * local_candidate_multiplier, n_results)
    vector_hits = _vector_hits(query_text, n_results=candidate_k)
    bm25_hits = _bm25_hits(query_text, n_results=candidate_k)
    
    bm25_scores = [h.get("bm25_score", 0.0) for h in bm25_hits]
    bm25_norm = _normalize_scores(bm25_scores)
    for h, score in zip(bm25_hits, bm25_norm):
        h["bm25_score_norm"] = score
        
    combined = {}
    for h in vector_hits:
        combined[h["id"]] = {
            **h,
            "bm25_score_norm": 0.0,
        }
        
    for h in bm25_hits:
        if h["id"] in combined:
            combined[h["id"]]["bm25_score_norm"] = max(
                combined[h["id"]].get("bm25_score_norm", 0.0),
                h.get("bm25_score_norm", 0.0),
            )
        else:
            combined[h["id"]] = {
                "id": h["id"],
                "document": h["document"],
                "metadata": h["metadata"],
                "vector_score": 0.0,
                "bm25_score_norm": h.get("bm25_score_norm", 0.0),
            }
    
    hits = []
    alpha = _get_alpha()
    for item in combined.values():
        vector_score = item.get("vector_score", 0.0)
        bm25_score = item.get("bm25_score_norm", 0.0)
        combined_score = (alpha * vector_score) + ((1.0 - alpha) * bm25_score)
        combined_score = max(0.0, min(1.0, combined_score))
        distance = (1.0 - combined_score) / max(combined_score, 1e-6)
        hits.append({
            "id": item["id"],
            "document": item["document"],
            "metadata": item["metadata"],
            "combined_score": combined_score,
            "distance": distance,
        })
        
    hits = sorted(hits, key=lambda h: h.get("combined_score", 0.0), reverse=True)
    hits = _apply_rerank(query_text, hits)
    
    return _format_results(hits[:n_results])
