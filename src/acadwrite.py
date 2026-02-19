"""
Acadwrite - 反向找文献

Reverse citation finder: given an article draft, extract representative
search queries, search external sources in parallel, and rank results
by semantic similarity to the article text.

No LLM dependency. Uses:
  - src.search_utils for tokenization and stopwords
  - src.scraper for all source search functions and deduplication
  - src.database.get_embedding_model() for semantic reranking

Public API:
  extract_article_queries(text, n_queries=3) -> List[str]
  search_all_sources_parallel(queries, sources, max_per_query=8) -> List[Dict]
  rank_by_similarity(article_text, papers) -> List[Dict]
  run_acadwrite_pipeline(article_text, sources, n_queries=3, max_per_query=8) -> dict
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from src import scraper
from src.database import get_embedding_model
from src.search_utils import _STOPWORDS, _tokenize


_ACADEMIC_STOPWORDS = _STOPWORDS | {
    "paper", "study", "show", "shows", "shown", "result", "results",
    "section", "figure", "table", "model", "models", "approach",
    "method", "methods", "using", "used", "also", "however",
    "thus", "therefore", "since", "while", "within", "across",
    "based", "propose", "proposed", "analysis", "present",
    "consider", "considered", "provide", "provides", "given",
    "find", "finds", "found", "well", "both", "each", "such",
    "more", "than", "that", "they", "them", "then", "when",
    "were", "been", "have", "here", "there", "first", "second",
    "third", "case", "cases", "form", "forms", "type", "types",
    "term", "terms", "often", "most", "many", "some", "other",
    "others", "further", "new", "known", "main", "large", "small",
    "high", "low", "general", "specific", "important", "key",
}

_SOURCE_FUNCTIONS = {
    "arxiv":                    scraper.search_arxiv,
    "semanticscholar":          scraper.search_semantic_scholar,
    "nber":                     scraper.search_nber,
    "ssrn":                     scraper.search_ssrn,
    "google_scholar":           scraper.search_google_scholar,
}


def extract_article_queries(text: str, n_queries: int = 3) -> List[str]:
    """
    Extract n_queries representative search queries from article text using
    section-aware TF-IDF scoring and bigram/trigram recovery. No LLM required.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) < 50:
        return [text[:200]]

    # 1. Split into three logical sections
    n = len(text)
    sections = {
        "intro":      text[:int(n * 0.20)],
        "body":       text[int(n * 0.20):int(n * 0.80)],
        "conclusion": text[int(n * 0.80):]
    }

    # 2. Whole-text unigram term frequency
    all_tokens = _tokenize(text)
    tf = Counter(
        tok for tok in all_tokens
        if tok not in _ACADEMIC_STOPWORDS and len(tok) >= 4
    )

    if not tf:
        return [text[:200]]

    # 3. Per-section TF
    section_tf: Dict[str, Counter] = {}
    for name, sec_text in sections.items():
        toks = _tokenize(sec_text)
        section_tf[name] = Counter(
            tok for tok in toks
            if tok not in _ACADEMIC_STOPWORDS and len(tok) >= 4
        )

    # 4. Score each unigram: global_count * max_section_dominance
    global_total = sum(tf.values()) or 1
    scored: Dict[str, float] = {}
    for tok, global_count in tf.items():
        global_freq = global_count / global_total
        max_dominance = 0.0
        for name, stf in section_tf.items():
            sec_total = sum(stf.values()) or 1
            sec_freq = stf.get(tok, 0) / sec_total
            dominance = sec_freq / global_freq if global_freq > 0 else 0.0
            max_dominance = max(max_dominance, dominance)
        scored[tok] = global_count * max_dominance

    # 5. Top-20 seed terms
    top_seeds = sorted(scored, key=lambda t: scored[t], reverse=True)[:20]
    seed_set = set(top_seeds)

    # 6. Bigram and trigram recovery from raw text
    words = re.findall(r"[A-Za-z][a-z]+", text)
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    for i, w in enumerate(words):
        w_low = w.lower()
        if w_low in seed_set and i + 1 < len(words):
            w2 = words[i + 1].lower()
            if w2 not in _ACADEMIC_STOPWORDS and len(w2) >= 4:
                bigram_counts[(w_low, w2)] += 1
                if i + 2 < len(words):
                    w3 = words[i + 2].lower()
                    if w3 not in _ACADEMIC_STOPWORDS and len(w3) >= 4:
                        trigram_counts[(w_low, w2, w3)] += 1

    # 7. Build candidate phrases: trigrams > bigrams > unigrams
    candidate_phrases = []

    for (t1, t2, t3), cnt in trigram_counts.most_common(10):
        phrase = f"{t1} {t2} {t3}"
        candidate_phrases.append((cnt * 3, phrase, {t1, t2, t3}))

    for (t1, t2), cnt in bigram_counts.most_common(15):
        phrase = f"{t1} {t2}"
        candidate_phrases.append((cnt * 2, phrase, {t1, t2}))

    for tok in top_seeds:
        candidate_phrases.append((scored[tok], tok, {tok}))

    # 8. Greedy diverse selection
    candidate_phrases.sort(reverse=True)
    queries: List[str] = []
    covered: set = set()
    for _, phrase, tok_set in candidate_phrases:
        if len(queries) >= n_queries:
            break
        overlap = tok_set & covered
        if len(overlap) < len(tok_set):
            queries.append(phrase)
            covered |= tok_set

    return queries[:n_queries] if queries else [text[:150]]


def search_all_sources_parallel(
    queries: List[str],
    sources: List[str],
    max_per_query: int = 8,
) -> List[Dict]:
    """
    Search all selected sources for all queries concurrently.
    Returns a deduplicated list of paper dicts.
    """
    tasks = []
    for query in queries:
        for source in sources:
            fn = _SOURCE_FUNCTIONS.get(source)
            if fn:
                tasks.append((query, source, fn))

    if not tasks:
        return []

    all_results: List[Dict] = []
    max_workers = min(len(tasks), 12)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fn, query, max_per_query): (query, source)
            for query, source, fn in tasks
        }
        for future in as_completed(futures):
            try:
                results = future.result(timeout=20)
                if results:
                    real = [
                        r for r in results
                        if not str(r.get("title", "")).startswith("⚠️")
                    ]
                    all_results.extend(real)
            except Exception as exc:
                query, source = futures[future]
                print(f"acadwrite: {source}/{query!r} failed: {exc}")

    return scraper.dedupe_results(all_results)


def _make_article_snippet(text: str, max_chars: int = 1000) -> str:
    """Take first 700 + last 300 chars to represent the full article."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:700]
    tail = text[-300:]
    return head + " ... " + tail


def rank_by_similarity(article_text: str, papers: List[Dict]) -> List[Dict]:
    """
    Re-rank papers by cosine similarity between article text embedding and
    each paper's (title + abstract) embedding, using the existing model.
    """
    if not papers:
        return []

    model = get_embedding_model()
    snippet = _make_article_snippet(article_text, max_chars=1000)
    article_emb = model.encode(snippet, normalize_embeddings=True)

    paper_texts = []
    for p in papers:
        title = p.get("title") or ""
        abstract = (p.get("summary") or p.get("abstract") or "")[:500]
        paper_texts.append(f"{title}. {abstract}".strip())

    paper_embs = model.encode(
        paper_texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    # Cosine similarity: dot product of L2-normalized vectors
    sims = np.dot(paper_embs, article_emb)

    for paper, sim in zip(papers, sims):
        paper["acadwrite_score"] = float(sim)
        paper["acadwrite_relevance_pct"] = round(float(sim) * 100, 1)

    return sorted(papers, key=lambda p: p["acadwrite_score"], reverse=True)


def run_acadwrite_pipeline(
    article_text: str,
    sources: List[str],
    n_queries: int = 3,
    max_per_query: int = 8,
) -> dict:
    """
    Full Acadwrite pipeline:
      1. Extract search queries from article text (no LLM)
      2. Search all selected sources in parallel
      3. Deduplicate and re-rank by semantic similarity

    Returns:
        {
            "queries": List[str],
            "papers": List[Dict],  # ranked, with acadwrite_score attached
            "raw_count": int,
            "dedup_count": int,
        }
    """
    queries = extract_article_queries(article_text, n_queries=n_queries)
    if not queries:
        return {"queries": [], "papers": [], "raw_count": 0, "dedup_count": 0}

    papers = search_all_sources_parallel(queries, sources, max_per_query=max_per_query)
    raw_count = len(papers)

    ranked = rank_by_similarity(article_text, papers)

    return {
        "queries": queries,
        "papers": ranked,
        "raw_count": raw_count,
        "dedup_count": len(ranked),
    }
