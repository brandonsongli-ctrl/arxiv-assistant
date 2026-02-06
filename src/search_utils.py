"""
Search utilities: query expansion, term extraction, snippets, and highlighting.
"""

import re
import html
from typing import Dict, List, Tuple


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with", "without", "via",
    "we", "our", "their", "his", "her", "its", "this", "these", "those", "can",
    "may", "might", "should", "would", "could", "what", "which", "when", "where",
    "who", "whom", "why", "how", "about", "into", "over", "under", "between",
}


_EXPANSION_RULES: List[Tuple[str, List[str]]] = [
    ("mechanism design", ["incentive compatibility", "strategyproof", "revelation principle", "allocation", "mechanism"]),
    ("auction", ["bidding", "reserve price", "mechanism design", "optimal auction"]),
    ("information design", ["bayesian persuasion", "signaling", "disclosure", "information structure"]),
    ("persuasion", ["information design", "bayesian persuasion", "signaling"]),
    ("contract theory", ["principal agent", "moral hazard", "adverse selection", "incentive"]),
    ("moral hazard", ["principal agent", "incentive", "hidden action"]),
    ("adverse selection", ["principal agent", "hidden information", "screening"]),
    ("screening", ["adverse selection", "menu", "self selection"]),
    ("matching", ["market design", "stable matching", "two-sided"]),
    ("public goods", ["free riding", "lindahl", "provision"]),
    ("multi-item", ["bundling", "menu", "allocation"]),
    ("dynamic", ["intertemporal", "sequential"]),
    ("bayesian", ["beliefs", "posterior", "prior"]),
    ("signaling", ["information design", "bayesian persuasion"]),
    ("contract", ["principal agent", "incentive", "screening"]),
]


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def extract_query_terms(query: str, min_len: int = 3) -> List[str]:
    """
    Extract content terms from a query string for highlighting/snippets.
    """
    tokens = _tokenize(query)
    cleaned = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        if len(tok) < min_len:
            continue
        cleaned.append(tok)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for tok in cleaned:
        if tok not in seen:
            ordered.append(tok)
            seen.add(tok)
    return ordered


def expand_query(query: str, max_terms: int = 8) -> Dict:
    """
    Expand a query with a small, domain-aware synonym list.
    Returns: {expanded_query, extra_terms, rules_matched}
    """
    if not query:
        return {"expanded_query": query, "extra_terms": [], "rules_matched": []}

    q_lower = " ".join(_tokenize(query))
    extra_terms: List[str] = []
    matched_rules: List[str] = []

    for key, expansions in _EXPANSION_RULES:
        if key in q_lower:
            matched_rules.append(key)
            for term in expansions:
                if term not in extra_terms:
                    extra_terms.append(term)
                if len(extra_terms) >= max_terms:
                    break
        if len(extra_terms) >= max_terms:
            break

    if not extra_terms:
        return {"expanded_query": query, "extra_terms": [], "rules_matched": []}

    expanded_query = f"{query} " + " ".join(extra_terms)
    return {
        "expanded_query": expanded_query,
        "extra_terms": extra_terms[:max_terms],
        "rules_matched": matched_rules
    }


def extract_snippet(text: str, terms: List[str], window: int = 360) -> str:
    """
    Extract a short snippet around the first matched term.
    """
    if not text:
        return ""
    if not terms:
        snippet = text[:window].strip()
        return snippet + ("..." if len(text) > window else "")

    lower = text.lower()
    best_idx = None
    for term in terms:
        idx = lower.find(term.lower())
        if idx != -1:
            best_idx = idx if best_idx is None else min(best_idx, idx)

    if best_idx is None:
        snippet = text[:window].strip()
        return snippet + ("..." if len(text) > window else "")

    start = max(0, best_idx - window // 2)
    end = min(len(text), start + window)
    snippet = text[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def highlight_text(text: str, terms: List[str]) -> str:
    """
    Highlight query terms in text using <mark>.
    """
    if not text or not terms:
        return html.escape(text or "")
    escaped_text = html.escape(text)
    safe_terms = [t for t in terms if t and len(t) >= 3]
    if not safe_terms:
        return escaped_text
    # Sort by length to avoid partial masking
    safe_terms = sorted(set(safe_terms), key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in safe_terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped_text)
