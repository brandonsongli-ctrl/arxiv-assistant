import os
import re
from openai import OpenAI
from src import database, retrieval

# LLM Configuration
# Priority:
# 1) user-provided OpenAI key (runtime)
# 2) OPENAI_API_KEY environment variable
# 3) local Ollama host

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "llama3.2"))
DEFAULT_BACKEND_PREFERENCE = os.getenv("ARXIV_ASSISTANT_LLM_BACKEND", "auto").strip().lower()

OPENAI_API_KEY = ""
OLLAMA_HOST = DEFAULT_OLLAMA_HOST
LLM_MODEL = None
LLM_AVAILABLE = False
LLM_BACKEND = "none"
LLM_BACKEND_PREFERENCE = DEFAULT_BACKEND_PREFERENCE
client = None


def configure_llm(
    openai_api_key: str = None,
    ollama_host: str = None,
    backend_preference: str = None,
    openai_model: str = None,
    ollama_model: str = None,
) -> dict:
    """
    Configure LLM backend at runtime.
    Returns status dict for UI display.
    """
    global OPENAI_API_KEY, OLLAMA_HOST, LLM_MODEL, LLM_AVAILABLE, LLM_BACKEND, LLM_BACKEND_PREFERENCE, client

    key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY", "")
    key = str(key or "").strip()

    host = ollama_host if ollama_host is not None else DEFAULT_OLLAMA_HOST
    host = str(host or DEFAULT_OLLAMA_HOST).strip()

    backend = backend_preference if backend_preference is not None else DEFAULT_BACKEND_PREFERENCE
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "openai", "ollama"}:
        backend = "auto"

    openai_model_name = str(openai_model or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    ollama_model_name = str(ollama_model or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL

    OPENAI_API_KEY = key
    OLLAMA_HOST = host
    LLM_BACKEND_PREFERENCE = backend

    # Prefer OpenAI when key exists and backend allows it.
    if backend in {"auto", "openai"} and key:
        client = OpenAI(api_key=key)
        LLM_MODEL = openai_model_name
        LLM_AVAILABLE = True
        LLM_BACKEND = "openai"
        return llm_status()

    if backend == "openai":
        client = None
        LLM_MODEL = openai_model_name
        LLM_AVAILABLE = False
        LLM_BACKEND = "none"
        return llm_status()

    # Try local Ollama when backend allows it.
    try:
        client = OpenAI(base_url=host, api_key="ollama")
        client.models.list()
        LLM_MODEL = ollama_model_name
        LLM_AVAILABLE = True
        LLM_BACKEND = "ollama"
    except Exception:
        client = None
        LLM_MODEL = ollama_model_name if backend == "ollama" else None
        LLM_AVAILABLE = False
        LLM_BACKEND = "none"

    return llm_status()


def llm_status() -> dict:
    return {
        "available": LLM_AVAILABLE,
        "backend": LLM_BACKEND,
        "model": LLM_MODEL,
        "ollama_host": OLLAMA_HOST,
        "backend_preference": LLM_BACKEND_PREFERENCE,
        "has_openai_key": bool(OPENAI_API_KEY),
    }


configure_llm()


DEFAULT_REFUSE_THRESHOLD = float(os.getenv("ARXIV_ASSISTANT_CONFIDENCE_REFUSE_THRESHOLD", "0.28"))
DEFAULT_DOWNGRADE_THRESHOLD = float(os.getenv("ARXIV_ASSISTANT_CONFIDENCE_DOWNGRADE_THRESHOLD", "0.48"))
DEFAULT_STRONG_EVIDENCE_SCORE = float(os.getenv("ARXIV_ASSISTANT_STRONG_EVIDENCE_SCORE", "0.55"))

def query_db(query_text: str, n_results: int = 5, mode: str = None):
    """
    Search the vector database for relevant chunks.
    Wraps database.query_similar to maintain API compatibility.
    """
    return retrieval.query(query_text, n_results=n_results, mode=mode)

def format_context(results) -> str:
    """
    Format the retrieval results into a string context.
    Adapts to the structure returned by database.query_similar.
    """
    context = ""
    if results.get('documents'):
        # results['documents'] is a list of lists (batch format)
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context += f"[Source: {meta.get('title', 'Unknown')}]\n{doc}\n\n"
    return context

def ask_llm(prompt: str, model: str = None) -> str:
    """
    Send a prompt to the LLM (OpenAI or Ollama).
    Returns error message if no LLM is available.
    """
    if not LLM_AVAILABLE:
        return (
            "⚠️ LLM not available. Configure OPENAI API Key in app sidebar (LLM Settings), "
            "set OPENAI_API_KEY environment variable, or run Ollama locally."
        )
    
    if model is None:
        model = LLM_MODEL
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant specializing in Microeconomic Theory. You are concise and precise."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM ({LLM_BACKEND}): {str(e)}"

def rewrite_sentence(sentence: str, context: str = "", style: str = "academic") -> str:
    """
    Rewrite a sentence to be more academic, using retrieved context as style reference if available.
    """
    style = (style or "academic").strip().lower()
    style_hint = {
        "journal": "journal article style (formal, precise, concise, third-person)",
        "working paper": "working paper style (clear, explanatory, slightly longer sentences)",
        "grant": "grant proposal style (persuasive, action-oriented, highlights impact)"
    }.get(style, "academic style (formal, precise)")
    
    prompt = f"Rewrite the following sentence in {style_hint}, suitable for a paper in Microeconomic Theory.\n\nSentence: {sentence}\n"
    if context:
        prompt += f"\nReference style/content from these papers:\n{context}"
    
    return ask_llm(prompt)

def generate_ideas(topic: str, context: str, structured: bool = False) -> str:
    """
    Generate research ideas based on the topic and retrieved literature.
    """
    structure_rules = ""
    if structured:
        structure_rules = """
Output format (Markdown):
## Research Gaps
1. Idea: ...
2. Idea: ...
3. Idea: ...

## Proposed Directions
1. Idea: ... | Motivation: ... | Method: ... | Contribution: ...
2. Idea: ... | Motivation: ... | Method: ... | Contribution: ...
3. Idea: ... | Motivation: ... | Method: ... | Contribution: ...

## Evidence Needed
- ...
- ...
"""
    
    prompt = f"""You are a research assistant specializing in Microeconomic Theory.

The user is interested in the topic: "{topic}"

Below are excerpts from academic papers in the user's local library that are MOST RELEVANT to this topic.
Your job is to identify research gaps or novel directions based ONLY on what is discussed (or notably missing) in these excerpts.

IMPORTANT RULES:
1. Do NOT invent or hallucinate papers, authors, or findings that are not mentioned below.
2. If the excerpts do not seem closely related to the user's topic, explicitly say so and suggest what types of papers might be needed.
3. Focus on SPECIFIC technical gaps, not generic suggestions like "study more empirically."
4. Reference the papers/excerpts when proposing ideas.
{structure_rules}

=== RETRIEVED LITERATURE EXCERPTS ===
{context}
=== END OF EXCERPTS ===

Based on the above, provide 3 specific research ideas or gaps. If the excerpts are not relevant enough, say so honestly."""
    return ask_llm(prompt)


def get_evidence(query_text: str, n_results: int = 8, mode: str = None) -> list:
    """
    Retrieve evidence chunks with normalized confidence scores.
    """
    results = query_db(query_text, n_results=n_results, mode=mode)
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    evidence = []
    for i, doc in enumerate(docs):
        distance = distances[i] if i < len(distances) else 1.0
        score = 1.0 / (1.0 + float(distance))
        meta = metas[i] if i < len(metas) else {}
        evidence.append({
            "label": f"S{i+1}",
            "id": ids[i] if i < len(ids) else str(i),
            "title": meta.get("title", "Unknown"),
            "chunk_index": meta.get("chunk_index"),
            "published": meta.get("published"),
            "venue": meta.get("venue"),
            "score": score,
            "text": doc,
        })
    return evidence


def assess_evidence_confidence(
    evidence: list,
    refuse_threshold: float = None,
    downgrade_threshold: float = None,
    strong_evidence_score: float = None,
) -> dict:
    """
    Score confidence from retrieved evidence and decide whether to answer.
    """
    if refuse_threshold is None:
        refuse_threshold = DEFAULT_REFUSE_THRESHOLD
    if downgrade_threshold is None:
        downgrade_threshold = DEFAULT_DOWNGRADE_THRESHOLD
    if strong_evidence_score is None:
        strong_evidence_score = DEFAULT_STRONG_EVIDENCE_SCORE

    refuse_threshold = float(max(0.0, min(1.0, refuse_threshold)))
    downgrade_threshold = float(max(refuse_threshold + 0.02, min(1.0, downgrade_threshold)))
    strong_evidence_score = float(max(0.0, min(1.0, strong_evidence_score)))

    if not evidence:
        return {
            "score": 0.0,
            "band": "low",
            "decision": "refuse",
            "reasons": ["no_evidence"],
            "metrics": {
                "count": 0,
                "top_score": 0.0,
                "mean_top3": 0.0,
                "strong_count": 0,
                "unique_titles": 0,
            },
            "thresholds": {
                "refuse": refuse_threshold,
                "downgrade": downgrade_threshold,
                "strong_evidence": strong_evidence_score,
            },
        }

    scores = [float(e.get("score", 0.0)) for e in evidence]
    scores = [max(0.0, min(1.0, s)) for s in scores]
    scores_sorted = sorted(scores, reverse=True)
    top_score = scores_sorted[0]
    top_n = scores_sorted[: min(3, len(scores_sorted))]
    mean_top3 = sum(top_n) / max(1, len(top_n))
    strong_count = sum(1 for s in scores if s >= strong_evidence_score)

    unique_titles = len({
        str(e.get("title", "")).strip().lower()
        for e in evidence
        if str(e.get("title", "")).strip()
    })
    coverage = min(1.0, unique_titles / 3.0) if unique_titles > 0 else 0.0
    strength = min(1.0, strong_count / 3.0)
    count_factor = min(1.0, len(scores) / 5.0)

    score = (0.45 * top_score) + (0.30 * mean_top3) + (0.15 * strength) + (0.10 * coverage)
    score *= (0.85 + 0.15 * count_factor)
    score = max(0.0, min(1.0, score))

    reasons = []
    if top_score < 0.22:
        reasons.append("very_low_top_evidence")
    if mean_top3 < 0.30:
        reasons.append("weak_average_evidence")
    if strong_count < 2:
        reasons.append("few_strong_evidence_chunks")
    if unique_titles < 2:
        reasons.append("low_source_diversity")

    if score < refuse_threshold or top_score < 0.20:
        band = "low"
        decision = "refuse"
    elif score < downgrade_threshold or strong_count < 2:
        band = "medium"
        decision = "downgrade"
    else:
        band = "high"
        decision = "answer"

    return {
        "score": round(score, 4),
        "band": band,
        "decision": decision,
        "reasons": reasons,
        "metrics": {
            "count": len(scores),
            "top_score": round(top_score, 4),
            "mean_top3": round(mean_top3, 4),
            "strong_count": strong_count,
            "unique_titles": unique_titles,
        },
        "thresholds": {
            "refuse": refuse_threshold,
            "downgrade": downgrade_threshold,
            "strong_evidence": strong_evidence_score,
        },
    }


def _short_evidence_snippet(text: str, max_chars: int = 220) -> str:
    snippet = str(text or "").replace("\n", " ").strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return snippet


def _build_refusal_answer(evidence: list, confidence: dict) -> str:
    lines = [
        "Insufficient high-confidence evidence in the local library to answer this question reliably.",
        "Available evidence is listed below without further synthesis:",
    ]
    for ev in evidence[:3]:
        lines.append(f"- [{ev['label']}] {_short_evidence_snippet(ev.get('text', ''))}")
    lines.append(
        f"Confidence={confidence.get('score', 0.0):.3f} ({confidence.get('band', 'low')}); decision=refuse."
    )
    return "\n".join(lines)


def _build_downgraded_answer(evidence: list, confidence: dict) -> str:
    lines = [
        "Evidence is partial, so this response is downgraded to direct, source-supported observations only:",
    ]
    for ev in evidence[:5]:
        lines.append(f"- [{ev['label']}] {_short_evidence_snippet(ev.get('text', ''))}")
    lines.append(
        f"Confidence={confidence.get('score', 0.0):.3f} ({confidence.get('band', 'medium')}); decision=downgrade."
    )
    return "\n".join(lines)


def _is_valid_tagged_answer(answer: str, max_source_index: int) -> bool:
    tags = re.findall(r"\[S(\d+)\]", str(answer or ""))
    if not tags:
        return False
    for t in tags:
        idx = int(t)
        if idx < 1 or idx > max_source_index:
            return False
    return True


def answer_with_evidence(
    query_text: str,
    n_results: int = 8,
    mode: str = None,
    confidence_policy: str = "auto",
    refuse_threshold: float = None,
    downgrade_threshold: float = None,
) -> dict:
    """
    Generate an answer with source tags [S1], [S2], ... and confidence.
    """
    evidence = get_evidence(query_text, n_results=n_results, mode=mode)
    confidence = assess_evidence_confidence(
        evidence,
        refuse_threshold=refuse_threshold,
        downgrade_threshold=downgrade_threshold,
    )
    if not evidence:
        return {
            "answer": "No relevant evidence found in the local library.",
            "evidence": [],
            "copy_markdown": "",
            "confidence": confidence,
            "decision": "refuse",
        }

    context_blocks = []
    for ev in evidence:
        context_blocks.append(f"[{ev['label']}] {ev['title']} (chunk {ev.get('chunk_index')})\n{ev['text']}")
    context = "\n\n".join(context_blocks)

    policy = str(confidence_policy or "auto").strip().lower()
    decision = confidence.get("decision", "answer")
    if policy in {"off", "disabled", "none"}:
        decision = "answer"

    if decision == "refuse":
        answer = _build_refusal_answer(evidence, confidence)
    elif decision == "downgrade":
        answer = _build_downgraded_answer(evidence, confidence)
    elif not LLM_AVAILABLE:
        fallback_lines = ["Evidence-only fallback (LLM unavailable):"]
        for ev in evidence[:4]:
            snippet = str(ev.get("text", "")).strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            fallback_lines.append(f"- [{ev['label']}] {snippet}")
        answer = "\n".join(fallback_lines)
    else:
        prompt = f"""Answer the user's question using only the evidence snippets below.

Question: {query_text}

Requirements:
1) Output 3-6 concise sentences in Markdown.
2) Every sentence must include one or more source tags like [S1], [S2].
3) Do not cite any source tag that does not exist.
4) If evidence is weak or conflicting, explicitly state uncertainty.
5) Do not infer facts that are not directly supported by the snippets.
6) If support is limited, keep claims narrow and conditional.

Evidence:
{context}
"""
        answer = ask_llm(prompt)
        if not _is_valid_tagged_answer(answer, max_source_index=len(evidence)):
            # Fallback to safe extractive mode when LLM output does not follow source-tag constraints.
            decision = "downgrade"
            answer = _build_downgraded_answer(evidence, confidence)

    source_lines = []
    for ev in evidence:
        conf = f"{ev['score']:.3f}"
        meta_parts = [f"[{ev['label']}]", ev["title"]]
        if ev.get("chunk_index") is not None:
            meta_parts.append(f"chunk {ev['chunk_index']}")
        if ev.get("venue"):
            meta_parts.append(str(ev["venue"]))
        source_lines.append(f"{' | '.join(meta_parts)} | confidence={conf}")

    conf_score = confidence.get("score", 0.0)
    conf_band = confidence.get("band", "unknown")
    conf_reasons = ", ".join(confidence.get("reasons", [])) or "none"
    copy_markdown = (
        f"## Confidence\n"
        f"- score: {conf_score:.3f}\n"
        f"- band: {conf_band}\n"
        f"- decision: {decision}\n"
        f"- reasons: {conf_reasons}\n\n"
        f"## Answer\n{answer}\n\n"
        "## Sources\n" + "\n".join(f"- {line}" for line in source_lines)
    )

    return {
        "answer": answer,
        "evidence": evidence,
        "copy_markdown": copy_markdown,
        "confidence": confidence,
        "decision": decision,
    }

