"""
Diagnostics Utilities
"""

import importlib
import os
from typing import Dict
from src import database, rag
from src import retrieval


def run_startup_checks() -> Dict:
    checks = []
    
    # Database connectivity
    try:
        db_ok = database.test_connection()
        checks.append({"name": "Database", "status": "ok" if db_ok else "error", "detail": "ChromaDB connection"})
    except Exception as e:
        checks.append({"name": "Database", "status": "error", "detail": str(e)})
    
    # Embedding model
    try:
        _ = database.get_embedding_model()
        checks.append({"name": "Embedding Model", "status": "ok", "detail": database.EMBEDDING_MODEL})
    except Exception as e:
        checks.append({"name": "Embedding Model", "status": "error", "detail": str(e)})
    
    # LLM
    status = rag.llm_status()
    if status.get("available"):
        checks.append({"name": "LLM", "status": "ok", "detail": f"{status.get('backend')}:{status.get('model')}"})
    else:
        checks.append(
            {
                "name": "LLM",
                "status": "warning",
                "detail": "No LLM available (configure in sidebar LLM Settings, set OPENAI_API_KEY, or run Ollama)",
            }
        )
    
    # Optional deps
    for pkg in ["rank_bm25", "umap", "hdbscan", "plotly", "scholarly"]:
        try:
            importlib.import_module(pkg)
            checks.append({"name": f"Dep: {pkg}", "status": "ok", "detail": "available"})
        except Exception:
            checks.append({"name": f"Dep: {pkg}", "status": "warning", "detail": "not installed"})
    
    # Retrieval status
    checks.append({
        "name": "Retrieval",
        "status": "ok",
        "detail": f"mode={os.getenv('ARXIV_ASSISTANT_RETRIEVAL_MODE', 'hybrid')}"
    })
    
    return {"checks": checks}


def get_retrieval_defaults() -> Dict:
    return {
        "mode": os.getenv("ARXIV_ASSISTANT_RETRIEVAL_MODE", "hybrid"),
        "alpha": float(os.getenv("ARXIV_ASSISTANT_HYBRID_ALPHA", "0.7")),
        "reranker_model": os.getenv("ARXIV_ASSISTANT_RERANKER_MODEL", ""),
        "reranker_top_k": int(os.getenv("ARXIV_ASSISTANT_RERANKER_TOP_K", "20")),
        "candidate_multiplier": int(os.getenv("ARXIV_ASSISTANT_HYBRID_CANDIDATE_MULTIPLIER", "3")),
    }
