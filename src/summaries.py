"""
Paper Summary Module

Generates paper-level summary cards using retrieved chunks.
"""

from typing import Optional
from src import database
from src import rag


def generate_paper_summary(title: str, max_chunks: int = 5, max_chars_per_chunk: int = 1200) -> Optional[str]:
    """
    Generate a concise paper summary using the first N chunks of the paper.
    """
    if not rag.llm_status().get("available"):
        return None
    
    result = database.get_chunks_by_title(title)
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    
    if not docs:
        return None
        
    # Order by chunk_index when available
    paired = list(zip(docs, metas))
    paired.sort(key=lambda x: int(x[1].get("chunk_index", 0)) if x[1] else 0)
    
    snippets = []
    for doc, _ in paired[:max_chunks]:
        if not doc:
            continue
        snippets.append(doc[:max_chars_per_chunk])
    
    if not snippets:
        return None
        
    context = "\n\n".join(snippets)
    
    prompt = f"""You are an academic assistant. Summarize the paper below using only the provided excerpts.

Requirements:
1. 120-180 words.
2. Include 3 bullet points of key contributions.
3. Avoid speculation beyond the excerpts.

EXCERPTS:
{context}

Summary:"""
    return rag.ask_llm(prompt)
