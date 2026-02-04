"""
Metadata Utilities

Provides canonical ID generation and helper parsers.
"""

import hashlib
import re
from typing import Optional, Dict


def extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
    return match.group(1) if match else None


def extract_openalex_id(text: str) -> Optional[str]:
    if not text:
        return None
    if "openalex.org" in text.lower():
        return text.rstrip("/").split("/")[-1]
    return None


def normalize_doi(doi: str) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi.lower() if doi else None


def compute_canonical_id(metadata: Dict) -> str:
    """
    Compute a canonical ID for a paper.
    Priority: DOI -> arXiv -> OpenAlex -> Title hash.
    """
    doi = normalize_doi(metadata.get("doi") or metadata.get("DOI"))
    if doi:
        return f"doi:{doi}"
    
    arxiv_id = metadata.get("arxiv_id") or extract_arxiv_id(str(metadata.get("entry_id", "")))
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    
    openalex_id = metadata.get("openalex_id") or extract_openalex_id(str(metadata.get("entry_id", "")))
    if openalex_id:
        return f"openalex:{openalex_id}"
    
    title = (metadata.get("title") or "").strip().lower()
    year = str(metadata.get("published") or "")
    basis = f"{title}:{year}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"title:{digest}"
