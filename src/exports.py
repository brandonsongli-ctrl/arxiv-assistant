"""
Export Utilities

Supports JSON and CSL-JSON exports.
"""

import json
import re
from typing import Dict, List
from src.database import get_all_papers


def _authors_to_csl(authors) -> List[Dict]:
    if not authors:
        return []
    if isinstance(authors, list):
        raw = authors
    else:
        raw = [a.strip() for a in str(authors).split(",") if a.strip()]
    output = []
    for a in raw:
        # Prefer literal to avoid mis-parsing names
        output.append({"literal": a})
    return output


def _extract_year(published) -> int:
    if not published:
        return None
    m = re.search(r"(\d{4})", str(published))
    return int(m.group(1)) if m else None


def export_library_json() -> str:
    papers = get_all_papers()
    return json.dumps(papers, ensure_ascii=True, indent=2)


def export_library_csl_json() -> str:
    papers = get_all_papers()
    entries = []
    
    for paper in papers:
        year = _extract_year(paper.get("published"))
        entry = {
            "id": paper.get("canonical_id") or paper.get("entry_id") or paper.get("title"),
            "type": "article-journal",
            "title": paper.get("title"),
            "author": _authors_to_csl(paper.get("authors")),
        }
        
        if year:
            entry["issued"] = {"date-parts": [[year]]}
        
        doi = paper.get("doi")
        if doi and doi not in ["Unknown", "None", ""]:
            entry["DOI"] = doi
        
        source = paper.get("source")
        if source and isinstance(source, str) and source.startswith("http"):
            entry["URL"] = source
        
        entries.append(entry)
    
    return json.dumps(entries, ensure_ascii=True, indent=2)
