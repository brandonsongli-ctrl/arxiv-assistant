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
        
        venue = paper.get("venue")
        if venue and str(venue).strip() not in ["Unknown", "None", "N/A", ""]:
            entry["container-title"] = str(venue)

        source = paper.get("source")
        if source and isinstance(source, str) and source.startswith("http"):
            entry["URL"] = source
        
        entries.append(entry)
    
    return json.dumps(entries, ensure_ascii=True, indent=2)


def export_library_ris() -> str:
    papers = get_all_papers()
    lines = []
    for p in papers:
        lines.append("TY  - JOUR")
        title = p.get("title")
        if title:
            lines.append(f"TI  - {title}")
        authors = p.get("authors")
        if isinstance(authors, list):
            author_list = [str(a).strip() for a in authors if str(a).strip()]
        else:
            author_list = [a.strip() for a in str(authors or "").split(",") if a.strip()]
        for a in author_list:
            lines.append(f"AU  - {a}")
        year = _extract_year(p.get("published"))
        if year:
            lines.append(f"PY  - {year}")
        doi = p.get("doi")
        if doi and doi not in ["Unknown", "None", ""]:
            lines.append(f"DO  - {doi}")
        venue = p.get("venue")
        if venue and str(venue).strip() not in ["Unknown", "None", "N/A", ""]:
            lines.append(f"JO  - {venue}")
        source = p.get("source")
        if source and isinstance(source, str) and source.startswith("http"):
            lines.append(f"UR  - {source}")
        tags = p.get("tags")
        if tags:
            tag_list = [t.strip() for t in re.split(r"[;,]", str(tags)) if t.strip()]
            for t in tag_list:
                lines.append(f"KW  - {t}")
        if p.get("is_favorite"):
            lines.append("N1  - Favorite: true")
        if p.get("in_reading_list"):
            lines.append("N1  - ReadingList: true")
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


def export_library_zotero_json() -> str:
    papers = get_all_papers()
    out = []
    for p in papers:
        authors = p.get("authors")
        if isinstance(authors, list):
            author_list = [str(a).strip() for a in authors if str(a).strip()]
        else:
            author_list = [a.strip() for a in str(authors or "").split(",") if a.strip()]

        creators = []
        for a in author_list:
            if "," in a:
                parts = [x.strip() for x in a.split(",", 1)]
                creators.append({"creatorType": "author", "lastName": parts[0], "firstName": parts[1] if len(parts) > 1 else ""})
            else:
                parts = a.split()
                if len(parts) >= 2:
                    creators.append({"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
                else:
                    creators.append({"creatorType": "author", "name": a})

        tags = [t.strip() for t in re.split(r"[;,]", str(p.get("tags", "") or "")) if t.strip()]
        extra_parts = []
        if p.get("is_favorite"):
            extra_parts.append("Favorite: true")
        if p.get("in_reading_list"):
            extra_parts.append("ReadingList: true")
        if p.get("canonical_id"):
            extra_parts.append(f"CanonicalID: {p.get('canonical_id')}")

        out.append({
            "itemType": "journalArticle",
            "title": p.get("title", ""),
            "creators": creators,
            "date": str(_extract_year(p.get("published")) or ""),
            "publicationTitle": p.get("venue", ""),
            "DOI": p.get("doi", ""),
            "url": p.get("source", "") if str(p.get("source", "")).startswith("http") else "",
            "abstractNote": p.get("summary", ""),
            "tags": [{"tag": t} for t in tags],
            "extra": "; ".join(extra_parts),
        })
    return json.dumps(out, ensure_ascii=True, indent=2)
