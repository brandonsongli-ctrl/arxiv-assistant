"""
Metadata Enrichment Module

Automatically enriches paper metadata (DOI, authors, year, journal) using external APIs.
"""

import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import database
from .scraper import resolve_paper_metadata


def get_papers_needing_enrichment() -> List[Dict]:
    """
    Find all papers with missing metadata (marked as 'Unknown').
    """
    all_papers = database.get_all_papers()
    papers_to_enrich = []
    
    for paper in all_papers:
        # Check if any key metadata is missing
        if (paper.get('authors') in ['Unknown', ['Unknown'], None] or
            paper.get('published') in ['Unknown', None] or
            paper.get('doi') in ['Unknown', None, '']):
            papers_to_enrich.append(paper)
    
    return papers_to_enrich


def update_paper_metadata(title: str, new_metadata: Dict) -> int:
    """
    Update metadata for all chunks belonging to a paper (identified by title).
    Returns the number of chunks updated.
    """
    return database.update_paper_metadata_by_title(title, new_metadata)


def extract_query_from_pdf(filepath: str) -> str:
    """
    Extract a search query from a PDF file.
    Priority:
    1. PDF Metadata Title (if meaningful)
    2. First few lines of text (cleaned)
    """
    import os
    from pypdf import PdfReader
    
    if not os.path.exists(filepath):
        return None
        
    try:
        reader = PdfReader(filepath)
        
        # 1. Try Metadata
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip()
            # Filter out bad metadata
            if len(title) > 5 and "untitled" not in title.lower() and ".pdf" not in title.lower():
                return title
                
        # 2. Try Text Content
        if len(reader.pages) > 0:
            first_page = reader.pages[0].extract_text()
            if first_page:
                # A. Check for DOI - but ONLY in top portion of page (avoid cited paper DOIs)
                import re
                # Only look at first ~1000 characters for DOI (usually paper's own DOI is at top)
                top_of_page = first_page[:1000]
                doi_match = re.search(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+', top_of_page)
                if doi_match:
                    doi = doi_match.group(0)
                    # Filter out common false positives (journal DOIs, review article DOIs that aren't this paper)
                    # Skip if DOI looks like a journal-level DOI (too short or generic patterns)
                    if len(doi) > 20 and not any(x in doi.lower() for x in ['issn', 'journal', '/j.', 'collection']):
                        return f"DOI:{doi}"
                    
                lines = first_page.split('\n')
                # Take first 2-3 non-empty lines that aren't junk
                clean_lines = []
                for line in lines[:25]: # Scan deeper (top 25 lines)
                    line = line.strip()
                    if len(line) < 5: continue # Skip very short lines
                    
                    line_lower = line.lower()
                    # Junk keywords filters
                    junk_terms = [
                        "arxiv", "issn", "vol.", "no.", "downloaded", "http", "pii:", "copyright", 
                        "journal of", "available at", "elsevier", "sciencedirect", "www.", "doi:",
                        "university", "department", "school", "college", "institute", "faculty",
                        "working paper", "discussion paper", "seminar", "conference", "draft", 
                        "january", "february", "march", "april", "may", "june", 
                        "july", "august", "september", "october", "november", "december",
                        "received", "accepted", "published", "online", "open access"
                    ]
                    
                    if any(x in line_lower for x in junk_terms): 
                        continue
                    
                    # Stop if we hit "Abstract" or "Introduction"
                    if "abstract" in line_lower or "introduction" in line_lower:
                        break
                        
                    clean_lines.append(line)
                    # Title usually spans 1-2 lines. If we got 2 good lines, stop to avoid picking up authors.
                    if len(clean_lines) >= 2: break 
                
                # Validation: If we only extracted weird codes or numbers, fail
                if clean_lines:
                    candidate = " ".join(clean_lines)
                    
                    # Reject if it looks like PII code even if keyword missed (e.g. S0022-...)
                    if "pii:" in candidate.lower() or "s0" in candidate.lower() and len(candidate) < 30:
                         return None
                         
                    # If candidate is too short or looks like just numbers/codes
                    if len(candidate) < 10 or re.match(r'^[0-9\.\-\:\s]+$', candidate):
                        return None 
                    
                    # Ensure it has letters
                    if not re.search(r'[a-zA-Z]', candidate):
                        return None
                        
                    return candidate
                    
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        
    return None


def enrich_single_paper(paper: Dict) -> Tuple[str, bool, str]:
    """
    Enrich a single paper's metadata.
    Returns: (title, success, message)
    """
    current_title = paper.get('title', 'Unknown')
    source_path = paper.get('source')
    
    # Determine best search query
    search_query = current_title
    
    # If we have a local PDF, try to extract a better title from it
    if source_path and str(source_path).endswith('.pdf'):
        # Just use simple filename cleaning as baseline
        # search_query = os.path.basename(source_path).replace('.pdf', '')
        
        # Try advanced extraction
        content_query = extract_query_from_pdf(source_path)
        if content_query:
            search_query = content_query
    
    try:
        resolved = resolve_paper_metadata(search_query)
        
        if resolved.get('found'):
            chunks_updated = update_paper_metadata(current_title, resolved)
            return (current_title, True, f"Updated {chunks_updated} chunks (Match: {resolved['title'][:30]}...)")
        else:
            return (current_title, False, f"No match for query: {search_query[:30]}...")
    except Exception as e:
        return (current_title, False, str(e))


def enrich_all_papers(progress_callback=None) -> Dict:
    """
    Enrich metadata for all papers with missing information.
    
    Args:
        progress_callback: Optional function(current, total, message) for progress updates
    
    Returns:
        Dict with 'total', 'success', 'failed', 'details'
    """
    papers_to_enrich = get_papers_needing_enrichment()
    total = len(papers_to_enrich)
    
    if total == 0:
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'details': [],
            'message': 'All papers already have complete metadata!'
        }
    
    results = {
        'total': total,
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    # Process in parallel for speed, but with rate limiting
    completed = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(enrich_single_paper, p): p for p in papers_to_enrich}
        
        for future in as_completed(futures):
            completed += 1
            title, success, message = future.result()
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append({
                'title': title,
                'success': success,
                'message': message
            })
            
            if progress_callback:
                progress_callback(completed, total, f"Processing: {title[:40]}...")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
    
    results['message'] = f"Enriched {results['success']}/{total} papers successfully."
    return results
