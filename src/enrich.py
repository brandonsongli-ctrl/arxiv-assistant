"""
Metadata Enrichment Module

Automatically enriches paper metadata (DOI, authors, year, journal) using external APIs.
"""

import time
import os
import re
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
                
                def extract_candidates(lines, strict=True):
                    found = []
                    for line in lines: # Scan provided lines
                        line = line.strip()
                        if len(line) < 5: continue 
                        
                        line_lower = line.lower()
                        
                        # Junk keywords filters
                        if strict:
                            junk_terms = [
                                "arxiv", "issn", "vol.", "no.", "downloaded", "http", "pii:", "copyright", 
                                "journal of", "available at", "elsevier", "sciencedirect", "www.", "doi:",
                                "university", "department", "school", "college", "institute", "faculty",
                                "working paper", "discussion paper", "seminar", "conference", "draft", 
                                "january", "february", "march", "april", "may", "june", 
                                "july", "august", "september", "october", "november", "december",
                                "received", "accepted", "published", "online", "open access"
                            ]
                        else:
                            # Loose filter: only skip obvious metadata junk
                            junk_terms = ["http", "www", "doi:", "copyright", "downloaded"]

                        if any(x in line_lower for x in junk_terms): 
                            continue
                        
                        # Stop if we hit "Abstract" or "Introduction"
                        if "abstract" in line_lower or "introduction" in line_lower:
                            break
                            
                        found.append(line)
                        if len(found) >= 2: break 
                    return found

                # Pass 1: Strict Scan (Top 25 lines)
                clean_lines = extract_candidates(lines[:25], strict=True)
                
                # Pass 2: Loose Scan (if Strict failed) - Maybe title contained "University" or date?
                if not clean_lines:
                     clean_lines = extract_candidates(lines[:25], strict=False)

                # Validation
                if clean_lines:
                    candidate = " ".join(clean_lines)
                    
                    if "pii:" in candidate.lower() or "s0" in candidate.lower() and len(candidate) < 30:
                         return None
                    if len(candidate) < 10 or re.match(r'^[0-9\.\-\:\s]+$', candidate):
                        return None 
                    if not re.search(r'[a-zA-Z]', candidate):
                        return None
                        
                    return candidate
                    
    except Exception as e:
        pass
        
    return None


def enrich_single_paper(paper: Dict) -> Tuple[str, bool, str]:
    """
    Enrich a single paper's metadata.
    Returns: (title, success, message)
    """
    
    def clean_filename(fname):
        name = os.path.basename(fname).replace('.pdf', '')
        # Split CamelCase: "PennStBerkeley" -> "Penn St Berkeley"
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        # Replace separators
        name = name.replace("_", " ").replace("-", " ")
        return name

    current_title = paper.get('title', 'Unknown')
    source_path = paper.get('source')
    
    # Determine best search query
    # Default to improving the current title (which might be the filename)
    search_query = clean_filename(current_title) if current_title == os.path.basename(str(source_path or "")) else current_title
    
    # If we have a local PDF, try to extract a better title from it
    extracted_source = "Filename"
    if source_path and str(source_path).endswith('.pdf'):
        # Try advanced extraction
        content_query = extract_query_from_pdf(source_path)
        if content_query:
            search_query = content_query
            extracted_source = "PDF Content"
        else:
             # Fallback to smart filename cleaning logic
             search_query = clean_filename(source_path)
    
    try:
        resolved = resolve_paper_metadata(search_query)
        
        if resolved.get('found'):
            chunks_updated = update_paper_metadata(current_title, resolved)
            return (current_title, True, f"Updated {chunks_updated} chunks (Source: {extracted_source}, Matched: {resolved['title'][:30]}...)")
        else:
            return (current_title, False, f"No match. Query ({extracted_source}): '{search_query[:50]}...'")
    except Exception as e:
        return (current_title, False, f"Error: {str(e)}")


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
