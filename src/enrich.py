"""
Metadata Enrichment Module

Automatically enriches paper metadata (DOI, authors, year, journal) using external APIs.
"""

import time
import os
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import database, scraper

def get_papers_needing_enrichment(force_recheck: bool = False) -> List[Dict]:
    """
    Find papers needing enrichment.
    If force_recheck is True, returns ALL papers that are likely manual uploads 
    (have 'manual_' entry_id or .pdf source), even if they have metadata.
    """
    all_papers = database.get_all_papers()
    papers_to_enrich = []
    
    for paper in all_papers:
        # Condition 1: Missing Metadata
        is_missing_info = (
            paper.get('authors') in ['Unknown', ['Unknown'], None] or
            paper.get('published') in ['Unknown', None] or
            paper.get('doi') in ['Unknown', None, '']
        )
        
        # Condition 2: Force Recheck (Manual/Local Files only)
        # We don't recheck ArXiv papers as they come from API source of truth
        is_manual_candidate = False
        if force_recheck:
            entry_id = str(paper.get('entry_id', ''))
            source = str(paper.get('source', ''))
            if entry_id.startswith('manual_') or source.endswith('.pdf'):
                is_manual_candidate = True
        
        if is_missing_info or (force_recheck and is_manual_candidate):
            papers_to_enrich.append(paper)
    
    return papers_to_enrich


def update_paper_metadata(title: str, new_metadata: Dict) -> int:
    """
    Update metadata for all chunks belonging to a paper (identified by title).
    Returns the number of chunks updated.
    """
    return database.update_paper_metadata_by_title(title, new_metadata)

# =============================================================================
# OPTIMIZED PDF EXTRACTION
# =============================================================================
# ... (Methods get_pdf_first_page_text, extract_doi_candidates, etc. unchanged)


def get_pdf_first_page_text(pdf_path: str) -> Optional[str]:
    """Extract text from the first page of a PDF."""
    try:
        from pypdf import PdfReader
        if not os.path.exists(pdf_path): return None
        
        reader = PdfReader(pdf_path)
        if len(reader.pages) > 0:
            return reader.pages[0].extract_text()
    except Exception:
        pass
    return None

def extract_doi_candidates(text: str) -> List[str]:
    """
    Find DOI strings in text.
    Returns a list of raw DOI strings found, e.g. ["10.1234/abc"].
    """
    if not text: return []
    
    # Regex for standard DOIs
    doi_pattern = r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+'
    matches = re.findall(doi_pattern, text)
    
    valid_dois = []
    for doi in matches:
        # Cleanup trailing punctuation often caught by regex
        doi = doi.rstrip('.,;)]')
        
        # Filter out common false positives or partials
        if len(doi) < 7: continue
        
        # Filter out "DOI: " prefix if captured (regex usually doesn't, but checks are good)
        if "doi.org" in doi: 
            doi = doi.split("doi.org/")[-1]
            
        valid_dois.append(doi)
        
    return list(set(valid_dois)) # Unique candidates

def extract_title_candidates(text: str) -> List[str]:
    """
    Guess potential titles from the first page text.
    Returns TOP 2 probable titles.
    """
    if not text: return []
    
    candidates = []
    lines = text.split('\n')
    
    # Heuristic 1: PDF Metadata Title (if passed in, but we handle that in caller usually)
    # Here we look at text lines.
    
    cleaned_lines = []
    for line in lines[:20]: # Only look at top 20 lines
        line = line.strip()
        if len(line) < 3: continue
        
        # Filter junk
        line_lower = line.lower()
        junk_terms = [
            "arxiv", "issn", "vol.", "no.", "downloaded", "http", "pii:", "copyright", 
            "available at", "elsevier", "sciencedirect", "www.", "doi:",
            "received", "accepted", "published", "online", "open access",
            "working paper", "discussion paper", "series", "journal",
            "reproduction", "rights reserved", "commercial re-use", "permissions",
            "original unedited manuscript", "downloaded from", "https://",
            "university", "universite", "universit", "department", "school ", "institute", "laborator"
        ]
        if any(x in line_lower for x in junk_terms): continue
        if "abstract" in line_lower: break # Stop at abstract header
        if line_lower.startswith("introduction"): break 
        
        # Check if line is mostly numbers/dates/codes (e.g. "25-011")
        if re.match(r'^[\w\-\.\:\s,]+$', line):
             # Heuristic: if it looks like a code (more numbers/symbols than letters), skip
             digit_count = sum(c.isdigit() for c in line)
             if digit_count > len(line) * 0.4: continue
        
        cleaned_lines.append(line)
    
    # Strategy A: First clean line that looks like a title (Title Case or ALL CAPS)
    # Often title is the largest text, but we don't have font size here.
    # We assume it's one of the first non-junk lines.
    
    if cleaned_lines:
        candidates.append(cleaned_lines[0]) # First valid line
        if len(cleaned_lines) > 1:
            candidates.append(f"{cleaned_lines[0]} {cleaned_lines[1]}") # First two lines combined
        if len(cleaned_lines) > 2:
            # Try a longer context (Title + Authors typically follow)
            # Limit to first 4 lines to avoid polluting with abstract
            context_snippet = " ".join(cleaned_lines[:4]).replace('*', '')
            candidates.append(context_snippet)
            
    return candidates

def verify_title_in_text(title: str, text: str, threshold: float = 0.6) -> bool:
    """
    CRITICAL: Verify that the `title` (from external metadata) actually exists 
    within the PDF `text`. This validates that the metadata belongs to THIS paper, 
    not a cited paper.
    """
    if not title or not text: return False
    
    # Normalize
    def normalize(s):
        return re.sub(r'[^\w\s]', '', s.lower()).split()
        
    title_words = normalize(title)
    text_words = set(normalize(text)) # Set for fast lookup
    
    if not title_words: return False
    
    # Check what % of title words appear in the text
    # We use a high threshold because the title SHOULD be there.
    # However, OCR errors or slight formatting diffs mean we can't do exact string match.
    
    matched_count = sum(1 for w in title_words if w in text_words)
    match_ratio = matched_count / len(title_words)
    
    return match_ratio >= threshold


def enrich_single_paper(paper: Dict, force_recheck: bool = False) -> Tuple[str, bool, str]:
    """
    Enrich a single paper's metadata using Robust Verification.
    
    Strategy:
    1. Extract Text from PDF.
    2. Candidate 1: DOIs found in text.
       -> Lookup DOI.
       -> VERIFY: Does result title appear in PDF text?
       -> If yes, MATCH.
    3. Candidate 2: Heuristic Title from Text.
       -> Search Title.
       -> VERIFY: Does result title appear in PDF text?
       -> If yes, MATCH.
    4. Candidate 3: Filename.
       -> Search Filename.
       -> VERIFY: Does result title appear in PDF text? (Or loose match if PDF text reads failed)
       -> If yes, MATCH.
    """
    
    current_title = paper.get('title', 'Unknown')
    source_path = paper.get('source')
    
    # Resolve source path
    if not source_path or source_path == 'manual_upload' or not os.path.exists(str(source_path)):
        # Try to find file in default data dir
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, 'data', 'pdfs')
        possible_names = [
            paper.get('entry_id', '').replace('manual_', ''),
            current_title,
            current_title + ".pdf"
        ]
        for name in possible_names:
            if name and os.path.exists(os.path.join(data_dir, name)):
                source_path = os.path.join(data_dir, name)
                break
                
    # 1. Get Text Content
    pdf_text = None
    if source_path and str(source_path).endswith('.pdf'):
        pdf_text = get_pdf_first_page_text(source_path)
    
    # If no text, we can't perform robust verification, fall back to simple filename search
    # But usually we have text.
    
    candidates = [] # List of (query, type)
    
    if pdf_text:
        # Priority 1: Title Candidates from text (Highest probability of being the actual paper)
        titles = extract_title_candidates(pdf_text)
        for t in titles:
            candidates.append((t, "Title Prediction from PDF"))
            
        # Priority 2: DOIs on page (Good, but risk being citations)
        dois = extract_doi_candidates(pdf_text)
        for d in dois:
            candidates.append((f"DOI:{d}", "DOI from PDF"))
            
    # Priority 3: Filename - REMOVED per user request
    def clean_filename(fname):
        if not fname: return ""
        name = os.path.basename(str(fname)).replace('.pdf', '')
        # Only replace obvious separators. 
        # DO NOT split numbers/letters as it ruins UUIDs (e.g. "9b79" -> "9 b 79")
        name = name.replace("_", " ").replace("-", " ")
        # Optional: Split CamelCase only if mostly letters? 
        # For now, keep it simple to avoid garbling.
        return name.strip()
        
    fname_clean = ""
    if source_path:
        fname_clean = clean_filename(source_path)
        # ERROR CORRECTION: Do NOT search by filename.
        # candidates.append((fname_clean, "Filename"))
    
    # Execute Search & Verification Loop
    for query, source_type in candidates:
        if not query or len(query) < 5: continue
        
        try:
            # SEARCH
            resolved = scraper.resolve_paper_metadata(query)
            
            if resolved.get('found'):
                found_title = resolved['title']
                
                # VERIFY
                # If we have PDF text, we MUST verify the found title exists in it.
                if pdf_text:
                    if verify_title_in_text(found_title, pdf_text):
                        # Match Confirmed!
                        chunks_updated = update_paper_metadata(current_title, resolved)
                        return (current_title, True, f"Matched via {source_type}: {found_title[:50]}...")
                    else:
                        continue 
                else:
                    # No PDF text? Skip if we want strict mode, or trust query if derived from filename?
                    # If we don't have text, we can't do much.
                    pass
                    
        except Exception as e:
            continue
            
    # If we are here, NO robust match was found via API.
    
    # NEW FEATURE: Local Metadata Extraction Fallback
    # If we have text, we should NOT leave it as "Unknown". We trust our local extraction more than "Unknown".
    if pdf_text:
        # 1. Try to extract meaningful metadata from the text itself
        local_title = ""
        local_authors = ["Unknown"]
        
        # Re-use extraction logic to get best title guess
        cleaned_lines = []
        lines = pdf_text.split('\n')
        for line in lines[:20]:
            line = line.strip()
            if len(line) < 3: continue
            line_lower = line.lower()
            junk_terms = [
                "arxiv", "issn", "vol.", "no.", "downloaded", "http", "pii:", "copyright", 
                "available at", "elsevier", "sciencedirect", "www.", "doi:",
                "received", "accepted", "published", "online", "open access",
                "working paper", "discussion paper", "series", "journal",
                "reproduction", "rights reserved", "commercial re-use", "permissions",
                "original unedited manuscript", "downloaded from", "https://",
                "university", "universite", "universit", "department", "school ", "institute", "laborator"
            ]
            if any(x in line_lower for x in junk_terms): continue
            if "abstract" in line_lower: break
            if line_lower.startswith("introduction"): break
            if re.match(r'^[\w\-\.\:\s,]+$', line):
                 digit_count = sum(c.isdigit() for c in line)
                 if digit_count > len(line) * 0.4: continue
            cleaned_lines.append(line)
            
        if cleaned_lines:
            local_title = cleaned_lines[0].replace('*', '').strip()
            # Heuristic: 2nd line might be author?
            if len(cleaned_lines) > 1:
                # Attempt to split by commas or 'and'
                a_line = cleaned_lines[1]
                if "," in a_line:
                    local_authors = [a.strip() for a in a_line.split(",") if len(a.strip()) > 2]
                elif " and " in a_line:
                    local_authors = [a.strip() for a in a_line.split(" and ") if len(a.strip()) > 2]
                else:
                    local_authors = [a_line]
        
        # If we found a plausible title locally, use it!
        if local_title and len(local_title) > 5:
             # Construct local metadata object
             local_meta = {
                 'title': local_title,
                 'authors': local_authors,
                 'doi': 'Unknown', # We couldn't verify DOI
                 'published': 'Unknown',
                 'summary': pdf_text[:500].replace('\n', ' ') + "...", # Use start of text as summary
                 'entry_id': paper.get('entry_id'), # Keep original ID
                 'source': 'local_extraction'
             }
             
             # Save to DB
             database.update_paper_metadata_by_title(current_title, local_meta)
             return (local_title, True, f"Local Extraction (API Not Found): '{local_title}'")
             
    # If even local extraction failed (no text?), THEN fall back to reset/unknown logic
    if force_recheck and pdf_text:
        # Check integrity of current title (if we didn't use local extraction above for some reason)
        if not verify_title_in_text(current_title, pdf_text):
            if fname_clean:
                 reset_meta = {
                     'title': fname_clean,
                     'doi': 'Unknown',
                     'published': 'Unknown',
                     'authors': 'Unknown',
                     'summary': 'Metadata reset due to verification failure.'
                 }
                 database.update_paper_metadata_by_title(current_title, reset_meta)
                 return (fname_clean, True, f"PURGED Incorrect Metadata. Reset to: {fname_clean}")

    return (current_title, False, f"No robust match found. Scanned {len(candidates)} candidates.")


def enrich_all_papers(force_recheck: bool = False, progress_callback=None) -> Dict:
    """
    Enrich metadata for all papers with missing information (or ALL manual papers if force_recheck=True).
    
    Args:
        force_recheck: If True, re-process all manual uploads/local PDF papers even if they have metadata.
        progress_callback: Optional function(current, total, message) for progress updates
    
    Returns:
        Dict with 'total', 'success', 'failed', 'details'
    """
    papers_to_enrich = get_papers_needing_enrichment(force_recheck=force_recheck)
    total = len(papers_to_enrich)
    
    if total == 0:
        return {
            'total': 0, 'success': 0, 'failed': 0, 'details': [],
            'message': 'All papers already have complete metadata!'
        }
    
    results = {
        'total': total, 'success': 0, 'failed': 0, 'details': []
    }
    
    completed = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Pass force_recheck to enrich_single_paper
        futures = {executor.submit(enrich_single_paper, p, force_recheck): p for p in papers_to_enrich}
        
        for future in as_completed(futures):
            completed += 1
            title, success, message = future.result()
            
            if success: results['success'] += 1
            else: results['failed'] += 1
            
            results['details'].append({'title': title, 'success': success, 'message': message})
            
            if progress_callback:
                progress_callback(completed, total, f"Processing: {title[:40]}...")
                
            time.sleep(0.1)
    
    results['message'] = f"Enriched {results['success']}/{total} papers successfully."
    return results

