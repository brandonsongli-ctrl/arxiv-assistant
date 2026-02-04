"""
Deduplication Module

Identifies and removes duplicate papers from the database.
Criteria:
1. Exact DOI match
2. High Text Similarity (Title + First Author)
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
from src import database
import re
from difflib import SequenceMatcher

def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    if not title: return ""
    return re.sub(r'[^a-zA-Z0-9]', '', title.lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_authors(authors) -> Set[str]:
    if not authors:
        return set()
    if isinstance(authors, list):
        raw = authors
    else:
        raw = re.split(r",| and ", str(authors))
    last_names = set()
    for a in raw:
        a = a.strip()
        if not a:
            continue
        if "," in a:
            last = a.split(",")[0].strip()
        else:
            parts = a.split()
            last = parts[-1] if parts else ""
        last = re.sub(r"[^a-zA-Z0-9]", "", last.lower())
        if last:
            last_names.add(last)
    return last_names


def _extract_year(published) -> int:
    if not published:
        return -1
    m = re.search(r"(\d{4})", str(published))
    return int(m.group(1)) if m else -1


def _title_similarity(t1: str, t2: str) -> float:
    norm1 = normalize_title(t1)
    norm2 = normalize_title(t2)
    if not norm1 or not norm2:
        return 0.0
    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    tokens1 = set(_tokenize(t1))
    tokens2 = set(_tokenize(t2))
    if not tokens1 or not tokens2:
        jacc = 0.0
    else:
        jacc = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    return max(seq_ratio, jacc)


def _author_similarity(a1, a2) -> float:
    s1 = _normalize_authors(a1)
    s2 = _normalize_authors(a2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _year_similarity(y1: int, y2: int) -> float:
    if y1 <= 0 or y2 <= 0:
        return 0.0
    diff = abs(y1 - y2)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    if diff == 2:
        return 0.4
    return 0.0


def _is_fuzzy_duplicate(p1: Dict, p2: Dict) -> bool:
    title_sim = _title_similarity(p1.get("title", ""), p2.get("title", ""))
    if title_sim < 0.8:
        return False
    author_sim = _author_similarity(p1.get("authors"), p2.get("authors"))
    year_sim = _year_similarity(_extract_year(p1.get("published")), _extract_year(p2.get("published")))
    
    combined = (0.7 * title_sim) + (0.2 * author_sim) + (0.1 * year_sim)
    return combined >= 0.85

def find_duplicates() -> List[List[Dict]]:
    """
    Find groups of duplicate papers.
    Returns: List of lists, where each inner list contains duplicate paper records.
    """
    all_papers = database.get_all_papers()
    
    # 1. Group by Canonical ID (if present)
    id_map = defaultdict(list)
    for paper in all_papers:
        cid = paper.get('canonical_id')
        if cid:
            id_map[cid].append(paper)
    
    duplicates = []
    for cid, papers in id_map.items():
        if cid and len(papers) > 1:
            duplicates.append(papers)
    
    # 2. Group by DOI
    doi_map = defaultdict(list)
    for paper in all_papers:
        doi = paper.get('doi')
        if doi and doi not in ['Unknown', 'None', '']:
            doi_map[doi].append(paper)
    
    # Add DOI duplicates
    for doi, papers in doi_map.items():
        if len(papers) > 1:
            duplicates.append(papers)
            
    # 3. Group by Normalized Title (for papers without DOI or different DOIs??)
    # Be careful not to double count
    processed_titles = {p['title'] for group in duplicates for p in group}
    
    title_map = defaultdict(list)
    for paper in all_papers:
        if paper['title'] in processed_titles:
            continue
        
        norm_title = normalize_title(paper['title'])
        if len(norm_title) > 10: # Ignore very short titles
            title_map[norm_title].append(paper)
            
    for norm_title, papers in title_map.items():
        if len(papers) > 1:
            duplicates.append(papers)
            
    # 3. Fuzzy match for near-duplicates using title + authors + year
    processed_titles = {p['title'] for group in duplicates for p in group}
    remaining = [p for p in all_papers if p['title'] not in processed_titles]
    
    if len(remaining) > 1:
        # Union-Find to build groups
        parent = list(range(len(remaining)))
        
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        
        def union(i, j):
            ri = find(i)
            rj = find(j)
            if ri != rj:
                parent[rj] = ri
        
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if _is_fuzzy_duplicate(remaining[i], remaining[j]):
                    union(i, j)
        
        groups = defaultdict(list)
        for i, paper in enumerate(remaining):
            groups[find(i)].append(paper)
        
        for group in groups.values():
            if len(group) > 1:
                duplicates.append(group)
    
    return duplicates

def merge_duplicates(duplicate_group: List[Dict], keep_strategy: str = "best_metadata"):
    """
    Merge a group of duplicates into one.
    Actually, we just keep one and delete the others for now.
    
    keep_strategy:
    - 'best_metadata': Keep the one with most complete metadata (DOI, Year, Authors)
    - 'latest': Keep the newest one added? (We don't track added date well)
    """
    if not duplicate_group:
        return
    
    # Score papers
    def score_paper(p):
        score = 0
        if p.get('doi') and p.get('doi') not in ['Unknown', '']: score += 10
        if p.get('published') and p.get('published') != 'Unknown': score += 5
        if p.get('authors') and len(p.get('authors')) > 0 and p.get('authors') != ['Unknown']: score += 5
        return score
        
    sorted_papers = sorted(duplicate_group, key=score_paper, reverse=True)
    
    keep_paper = sorted_papers[0]
    remove_papers = sorted_papers[1:]
    
    results = []
    for p in remove_papers:
        success = database.delete_paper_by_title(p['title'])
        results.append((p['title'], success))
        
    return keep_paper, results
