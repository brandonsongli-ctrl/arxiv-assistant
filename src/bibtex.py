"""
BibTeX Export Module

Generates BibTeX entries from paper metadata stored in the database.
"""

import re
from typing import Dict, List
from src.database import get_all_papers


def sanitize_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    if not text:
        return ""
    # Replace special chars
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_cite_key(paper: Dict) -> str:
    """
    Generate a citation key from paper metadata.
    Format: FirstAuthorLastNameYear (e.g., Smith2024)
    """
    authors = paper.get('authors', 'Unknown')
    published = paper.get('published', 'Unknown')
    
    # Extract first author's last name
    if isinstance(authors, str):
        first_author = authors.split(',')[0].strip()
    elif isinstance(authors, list):
        first_author = authors[0] if authors else 'Unknown'
    else:
        first_author = 'Unknown'
    
    # Get last name (assume "First Last" or "Last, First" format)
    if ',' in first_author:
        last_name = first_author.split(',')[0].strip()
    else:
        parts = first_author.split()
        last_name = parts[-1] if parts else 'Unknown'
    
    # Clean last name for use as key
    last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    
    # Extract year
    year = 'Unknown'
    if published and published != 'Unknown':
        year_match = re.search(r'(\d{4})', str(published))
        if year_match:
            year = year_match.group(1)
    
    return f"{last_name}{year}"


def generate_bibtex_entry(paper: Dict) -> str:
    """
    Generate a BibTeX entry string for a single paper.
    
    Args:
        paper: Dictionary with keys: title, authors, published, entry_id, doi (optional)
    
    Returns:
        BibTeX formatted string
    """
    cite_key = generate_cite_key(paper)
    title = sanitize_latex(paper.get('title', 'Unknown Title'))
    
    # Format authors
    authors_raw = paper.get('authors', 'Unknown')
    if isinstance(authors_raw, list):
        authors = ' and '.join(authors_raw)
    else:
        # Assume comma-separated string
        author_list = [a.strip() for a in authors_raw.split(',')]
        authors = ' and '.join(author_list)
    authors = sanitize_latex(authors)
    
    # Extract year
    published = paper.get('published', 'Unknown')
    year = 'Unknown'
    if published and published != 'Unknown':
        year_match = re.search(r'(\d{4})', str(published))
        if year_match:
            year = year_match.group(1)
    
    # Build entry
    entry_id = paper.get('entry_id', '')
    doi = paper.get('doi', '')
    
    # Determine entry type and add arXiv-specific fields
    lines = [f"@article{{{cite_key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{authors}}},")
    lines.append(f"  year = {{{year}}},")
    
    # Add DOI if available
    if doi and doi != 'Unknown' and doi != 'None':
        lines.append(f"  doi = {{{doi}}},")
    
    # Add arXiv ID if it's an arXiv paper
    if entry_id and 'arxiv' in str(entry_id).lower():
        # Extract arXiv ID from entry_id URL or string
        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', str(entry_id))
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append(f"  archivePrefix = {{arXiv}},")
    
    lines.append("}")
    
    return '\n'.join(lines)


def export_library_bibtex() -> str:
    """
    Export all papers in the library as a single BibTeX file content.
    
    Returns:
        Complete BibTeX file content as string
    """
    papers = get_all_papers()
    
    if not papers:
        return "% No papers in library"
    
    entries = []
    entries.append("% BibTeX export from Local Academic Assistant")
    entries.append(f"% Total papers: {len(papers)}")
    entries.append("")
    
    # Track used cite keys to avoid duplicates
    used_keys = {}
    
    for paper in papers:
        entry = generate_bibtex_entry(paper)
        
        # Handle duplicate keys by adding suffix
        cite_key = generate_cite_key(paper)
        if cite_key in used_keys:
            used_keys[cite_key] += 1
            # Modify entry to use unique key
            new_key = f"{cite_key}{chr(96 + used_keys[cite_key])}"  # a, b, c...
            entry = entry.replace(f"@article{{{cite_key},", f"@article{{{new_key},", 1)
        else:
            used_keys[cite_key] = 1
        
        entries.append(entry)
        entries.append("")
    
    return '\n'.join(entries)


def get_paper_bibtex(title: str) -> str:
    """
    Get BibTeX entry for a specific paper by title.
    
    Args:
        title: Paper title to search for
    
    Returns:
        BibTeX entry string or error message
    """
    papers = get_all_papers()
    
    for paper in papers:
        if paper.get('title', '').lower() == title.lower():
            return generate_bibtex_entry(paper)
    
    return f"% Paper not found: {title}"
