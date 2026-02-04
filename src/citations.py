"""
Citation Recommendation Module

Finds relevant papers to cite based on semantic similarity to user's text.
"""

from typing import List, Dict, Tuple
from src import retrieval
from src.bibtex import generate_bibtex_entry


def recommend_citations(text: str, n_results: int = 5) -> List[Dict]:
    """
    Find papers most relevant to the given text passage.
    
    Args:
        text: The paragraph or sentence to find citations for
        n_results: Number of recommendations to return
    
    Returns:
        List of paper dictionaries with relevance scores
    """
    if not text.strip():
        return []
    
    # Query Database for similar chunks
    results = retrieval.query(text, n_results=n_results * 3)
    
    if not results.get('metadatas') or not results['metadatas'][0]:
        return []
    
    # Aggregate by paper title (best score per paper)
    paper_scores = {}
    
    for i, meta in enumerate(results['metadatas'][0]):
        title = meta.get('title', 'Unknown')
        distance = results['distances'][0][i] if results.get('distances') else 0
        # Convert distance to similarity score (lower distance = higher similarity)
        similarity = 1.0 / (1.0 + distance)
        
        if title not in paper_scores or similarity > paper_scores[title]['similarity']:
            paper_scores[title] = {
                'title': title,
                'authors': meta.get('authors', 'Unknown'),
                'published': meta.get('published', 'Unknown'),
                'entry_id': meta.get('entry_id', ''),
                'doi': meta.get('doi', 'Unknown'),
                'similarity': similarity,
                'distance': distance
            }
    
    # Sort by similarity (descending) and take top n
    sorted_papers = sorted(paper_scores.values(), key=lambda x: x['similarity'], reverse=True)
    return sorted_papers[:n_results]


def format_citation_list(papers: List[Dict], include_bibtex: bool = False) -> str:
    """
    Format the citation recommendations as a readable string.
    
    Args:
        papers: List of paper recommendation dictionaries
        include_bibtex: Whether to include BibTeX entries
    
    Returns:
        Formatted string
    """
    if not papers:
        return "No relevant papers found."
    
    lines = []
    for i, paper in enumerate(papers, 1):
        score_pct = paper['similarity'] * 100
        lines.append(f"### {i}. {paper['title']}")
        lines.append(f"**Authors:** {paper['authors']}")
        lines.append(f"**Published:** {paper['published']}")
        lines.append(f"**Relevance Score:** {score_pct:.1f}%")
        
        if include_bibtex:
            lines.append("\n**BibTeX:**")
            lines.append(f"```bibtex\n{generate_bibtex_entry(paper)}\n```")
        
        lines.append("")
    
    return '\n'.join(lines)


def get_citation_context(text: str, n_results: int = 3) -> Tuple[List[Dict], str]:
    """
    Get citation recommendations along with relevant text excerpts.
    
    Args:
        text: The text to find citations for
        n_results: Number of recommendations
    
    Returns:
        Tuple of (paper list, context string with excerpts)
    """
    # Query for chunks with documents
    results = retrieval.query(text, n_results=n_results * 2)
    
    if not results.get('documents') or not results['documents'][0]:
        return [], "No relevant content found."
    
    # Build context with excerpts
    paper_excerpts = {}
    
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        title = meta.get('title', 'Unknown')
        distance = results['distances'][0][i] if results.get('distances') else 0
        similarity = 1.0 / (1.0 + distance)
        
        if title not in paper_excerpts:
            paper_excerpts[title] = {
                'title': title,
                'authors': meta.get('authors', 'Unknown'),
                'published': meta.get('published', 'Unknown'),
                'entry_id': meta.get('entry_id', ''),
                'doi': meta.get('doi', 'Unknown'),
                'similarity': similarity,
                'excerpts': []
            }
        
        # Add excerpt (truncated for readability)
        excerpt = doc[:500] + "..." if len(doc) > 500 else doc
        paper_excerpts[title]['excerpts'].append(excerpt)
    
    # Sort and limit
    sorted_papers = sorted(paper_excerpts.values(), key=lambda x: x['similarity'], reverse=True)
    top_papers = sorted_papers[:n_results]
    
    # Format context
    context_lines = []
    for paper in top_papers:
        context_lines.append(f"**From: {paper['title']}**")
        for excerpt in paper['excerpts'][:2]:  # Max 2 excerpts per paper
            context_lines.append(f"> {excerpt}")
        context_lines.append("")
    
    return top_papers, '\n'.join(context_lines)
