"""
Literature Review Generator Module

Generates comparative literature reviews from selected papers using the local LLM.
"""

from typing import List, Dict
import re
from .ingest import get_all_papers
from .rag import ask_llm

def get_papers_by_title(titles: List[str]) -> List[Dict]:
    """
    Retrieve full metadata for specific papers by title.
    """
    # This is inefficient for large DBs (fetching all then filtering), 
    # but fine for local use (<1000 papers). 
    # Optimization: If ingest.py supported get_by_title, we'd use that.
    all_papers = get_all_papers()
    target_papers = [p for p in all_papers if p['title'] in titles]
    return target_papers


def _format_references(papers: List[Dict]) -> str:
    if not papers:
        return ""
    lines = []
    for paper in papers:
        authors = paper.get('authors', 'Unknown')
        if isinstance(authors, list):
            author_text = ", ".join(authors)
        else:
            author_text = str(authors)
        year = "Unknown"
        published = paper.get('published', 'Unknown')
        if published and published != 'Unknown':
            match = re.search(r'(\d{4})', str(published))
            if match:
                year = match.group(1)
        title = paper.get('title', 'Unknown Title')
        doi = paper.get('doi', '')
        doi_text = f" DOI: {doi}." if doi and doi not in ['Unknown', 'None'] else ""
        lines.append(f"- {author_text} ({year}). {title}.{doi_text}")
    return "\n".join(lines)

def generate_literature_review(titles: List[str]) -> str:
    """
    Generate a comparative literature review for the selected paper titles.
    """
    if not titles:
        return "Please select at least one paper."
    
    papers = get_papers_by_title(titles)
    
    if not papers:
        return "Could not find metadata for the selected papers."

    # Construct context from summaries
    context = ""
    for i, paper in enumerate(papers, 1):
        context += f"PAPER {i}:\n"
        context += f"Title: {paper.get('title', 'Unknown')}\n"
        context += f"Authors: {paper.get('authors', 'Unknown')}\n"
        context += f"Year: {paper.get('published', 'Unknown')}\n"
        context += f"Abstract: {paper.get('summary', 'No summary available')}\n\n"

    # Prompt
    prompt = f"""You are an expert academic researcher.
    
Your task is to write a COMPARATIVE LITERATURE REVIEW based *only* on the following papers.
Group the papers by theme or approach, compare their methodologies/findings, and synthesize their contributions.
Do not just list them one by one; create a narrative.

{context}

Structure the review as follows:
1. **Introduction**: Brief overview of the topics covered.
2. **Thematic Analysis**: Compare and contrast the papers.
3. **Conclusion**: Summary of the state of this specific collection and potential gaps.

Write in a formal, academic tone.
"""

    review_text = ask_llm(prompt)
    refs = _format_references(papers)
    if refs:
        review_text += "\n\n---\n\n## References\n" + refs
    return review_text

def generate_deep_literature_review(titles: List[str] = None, topic: str = None) -> str:
    """
    Generate a deep agentic literature review.
    """
    from .agents import run_agentic_review
    review_text = run_agentic_review(topic, titles)
    if titles:
        papers = get_papers_by_title(titles)
        refs = _format_references(papers)
        if refs:
            review_text += "\n\n---\n\n## References\n" + refs
    return review_text
