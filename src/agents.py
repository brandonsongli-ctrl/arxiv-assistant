"""
Agentic Workflow Module

Handles specialized agents for deep academic analysis.
Includes roles: Retriever, Critic, and Synthesizer.
"""

from typing import List, Dict
from src import rag

class AcademicAgents:
    @staticmethod
    def retriever_agent(query: str) -> List[str]:
        """
        Analyzes the research query and generates a set of 
        sub-queries or related keywords to improve retrieval coverage.
        """
        prompt = f"""You are a 'Retriever Agent' specializing in Microeconomic Theory.
The user wants to research: "{query}"

Analyze this query and decompose it into 3-4 specific technical sub-questions or keyword sets that would help find a broad range of related literature.
Output ONLY the sub-queries, one per line, no numbering.
"""
        response = rag.ask_llm(prompt)
        # Parse lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return lines[:4]

    @staticmethod
    def critic_agent(excerpts: str) -> str:
        """
        Critically analyzes the retrieved excerpts for methodology, 
        rigor, and potential gaps.
        """
        prompt = f"""You are a 'Critic Agent' specializing in Microeconomic Theory.
Below are excerpts from several papers. Your job is to critically evaluate them:
1. What are the common methodological approaches?
2. Are there any visible biases or restrictive assumptions?
3. Where is the technical depth lacking in these specific snippets?

RETRIEVED LITERATURE:
{excerpts}

Provide a concise critical analysis (max 300 words).
"""
        return rag.ask_llm(prompt)

    @staticmethod
    def synthesizer_agent(query: str, critic_analysis: str, excerpts: str) -> str:
        """
        Synthesizes the raw data and critical analysis into a final high-quality review.
        """
        prompt = f"""You are a 'Synthesizer Agent'. Your goal is to write a Deep Literature Review.
Subject: {query}

Raw Evidence (Key Excerpts):
{excerpts}

Critical Evaluation:
{critic_analysis}

Final Task: Combine the evidence and the evaluation into a cohesive, narrative literature review. 
Do not just provide a list. Create a synthesis that shows the "state of the art," the tensions in the literature, and the path forward.

**CITATION RULE**: You MUST cite the papers using the format `(Author, Year)` whenever you reference their ideas.

Structure:
# Deep Literature Review: {query}
## 1. Overview and Core Contributions
## 2. Technical Synthesis & Methodologies
## 3. Critical Critique & Gaps
## 4. Conclusion & Research Outlook

Output in Markdown format.
"""
        return rag.ask_llm(prompt)

def run_agentic_review(query: str, selected_paper_titles: List[str] = None) -> str:
    """
    Coordinates the multi-agent process to generate a deep review.
    If selected_paper_titles is provided, it uses only those.
    Otherwise, it uses the Retriever agent to find papers.
    """
    from src import review, rag
    
    # 1. Gather excerpts
    excerpts = ""
    if selected_paper_titles:
        papers = review.get_papers_by_title(selected_paper_titles)
        for p in papers:
            excerpts += f"TITLE: {p['title']}\nSUMMARY: {p['summary']}\n\n"
    else:
        # Agentic search: use Retriever to get sub-queries
        sub_queries = AcademicAgents.retriever_agent(query)
        # Add the original query too
        all_queries = [query] + sub_queries
        
        seen_chunks = set()
        for q in all_queries:
            results = rag.query_db(q, n_results=3)
            # Avoid duplication
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                chunk_id = f"{meta.get('title')}_{meta.get('chunk_index')}"
                if chunk_id not in seen_chunks:
                    excerpts += f"[Source: {meta.get('title')}]\n{doc}\n\n"
                    seen_chunks.add(chunk_id)

    # 2. Critically evaluate
    critic_analysis = AcademicAgents.critic_agent(excerpts)
    
    # 3. Final synthesis
    final_review = AcademicAgents.synthesizer_agent(query, critic_analysis, excerpts)
    
    return final_review
