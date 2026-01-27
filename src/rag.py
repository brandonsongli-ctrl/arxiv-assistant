import os
from openai import OpenAI
from src import database

# LLM Configuration
# Priority: OPENAI_API_KEY (cloud) -> OLLAMA_HOST (local)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")

# Determine which LLM backend to use
if OPENAI_API_KEY:
    # Use OpenAI API (cloud deployment)
    client = OpenAI(api_key=OPENAI_API_KEY)
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_AVAILABLE = True
    LLM_BACKEND = "openai"
else:
    # Try local Ollama
    try:
        client = OpenAI(base_url=OLLAMA_HOST, api_key='ollama')
        LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
        # Test connection
        client.models.list()
        LLM_AVAILABLE = True
        LLM_BACKEND = "ollama"
    except Exception:
        client = None
        LLM_MODEL = None
        LLM_AVAILABLE = False
        LLM_BACKEND = "none"

def query_db(query_text: str, n_results: int = 5):
    """
    Search the vector database for relevant chunks.
    Wraps database.query_similar to maintain API compatibility.
    """
    return database.query_similar(query_text, n_results=n_results)

def format_context(results) -> str:
    """
    Format the retrieval results into a string context.
    Adapts to the structure returned by database.query_similar.
    """
    context = ""
    if results.get('documents'):
        # results['documents'] is a list of lists (batch format)
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context += f"[Source: {meta.get('title', 'Unknown')}]\n{doc}\n\n"
    return context

def ask_llm(prompt: str, model: str = None) -> str:
    """
    Send a prompt to the LLM (OpenAI or Ollama).
    Returns error message if no LLM is available.
    """
    if not LLM_AVAILABLE:
        return "⚠️ LLM not available. Set OPENAI_API_KEY environment variable or run Ollama locally."
    
    if model is None:
        model = LLM_MODEL
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant specializing in Microeconomic Theory. You are concise and precise."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM ({LLM_BACKEND}): {str(e)}"

def rewrite_sentence(sentence: str, context: str = "", style: str = "academic") -> str:
    """
    Rewrite a sentence to be more academic, using retrieved context as style reference if available.
    """
    prompt = f"Rewrite the following sentence to be more academic and formal suitable for a paper on Microeconomic Theory.\n\nSentence: {sentence}\n"
    if context:
        prompt += f"\nReference style/content from these papers:\n{context}"
    
    return ask_llm(prompt)

def generate_ideas(topic: str, context: str) -> str:
    """
    Generate research ideas based on the topic and retrieved literature.
    """
    prompt = f"""You are a research assistant specializing in Microeconomic Theory.

The user is interested in the topic: "{topic}"

Below are excerpts from academic papers in the user's local library that are MOST RELEVANT to this topic.
Your job is to identify research gaps or novel directions based ONLY on what is discussed (or notably missing) in these excerpts.

IMPORTANT RULES:
1. Do NOT invent or hallucinate papers, authors, or findings that are not mentioned below.
2. If the excerpts do not seem closely related to the user's topic, explicitly say so and suggest what types of papers might be needed.
3. Focus on SPECIFIC technical gaps, not generic suggestions like "study more empirically."
4. Reference the papers/excerpts when proposing ideas.

=== RETRIEVED LITERATURE EXCERPTS ===
{context}
=== END OF EXCERPTS ===

Based on the above, provide 3 specific research ideas or gaps. If the excerpts are not relevant enough, say so honestly."""
    return ask_llm(prompt)

