import arxiv
import os
from typing import List, Dict

def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Dict]:
    """
    Search arXiv for papers and return metadata.
    Supports sort_by: 'relevance', 'last_updated', 'submitted_date'
    If query looks like an ID, uses id_list.
    """
    client = arxiv.Client()
    import re
    id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
    
    criterion = arxiv.SortCriterion.Relevance
    if sort_by == "last_updated":
        criterion = arxiv.SortCriterion.LastUpdatedDate
    elif sort_by == "submitted_date":
        criterion = arxiv.SortCriterion.SubmittedDate

    results = [] # Store results here
    
    # 1. Try ID Search if it looks like an ID
    if re.match(id_pattern, query.strip()):
        try:
            id_search = arxiv.Search(id_list=[query.strip()])
            results = list(client.results(id_search))
        except Exception:
            # ID search failed (e.g. invalid ID format that regex caught), fall through to keyword
            pass
            
    # 2. If no results from ID search (or didn't try), do Keyword Search
    # Note: If user INTENDED an ID but got it wrong, keyword search might find it or nothing.
    if not results:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=criterion,
            sort_order=arxiv.SortOrder.Descending # Always want newest/most relevant at top
        )
        try:
            results = list(client.results(search))
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []

    output = []
    for result in results:
        output.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
            "published": str(result.published),
            "obj": result,
            "source": "arxiv"
        })
    return output


def download_paper(paper_obj, download_dir: str):
    """
    Download the PDF of a paper.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Paper title as filename (sanitized)
    filename = "".join(x for x in paper_obj.title if x.isalnum() or x in " -_").strip() + ".pdf"
    path = os.path.join(download_dir, filename)
    
    if not os.path.exists(path):
        paper_obj.download_pdf(dirpath=download_dir, filename=filename)
        print(f"Downloaded: {filename}")
    else:
        print(f"Already exists: {filename}")
    
    return path

def search_semantic_scholar(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search Semantic Scholar for papers using direct API.
    Handles rate limiting and errors gracefully.
    """
    import requests
    import time
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,url,openAccessPdf,publicationDate,year"
    }
    
    try:
        # 5 second timeout to prevent hanging
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            print("Semantic Scholar Rate Limit hit.")
            # Return empty with a specific error indicator in title or handle in UI
            # For now, let's return a dummy result that warns the user
            return [{
                "title": "⚠️ Error: Semantic Scholar Rate Limit Hit",
                "authors": ["System"],
                "summary": "Too many requests. Please wait a minute and try again.",
                "pdf_url": None,
                "entry_id": "error_rate_limit",
                "published": "Now",
                "source": "semanticscholar",
                "obj": None
            }]
            
        response.raise_for_status()
        data = response.json()
        
        output = []
        if 'data' not in data:
            return []
            
        for item in data['data']:
            # Check if there is a PDF link
            pdf_url = None
            if item.get('openAccessPdf'):
                pdf_url = item.get('openAccessPdf').get('url')
            
            authors_list = [a.get('name') for a in item.get('authors', [])] if item.get('authors') else ["Unknown"]
            
            output.append({
                "title": item.get('title', 'No Title'),
                "authors": authors_list,
                "summary": item.get('abstract') if item.get('abstract') else "No abstract available.",
                "pdf_url": pdf_url,
                "entry_id": item.get('paperId'),
                "published": str(item.get('publicationDate')) if item.get('publicationDate') else str(item.get('year')),
                "source": "semanticscholar",
                "obj": None
            })
        return output
        
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
        return [{
            "title": f"⚠️ Error: {str(e)}",
            "authors": ["System"],
            "summary": "An error occurred while communicating with Semantic Scholar.",
            "pdf_url": None,
            "entry_id": "error_generic",
            "published": "Now",
            "source": "semanticscholar",
            "obj": None
        }]

def download_from_url(url: str, title: str, download_dir: str) -> str:
    """
    Generic file downloader for Semantic Scholar URLs.
    """
    import requests
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    filename = "".join(x for x in title if x.isalnum() or x in " -_").strip() + ".pdf"
    path = os.path.join(download_dir, filename)
    
    if os.path.exists(path):
        print(f"Already exists: {filename}")
        return path
        
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            return path
        else:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def resolve_paper_metadata(query: str) -> Dict:
    """
    Try to find DOI and canonical metadata using multiple APIs.
    Falls back: Semantic Scholar -> OpenAlex -> CrossRef
    Supports queries starting with 'DOI:' for direct lookup.
    """
    import requests
    import time
    
    # Try Semantic Scholar first
    result = _try_semantic_scholar(query)
    if result.get("found") and _is_title_similar(query, result.get("title", "")):
        return result
    
    # Fallback to OpenAlex (no rate limit, very reliable)
    result = _try_openalex(query)
    if result.get("found") and _is_title_similar(query, result.get("title", "")):
        return result
        
    # Fallback to CrossRef
    result = _try_crossref(query)
    if result.get("found") and _is_title_similar(query, result.get("title", "")):
        return result
        
    return {"found": False}


def _is_title_similar(query: str, title: str, threshold: float = 0.2) -> bool:
    """
    Check if the returned title is similar enough to the query.
    Uses word-based Jaccard similarity.
    Low threshold (0.2) because:
      - Query is often just filename with underscores/dashes
      - Title is the full paper title
      - We just want to avoid completely wrong matches
    """
    if not query or not title:
        return False
    
    # Normalize: lowercase, remove punctuation, split into words
    import re
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        words = set(text.split())
        # Remove very short words and numbers-only tokens
        words = {w for w in words if len(w) > 2 and not w.isdigit()}
        return words
    
    query_words = normalize(query)
    title_words = normalize(title)
    
    if not query_words or not title_words:
        return False
    
    # Jaccard similarity: intersection / union
    intersection = query_words & title_words
    union = query_words | title_words
    
    similarity = len(intersection) / len(union) if union else 0
    
    # Debug output (can be removed in production)
    # print(f"Similarity '{query[:30]}...' vs '{title[:30]}...': {similarity:.2f}")
    
    return similarity >= threshold


def _try_semantic_scholar(query: str) -> Dict:
    """Try Semantic Scholar API."""
    import requests
    
    try:
        if query.strip().upper().startswith("DOI:"):
            doi = query.strip()[4:].strip()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
            params = {"fields": "title,authors,externalIds,year,publicationDate,abstract"}
        else:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": 1, "fields": "title,authors,externalIds,year,publicationDate,abstract"}
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 429:
            print("Semantic Scholar rate limited, trying fallback...")
            return {"found": False}
            
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                if not data['data']:
                    return {"found": False}
                item = data['data'][0]
            else:
                item = data
                
            doi = item.get('externalIds', {}).get('DOI') if item.get('externalIds') else None
            authors_list = [a.get('name') for a in item.get('authors', [])] if item.get('authors') else ["Unknown"]
            
            return {
                "title": item.get('title'),
                "authors": authors_list,
                "published": str(item.get('publicationDate') or item.get('year')),
                "doi": doi,
                "summary": item.get('abstract'),
                "found": True,
                "source": "semantic_scholar"
            }
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
    return {"found": False}


def _try_openalex(query: str) -> Dict:
    """Try OpenAlex API (free, no rate limit for reasonable use)."""
    import requests
    
    try:
        # OpenAlex search endpoint
        url = "https://api.openalex.org/works"
        
        if query.strip().upper().startswith("DOI:"):
            doi = query.strip()[4:].strip()
            params = {"filter": f"doi:{doi}"}
        else:
            # Clean query for search
            clean_query = query.replace("*", "").replace("†", "").replace("‡", "").strip()
            params = {"search": clean_query, "per_page": 1}
        
        headers = {"User-Agent": "ArxivAssistant/1.0 (mailto:research@example.com)"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                item = results[0]
                
                # Extract authors
                authors = []
                for authorship in item.get('authorships', []):
                    if authorship.get('author', {}).get('display_name'):
                        authors.append(authorship['author']['display_name'])
                if not authors:
                    authors = ["Unknown"]
                
                # Extract year
                year = item.get('publication_year') or item.get('publication_date', '')[:4] if item.get('publication_date') else "Unknown"
                
                return {
                    "title": item.get('title'),
                    "authors": authors,
                    "published": str(year),
                    "doi": item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else None,
                    "summary": None,  # OpenAlex doesn't provide abstracts in basic response
                    "found": True,
                    "source": "openalex"
                }
    except Exception as e:
        print(f"OpenAlex error: {e}")
    return {"found": False}


def _try_crossref(query: str) -> Dict:
    """Try CrossRef API as last resort."""
    import requests
    
    try:
        if query.strip().upper().startswith("DOI:"):
            doi = query.strip()[4:].strip()
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=10)
        else:
            url = "https://api.crossref.org/works"
            params = {"query": query, "rows": 1}
            response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'message' in data:
                if 'items' in data['message']:
                    items = data['message']['items']
                    if not items:
                        return {"found": False}
                    item = items[0]
                else:
                    item = data['message']
                
                # Extract authors
                authors = []
                for author in item.get('author', []):
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                if not authors:
                    authors = ["Unknown"]
                
                # Extract year
                year = "Unknown"
                if item.get('published-print', {}).get('date-parts'):
                    year = str(item['published-print']['date-parts'][0][0])
                elif item.get('published-online', {}).get('date-parts'):
                    year = str(item['published-online']['date-parts'][0][0])
                elif item.get('created', {}).get('date-parts'):
                    year = str(item['created']['date-parts'][0][0])
                
                # Get title
                title = item.get('title', ['Unknown'])[0] if item.get('title') else "Unknown"
                
                return {
                    "title": title,
                    "authors": authors,
                    "published": year,
                    "doi": item.get('DOI'),
                    "summary": None,
                    "found": True,
                    "source": "crossref"
                }
    except Exception as e:
        print(f"CrossRef error: {e}")
    return {"found": False}
