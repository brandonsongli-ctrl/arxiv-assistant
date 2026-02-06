import arxiv
import os
import re
from datetime import date, datetime
from typing import List, Dict, Optional

def _safe_filename(title: str, fallback: str = "paper", ext: str = ".pdf") -> str:
    """
    Create a filesystem-safe filename for a paper.
    """
    stem = (title or "").strip()
    if stem.lower().endswith(ext):
        stem = stem[: -len(ext)]
    stem = stem.replace("/", " ").replace("\\", " ")
    stem = re.sub(r'[\t\r\n]+', ' ', stem)
    stem = re.sub(r'\s+', ' ', stem).strip()
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in " -_().,[]")
    stem = stem.strip(" .")
    if not stem:
        stem = fallback
    if len(stem) > 150:
        stem = stem[:150].rstrip()
    return f"{stem}{ext}"


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _abstract_from_inverted_index(inverted_index: Dict) -> Optional[str]:
    if not inverted_index:
        return None
    try:
        max_pos = -1
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))
        if max_pos < 0:
            return None
        words = [""] * (max_pos + 1)
        for token, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos <= max_pos:
                    words[pos] = token
        abstract = " ".join(w for w in words if w)
        abstract = re.sub(r"\s+", " ", abstract).strip()
        return abstract if abstract else None
    except Exception:
        return None


def _format_arxiv_date(d: date, end_of_day: bool = False) -> str:
    if isinstance(d, datetime):
        dt = d
    else:
        if end_of_day:
            dt = datetime(d.year, d.month, d.day, 23, 59)
        else:
            dt = datetime(d.year, d.month, d.day, 0, 0)
    return dt.strftime("%Y%m%d%H%M")


def _build_arxiv_query(query: str, category: Optional[str], date_from: Optional[date], date_to: Optional[date], date_field: str) -> str:
    terms = []
    base_query = (query or "").strip()
    if base_query:
        terms.append(f"({base_query})")
    if category:
        categories = [c.strip() for c in re.split(r"[,\s]+", category) if c.strip()]
        if categories:
            if len(categories) == 1:
                terms.append(f"cat:{categories[0]}")
            else:
                cat_terms = " OR ".join([f"cat:{c}" for c in categories])
                terms.append(f"({cat_terms})")
    if date_from or date_to:
        if not date_from:
            date_from = date(1900, 1, 1)
        if not date_to:
            date_to = date.today()
        field = "submittedDate" if date_field == "submitted" else "lastUpdatedDate"
        start = _format_arxiv_date(date_from, end_of_day=False)
        end = _format_arxiv_date(date_to, end_of_day=True)
        terms.append(f"{field}:[{start} TO {end}]")
    return " AND ".join(terms).strip()


def dedupe_results(results: List[Dict]) -> List[Dict]:
    """
    De-duplicate search results by DOI or normalized title.
    Prefer entries with PDF URLs and longer summaries.
    """
    if not results:
        return []
        
    by_key: Dict[str, Dict] = {}
    for paper in results:
        doi = (paper.get("doi") or "").lower().strip()
        if doi:
            key = f"doi:{doi}"
        else:
            key = f"title:{_normalize_title(paper.get('title', ''))}"
            
        if key not in by_key:
            by_key[key] = paper
            continue
            
        existing = by_key[key]
        score_existing = 0
        score_new = 0
        
        if existing.get("pdf_url"):
            score_existing += 2
        if paper.get("pdf_url"):
            score_new += 2
        if (existing.get("summary") or ""):
            score_existing += min(len(existing.get("summary", "")) / 200, 2)
        if (paper.get("summary") or ""):
            score_new += min(len(paper.get("summary", "")) / 200, 2)
        if existing.get("published") and existing.get("published") != "Unknown":
            score_existing += 1
        if paper.get("published") and paper.get("published") != "Unknown":
            score_new += 1
            
        if score_new > score_existing:
            by_key[key] = paper
    
    return list(by_key.values())

def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance",
                 category: Optional[str] = None,
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 date_field: str = "submitted") -> List[Dict]:
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
    
    # 1. Try ID Search if it looks like an ID (ignore filters)
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
        final_query = _build_arxiv_query(query, category, date_from, date_to, date_field)
        search = arxiv.Search(
            query=final_query or query,
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
        venue = getattr(result, "journal_ref", None)
        output.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
            "published": str(result.published),
            "obj": result,
            "source": "arxiv",
            "doi": result.doi if hasattr(result, "doi") else None,
            "venue": venue
        })
    return output


def download_paper(paper_obj, download_dir: str):
    """
    Download the PDF of a paper.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    
    # Paper title as filename (sanitized)
    fallback_name = getattr(paper_obj, "entry_id", "arxiv_paper")
    filename = _safe_filename(getattr(paper_obj, "title", None), fallback=str(fallback_name))
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
        "fields": "title,authors,abstract,url,openAccessPdf,publicationDate,year,externalIds,venue,publicationVenue"
    }
    
    try:
        max_retries = 3
        base_wait = 2

        response = None
        for attempt in range(max_retries + 1):
            try:
                # 10 second timeout
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    wait_time = base_wait * (2 ** attempt)
                    print(f"Semantic Scholar Rate Limit hit. Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries+1})")
                    time.sleep(wait_time)
                    continue # Retry
                    
                response.raise_for_status()
                break # Success, exit loop
                
            except requests.exceptions.RequestException as e:
                # If it's the last attempt, raise it to be caught by the outer try/except can handle logging
                if attempt == max_retries:
                    raise e
                time.sleep(1)
                continue
                
        # If we fall through here without breaking, we might have final 429 response or success
        if response and response.status_code == 429:
             # Still rate limited after retries
             return [{
                "title": "⚠️ Error: Semantic Scholar Rate Limit Hit (Persistent)",
                "authors": ["System"],
                "summary": "Too many requests. Please wait a minute and try again.",
                "pdf_url": None,
                "entry_id": "error_rate_limit",
                "published": "Now",
                "source": "semanticscholar",
                "obj": None
            }]
    
        if not response:
            return []

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
            venue = None
            if item.get('publicationVenue'):
                venue = item.get('publicationVenue', {}).get('name') or item.get('publicationVenue', {}).get('display_name')
            if not venue:
                venue = item.get('venue')
            
            output.append({
                "title": item.get('title', 'No Title'),
                "authors": authors_list,
                "summary": item.get('abstract') if item.get('abstract') else "No abstract available.",
                "pdf_url": pdf_url,
                "entry_id": item.get('paperId'),
                "published": str(item.get('publicationDate')) if item.get('publicationDate') else str(item.get('year')),
                "source": "semanticscholar",
                "obj": None,
                "doi": item.get('externalIds', {}).get('DOI') if item.get('externalIds') else None,
                "venue": venue
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
    import time
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
        
    filename = _safe_filename(title, fallback="paper")
    path = os.path.join(download_dir, filename)
    
    if os.path.exists(path):
        print(f"Already exists: {filename}")
        return path
        
    headers = {
        "User-Agent": "ArxivAssistant/1.0 (+https://example.com)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    
    max_retries = 3
    backoff = 2
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            if response.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded: {filename}")
                return path
            
            if response.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_retries:
                    time.sleep(backoff * (2 ** attempt))
                    continue
            
            raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            print(f"Failed to download {url}: {e}")
            return None
    
    return None

def search_nber(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search NBER for working papers.
    Uses OpenAlex with specific keyword filters.
    """
    return _search_openalex_simple(f"NBER {query}", max_results, "NBER")

def search_ssrn(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search SSRN for papers. 
    Uses OpenAlex with SSRN as a keyword.
    """
    return _search_openalex_simple(f"SSRN {query}", max_results, "SSRN")

def _search_openalex_simple(query: str, max_results: int, source_name: str, source_id: str = None) -> List[Dict]:
    import requests
    url = "https://api.openalex.org/works"
    
    params = {
        "search": query,
        "per_page": max_results,
        "select": "id,title,authorships,publication_year,publication_date,doi,abstract_inverted_index,best_oa_location,primary_location,host_venue"
    }
    
    if source_id:
        params["filter"] = f"primary_location.source.id:{source_id}"

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            output = []
            for item in data.get('results', []):
                authors = [a.get('author', {}).get('display_name') for a in item.get('authorships', [])]
                # Filter out None authors
                authors = [a for a in authors if a]
                
                pdf_url = item.get('best_oa_location', {}).get('pdf_url') if item.get('best_oa_location') else None
                abstract = _abstract_from_inverted_index(item.get('abstract_inverted_index') or {})
                
                # For NBER, we want to ensure it's actually NBER if the filter is loose
                if source_name == "NBER":
                    primary_loc = item.get('primary_location') or {}
                    source_obj = primary_loc.get('source') or {}
                    display_name = source_obj.get('display_name') or ''
                    if "National Bureau of Economic Research" not in display_name and "NBER" not in display_name:
                        # For now, we trust OpenAlex's relevance if we searched "NBER {query}"
                        pass

                summary = abstract if abstract else f"Source: {source_name}. DOI: {item.get('doi') or 'N/A'}"

                venue = None
                host_venue = item.get("host_venue") or {}
                if host_venue.get("display_name"):
                    venue = host_venue.get("display_name")
                if not venue:
                    primary_loc = item.get("primary_location") or {}
                    source_obj = primary_loc.get("source") or {}
                    if source_obj.get("display_name"):
                        venue = source_obj.get("display_name")

                output.append({
                    "title": item.get('title', 'No Title'),
                    "authors": authors if authors else ["Unknown"],
                    "summary": summary,
                    "pdf_url": pdf_url,
                    "entry_id": item.get('id'),
                    "published": str(item.get('publication_year')),
                    "source": source_name.lower(),
                    "obj": None,
                    "doi": item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else None,
                    "venue": venue
                })
            return output
    except Exception as e:
        print(f"Error searching {source_name}: {e}")
    return []


def search_google_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Google Scholar using 'scholarly' library.
    WARNING: Highly rate-limited. Use with caution.
    """
    try:
        from scholarly import scholarly
        try:
            from scholarly import ProxyGenerator
        except Exception:
            ProxyGenerator = None
        
        # Proxy configuration (optional)
        proxy_url = os.getenv("SCHOLAR_PROXY", "").strip()
        use_free_proxy = os.getenv("SCHOLAR_USE_FREE_PROXY", "0") == "1"
        use_tor = os.getenv("SCHOLAR_USE_TOR", "0") == "1"
        
        if ProxyGenerator is not None:
            pg = ProxyGenerator()
            configured = False
            
            if proxy_url:
                try:
                    if pg.SingleProxy(proxy_url):
                        scholarly.use_proxy(pg)
                        configured = True
                except Exception:
                    configured = False
            
            if not configured and use_free_proxy:
                try:
                    if hasattr(pg, "FreeProxies") and pg.FreeProxies():
                        scholarly.use_proxy(pg)
                        configured = True
                except Exception:
                    configured = False
            
            if not configured and use_tor:
                try:
                    if hasattr(pg, "Tor_Internal") and pg.Tor_Internal():
                        scholarly.use_proxy(pg)
                        configured = True
                    elif hasattr(pg, "Tor_External") and pg.Tor_External("127.0.0.1", 9050):
                        scholarly.use_proxy(pg)
                        configured = True
                except Exception:
                    configured = False
        
        search_query = scholarly.search_pubs(query)
        output = []
        count = 0
        
        for item in search_query:
            if count >= max_results:
                break
                
            # Parse scholar result
            bib = item.get('bib', {})
            pub_url = item.get('pub_url')
            eprint_url = item.get('eprint_url') # Often the PDF link
            
            output.append({
                "title": bib.get('title', 'No Title'),
                "authors": bib.get('author', ["Unknown"]),
                "summary": bib.get('abstract', 'No abstract available.'),
                "pdf_url": eprint_url if eprint_url else pub_url,
                "entry_id": item.get('author_id', ['unknown'])[0] if item.get('author_id') else f"gs_{count}",
                "published": str(bib.get('pub_year', 'Unknown')),
                "source": "google_scholar",
                "obj": None,
                "venue": bib.get("venue")
            })
            count += 1
            
        return output
    except Exception as e:
        print(f"Error searching Google Scholar: {e}")
        # Fallback to OpenAlex when Scholar fails
        fallback = _search_openalex_simple(query, max_results, "OpenAlex (Scholar Fallback)")
        if fallback:
            for item in fallback:
                item["source"] = "google_scholar_fallback"
                item["summary"] = f"[OpenAlex fallback] {item.get('summary', '')}".strip()
            return fallback
        
        return [{
            "title": f"⚠️ Google Scholar Error: {str(e)}",
            "authors": ["System"],
            "summary": "Google Scholar scraping failed (likely rate limit). Fallback also failed.",
            "pdf_url": None,
            "entry_id": "error_gs",
            "published": "Now",
            "source": "google_scholar",
            "obj": None
        }]

def resolve_paper_metadata(query: str) -> Dict:
    """
    Try to find DOI and canonical metadata using multiple APIs.
    Falls back: Semantic Scholar -> OpenAlex -> CrossRef
    Supports queries starting with 'DOI:' for direct lookup.
    """
    import requests
    import time
    
    # Helper to check similarity against Title + Authors to support queries like "Title AuthorName"
    def check_match(query, result):
        if not result.get("found"): return False
        
        # 1. DOI Exact Match Bypass
        # If we searched for a DOI, and got a result with that DOI, we trust it.
        # We don't need title similarity in this case.
        if query.strip().upper().startswith("DOI:"):
            query_doi = query.strip()[4:].strip().lower()
            result_doi = (result.get("doi") or "").lower()
            # Loose comparison: check if one contains the other to handle variations like "10.1000/1" vs "https://doi.org/10.1000/1"
            if query_doi in result_doi or result_doi in query_doi:
                return True
            # If DOIs don't match, fall back to title similarity (unlikely to help, but safe)

        # 2. Text Similarity Match
        # Compare against Title + Authors
        candidate_text = (result.get("title") or "") + " " + " ".join(result.get("authors") or [])
        return _is_title_similar(query, candidate_text)

    # Try Semantic Scholar first
    result = _try_semantic_scholar(query)
    if check_match(query, result):
        return result
    
    # Fallback to OpenAlex (no rate limit, very reliable)
    result = _try_openalex(query)
    if check_match(query, result):
        return result
        
    # Fallback to CrossRef
    result = _try_crossref(query)
    if check_match(query, result):
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
    
    # Subset match: If one title is mostly contained in the other
    # Useful when query includes author name: "Strategic Thinking Crawford" vs "Strategic Thinking"
    min_len = min(len(query_words), len(title_words))
    subset_ratio = len(intersection) / min_len if min_len > 0 else 0
    
    return (similarity >= threshold) or (subset_ratio >= 0.8)
    
    # Debug output (can be removed in production)
    # print(f"Similarity '{query[:30]}...' vs '{title[:30]}...': {similarity:.2f}")
    
    return similarity >= threshold


def _try_semantic_scholar(query: str) -> Dict:
    """Try Semantic Scholar API with rate-limit handling."""
    import requests
    import time
    
    max_retries = 3
    base_wait = 2
    
    for attempt in range(max_retries + 1):
        try:
            if query.strip().upper().startswith("DOI:"):
                doi = query.strip()[4:].strip()
                url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
                params = {"fields": "title,authors,externalIds,year,publicationDate,abstract,venue,publicationVenue"}
            else:
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {"query": query, "limit": 1, "fields": "title,authors,externalIds,year,publicationDate,abstract,venue,publicationVenue"}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 429:
                wait_time = base_wait * (2 ** attempt)
                print(f"Semantic Scholar rate limited, sleeping {wait_time}s... (attempt {attempt+1}/{max_retries+1})")
                time.sleep(wait_time)
                continue  # Retry
                
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
                
                venue = None
                if item.get("publicationVenue"):
                    venue = item.get("publicationVenue", {}).get("name") or item.get("publicationVenue", {}).get("display_name")
                if not venue:
                    venue = item.get("venue")

                return {
                    "title": item.get('title'),
                    "authors": authors_list,
                    "published": str(item.get('publicationDate') or item.get('year')),
                    "doi": doi,
                    "summary": item.get('abstract'),
                    "venue": venue,
                    "found": True,
                    "source": "semantic_scholar"
                }
            
            # Other error codes - don't retry
            return {"found": False}
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
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
        
        params["select"] = "title,authorships,publication_year,publication_date,doi,abstract_inverted_index,primary_location,host_venue"
        
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
                
                abstract = _abstract_from_inverted_index(item.get('abstract_inverted_index') or {})
                
                venue = None
                host_venue = item.get("host_venue") or {}
                if host_venue.get("display_name"):
                    venue = host_venue.get("display_name")
                
                return {
                    "title": item.get('title'),
                    "authors": authors,
                    "published": str(year),
                    "doi": item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else None,
                    "summary": abstract,
                    "venue": venue,
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
                
                abstract = item.get("abstract")
                if abstract:
                    abstract = re.sub(r"<[^>]+>", " ", abstract)
                    abstract = re.sub(r"\s+", " ", abstract).strip()
                
                container_titles = item.get("container-title", [])
                venue = container_titles[0] if container_titles else None
                
                return {
                    "title": title,
                    "authors": authors,
                    "published": year,
                    "doi": item.get('DOI'),
                    "summary": abstract,
                    "venue": venue,
                    "found": True,
                    "source": "crossref"
                }
    except Exception as e:
        print(f"CrossRef error: {e}")
    return {"found": False}
