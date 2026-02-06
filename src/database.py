"""
ChromaDB Database Module

Handles all database operations for the Academic Literature Assistant using local ChromaDB.
"""

import chromadb
from chromadb.config import Settings
import os
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer

_embed_cache: Dict[str, List[float]] = {}

# Constants
DB_DIR = os.getenv("ARXIV_ASSISTANT_DB_DIR", os.path.expanduser("~/.arxiv_assistant/chroma_db"))
EMBEDDING_MODEL = os.getenv("ARXIV_ASSISTANT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Lazy loaded singletons
_client = None
_collection = None
_embedding_model = None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = [str(v).strip() for v in value if str(v).strip()]
    else:
        raw = [t.strip() for t in re.split(r"[;,]", str(value)) if t.strip()]
    seen = set()
    tags = []
    for tag in raw:
        low = tag.lower()
        if low in seen:
            continue
        seen.add(low)
        tags.append(tag)
    return tags

def get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_client():
    """Get ChromaDB client."""
    global _client
    if _client is None:
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client

def get_collection():
    """Get the main paper collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def init_db():
    """Initialize database (no-op for ChromaDB as it auto-inits)."""
    print("Initializing ChromaDB at", DB_DIR)
    get_collection()
    print("Database initialized successfully!")

def drop_tables():
    """Reset the database."""
    global _client, _collection
    if _client:
        try:
            _client.delete_collection("papers")
            _collection = None
            print("Collection deleted.")
        except Exception as e:
            print(f"Error dropping collection: {e}")

# ============== Paper Operations ==============

def insert_paper(metadata: Dict) -> Optional[str]:
    """
    Insert a paper concept. In Chroma, we don't have a separate papers table,
    so we just return the entry_id or title as a pseudo-ID.
    """
    # Verify paper doesn't strictly exist? 
    # With Chroma we just return the title/id to be used for chunk metadata.
    return metadata.get('title', 'Unknown')

def insert_chunks(paper_identifier: str, chunks: List[str], metadata: Dict = None):
    """
    Insert text chunks.
    args:
        paper_identifier: unused (legacy), metadata usually contains the ID links
        chunks: list of text strings
        metadata: dict containing paper info (title, authors, etc) to be attached to EACH chunk
    NOTE: The refactored database.py interface from Postgres version had `insert_chunks(paper_id, chunks)`,
    but `insert_paper` returned an int ID.
    In Chroma, we need the full metadata passed here to attach to chunks.
    However, the caller (`ingest.py`) likely calls `insert_paper` then `insert_chunks`.
    
    We need to handle this impedance mismatch.
    If `insert_paper` is called, we can cache the metadata?
    Or better, `ingest.py` should be updated to pass metadata to `insert_chunks`.
    
    checking `ingest.py`:
        paper_id = database.insert_paper(metadata)
        database.insert_chunks(paper_id, chunks)
        
    Since we are keeping the API, `paper_id` will be what we returned from `insert_paper` (i.e. title).
    BUT we lost the other metadata (authors, published, etc) if we don't pass it to insert_chunks.
    
    CRITICAL FIX: For the API to work with Chroma, `insert_paper` logic is tricky because Chroma is flat.
    
    Workaround: `insert_paper` returns `title`.
    `insert_chunks` takes `title`. But where does it get authors/etc?
    
    Imperfect solution: `insert_paper` stores metadata in a temporary global/cache or we query existing?
    Actually, let's look at `ingest.py` history. It used to pass metadata directly to `ingest_paper` which did eveyrthing.
    
    Let's modify `insert_chunks` to accept `chunk_metadatas`.
    Wait, `ingest.py` splits 
    `paper_id = db.insert_paper(metadata)` 
    `db.insert_chunks(paper_id, chunks)`
    
    I'll have to change `ingest.py` as well to make this smooth.
    But purely for `database.py`:
    We can fetch existing chunks for this paper to get metadata? No, it's a new paper.
    
    Let's change the `insert_chunks` signature in `database.py` and update `ingest.py` in next step.
    Or, `insert_paper` can actually write a dummy chunk? No.
    
    Let's assume `insert_paper` does nothing but return the metadata object or ID, 
    and `insert_chunks` will need to receive the metadata.
    
    Let's check `ingest.py` content to see how to pivot.
    """
    # For now, implementing standard Chroma logic assuming arguments will be fixed.
    collection = get_collection()
    model = get_embedding_model()
    
    # Generate embeddings
    embeddings = model.encode(chunks).tolist()
    
    # Generate IDs
    import uuid
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Prepare metadatas
    # We NEED the paper metadata here.
    # If the caller doesn't provide it, we are in trouble.
    # I will modify ingest.py to pass metadata to insert_chunks.
    
    if metadata is None:
        # Fallback/Error
        metadata = {'title': str(paper_identifier)}
        
    metadatas = []
    for i in range(len(chunks)):
        m = metadata.copy()
        m['chunk_index'] = i
        # Flatten lists for Chroma (e.g. authors)
        if isinstance(m.get('authors'), list):
            m['authors'] = ", ".join(m['authors'])
        
        # CRITICAL: Sanitize metadata for ChromaDB
        # ChromaDB only accepts str, int, float, bool - not None or complex types
        sanitized = {}
        for key, value in m.items():
            if value is None:
                sanitized[key] = ""  # Convert None to empty string
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ", ".join(str(v) for v in value)
            else:
                sanitized[key] = str(value)  # Convert other types to string
        metadatas.append(sanitized)
        
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Inserted {len(chunks)} chunks for {paper_identifier}")

def get_all_papers() -> List[Dict]:
    """Get all unique papers."""
    collection = get_collection()
    # This is inefficient in Chroma (fetching all metadata) but standard for small apps
    result = collection.get(include=["metadatas"])
    
    papers = {}
    if result and result['metadatas']:
        for meta in result['metadatas']:
            title = meta.get('title', 'Unknown')
            if title not in papers:
                tags = _parse_tags(meta.get('tags', ''))
                papers[title] = {
                    'title': title,
                    'authors': meta.get('authors', 'Unknown'),
                    'published': meta.get('published', 'Unknown'),
                    'doi': meta.get('doi', ''),
                    'entry_id': meta.get('entry_id', ''),
                    'chunk_count': 0,
                    'source': meta.get('source', 'Unknown'),
                    'canonical_id': meta.get('canonical_id', ''),
                    'arxiv_id': meta.get('arxiv_id', ''),
                    'openalex_id': meta.get('openalex_id', ''),
                    'summary': meta.get('summary', ''),
                    'venue': meta.get('venue', ''),
                    'tags': tags,
                    'is_favorite': _parse_bool(meta.get('is_favorite')),
                    'in_reading_list': _parse_bool(meta.get('in_reading_list')),
                    'parse_pages': _parse_int(meta.get('parse_pages', 0)),
                    'parse_ocr_pages': _parse_int(meta.get('parse_ocr_pages', 0)),
                    'parse_extraction_mode': meta.get('parse_extraction_mode', 'embedded_only'),
                    'parse_quality_score': _parse_float(meta.get('parse_quality_score', 0.0)),
                    'parse_quality_label': meta.get('parse_quality_label', 'unknown'),
                    'parse_table_lines': _parse_int(meta.get('parse_table_lines', 0)),
                    'parse_formula_lines': _parse_int(meta.get('parse_formula_lines', 0)),
                    'parse_figure_caption_lines': _parse_int(meta.get('parse_figure_caption_lines', 0))
                }
            papers[title]['chunk_count'] += 1
            
            # Prefer a non-empty summary if available
            if not papers[title].get('summary') and meta.get('summary'):
                papers[title]['summary'] = meta.get('summary')
            if not papers[title].get('venue') and meta.get('venue'):
                papers[title]['venue'] = meta.get('venue')
            if not papers[title].get('is_favorite') and _parse_bool(meta.get('is_favorite')):
                papers[title]['is_favorite'] = True
            if not papers[title].get('in_reading_list') and _parse_bool(meta.get('in_reading_list')):
                papers[title]['in_reading_list'] = True
            if meta.get('tags'):
                merged_tags = papers[title].get('tags', []) + _parse_tags(meta.get('tags'))
                papers[title]['tags'] = _parse_tags(merged_tags)
            papers[title]['parse_pages'] = max(
                papers[title].get('parse_pages', 0),
                _parse_int(meta.get('parse_pages', 0))
            )
            papers[title]['parse_ocr_pages'] = max(
                papers[title].get('parse_ocr_pages', 0),
                _parse_int(meta.get('parse_ocr_pages', 0))
            )
            papers[title]['parse_table_lines'] = max(
                papers[title].get('parse_table_lines', 0),
                _parse_int(meta.get('parse_table_lines', 0))
            )
            papers[title]['parse_formula_lines'] = max(
                papers[title].get('parse_formula_lines', 0),
                _parse_int(meta.get('parse_formula_lines', 0))
            )
            papers[title]['parse_figure_caption_lines'] = max(
                papers[title].get('parse_figure_caption_lines', 0),
                _parse_int(meta.get('parse_figure_caption_lines', 0))
            )
            score = _parse_float(meta.get('parse_quality_score', 0.0))
            if score > papers[title].get('parse_quality_score', 0.0):
                papers[title]['parse_quality_score'] = score
                papers[title]['parse_quality_label'] = meta.get('parse_quality_label', papers[title].get('parse_quality_label', 'unknown'))
                papers[title]['parse_extraction_mode'] = meta.get('parse_extraction_mode', papers[title].get('parse_extraction_mode', 'embedded_only'))
            
    return list(papers.values())

def get_paper_count() -> int:
    return len(get_all_papers())

def get_chunk_count() -> int:
    collection = get_collection()
    return collection.count()

def get_all_chunks() -> Dict:
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    return {
        "ids": result.get("ids", []),
        "documents": result.get("documents", []),
        "metadatas": result.get("metadatas", []),
    }


def has_paper_by_metadata(field: str, value: str) -> bool:
    """Check if any chunks exist with a given metadata field value."""
    if not field or not value:
        return False
    collection = get_collection()
    try:
        result = collection.get(where={field: value}, include=["ids"])
        return bool(result and result.get("ids"))
    except Exception:
        return False


def get_chunks_by_title(title: str) -> Dict:
    """Get all chunks for a given paper title."""
    collection = get_collection()
    # Note: ChromaDB `include` does not accept "ids" (ids are returned by default)
    result = collection.get(where={"title": title}, include=["documents", "metadatas"])
    return {
        "ids": result.get("ids", []),
        "documents": result.get("documents", []),
        "metadatas": result.get("metadatas", []),
    }

def update_paper_metadata_by_title(title: str, new_metadata: Dict) -> int:
    """Update metadata for all chunks of a paper."""
    collection = get_collection()
    
    # Find all chunks for this paper
    result = collection.get(
        where={"title": title},
        include=["metadatas"]
    )
    
    if not result or not result['ids']:
        return 0
        
    ids_to_update = result['ids']
    new_metadatas = []
    
    for old_meta in result['metadatas']:
        updated = old_meta.copy()
        # Update fields
        for k, v in new_metadata.items():
            if k == 'authors' and isinstance(v, list):
                updated[k] = ", ".join(v)
            elif k == 'tags':
                updated[k] = ", ".join(_parse_tags(v))
            elif k in {'is_favorite', 'in_reading_list'}:
                updated[k] = bool(v)
            elif v is None:
                updated[k] = ""
            else:
                updated[k] = v
        new_metadatas.append(updated)
        
    collection.update(
        ids=ids_to_update,
        metadatas=new_metadatas
    )
    
    return len(ids_to_update)

def delete_paper_by_title(title: str) -> bool:
    """Delete a paper and all its chunks by title."""
    collection = get_collection()
    
    # Check if exists first
    result = collection.get(
        where={"title": title},
        include=["metadatas"]
    )
    
    if not result or not result['ids']:
        return False
        
    try:
        collection.delete(
            where={"title": title}
        )
        return True
    except Exception as e:
        print(f"Error deleting paper {title}: {e}")
        return False


def delete_paper_by_metadata(field: str, value: str) -> bool:
    """Delete paper chunks by a specific metadata field value."""
    if not field or value is None:
        return False
    collection = get_collection()
    try:
        result = collection.get(where={field: value}, include=["metadatas"])
        if not result or not result.get("ids"):
            return False
        collection.delete(where={field: value})
        return True
    except Exception as e:
        print(f"Error deleting by metadata {field}={value}: {e}")
        return False


def update_paper_metadata_by_field(field: str, value: str, new_metadata: Dict) -> int:
    """Update metadata for chunks matched by a specific metadata field."""
    if not field or value is None:
        return 0

    collection = get_collection()
    try:
        result = collection.get(where={field: value}, include=["metadatas"])
    except Exception:
        return 0

    if not result or not result.get("ids"):
        return 0

    ids_to_update = result["ids"]
    new_metadatas = []
    for old_meta in result["metadatas"]:
        updated = old_meta.copy()
        for k, v in (new_metadata or {}).items():
            if k == "authors" and isinstance(v, list):
                updated[k] = ", ".join(v)
            elif k == "tags":
                updated[k] = ", ".join(_parse_tags(v))
            elif k in {"is_favorite", "in_reading_list"}:
                updated[k] = bool(v)
            elif v is None:
                updated[k] = ""
            else:
                updated[k] = v
        new_metadatas.append(updated)

    collection.update(ids=ids_to_update, metadatas=new_metadatas)
    return len(ids_to_update)


def reindex_chunks_by_title(title: str) -> int:
    """
    Recompute embeddings for all chunks of a paper title.
    Useful after incremental maintenance operations.
    """
    if not title:
        return 0
    collection = get_collection()
    result = collection.get(where={"title": title}, include=["documents"])
    ids = result.get("ids", []) if result else []
    docs = result.get("documents", []) if result else []
    if not ids or not docs:
        return 0

    model = get_embedding_model()
    embeddings = model.encode(docs).tolist()
    collection.update(ids=ids, embeddings=embeddings)
    return len(ids)

# ============== Vector Search Operations ==============

def query_similar(query_text: str, n_results: int = 5) -> Dict:
    collection = get_collection()
    model = get_embedding_model()
    # Simple in-memory embedding cache for repeated queries
    if query_text in _embed_cache:
        query_embedding = _embed_cache[query_text]
    else:
        query_embedding = model.encode(query_text).tolist()
        _embed_cache[query_text] = query_embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

def get_all_embeddings() -> Tuple[List[str], List[np.ndarray], List[Dict]]:
    """Get all embeddings for clustering."""
    collection = get_collection()
    try:
        result = collection.get(include=["embeddings", "metadatas"])
        titles = [m.get('title', 'Unknown') for m in result['metadatas']]
        embeddings = [np.array(e) for e in result['embeddings']]
        return titles, embeddings, result['metadatas']
    except Exception as e:
        # Fallback: recompute embeddings from documents if stored embeddings are unavailable/corrupt
        try:
            print(f"Warning: failed to fetch stored embeddings, recomputing. Error: {e}")
            result = collection.get(include=["documents", "metadatas"])
            docs = result.get("documents", [])
            titles = [m.get('title', 'Unknown') for m in result.get('metadatas', [])]
            if not docs:
                return [], [], []
            model = get_embedding_model()
            embeddings = [np.array(e) for e in model.encode(docs).tolist()]
            return titles, embeddings, result.get('metadatas', [])
        except Exception as e2:
            print(f"Error recomputing embeddings: {e2}")
            return [], [], []

def get_paper_embeddings_aggregated() -> Tuple[List[str], np.ndarray]:
    """Average embedding per paper."""
    titles, embeddings, _ = get_all_embeddings()
    
    if not titles:
        return [], np.array([])
        
    paper_map = {}
    for title, emb in zip(titles, embeddings):
        if title not in paper_map:
            paper_map[title] = []
        paper_map[title].append(emb)
        
    unique_titles = []
    avg_embeddings = []
    
    for title, embs in paper_map.items():
        unique_titles.append(title)
        avg_embeddings.append(np.mean(embs, axis=0))
        
    return unique_titles, np.array(avg_embeddings)

def test_connection() -> bool:
    try:
        get_client()
        return True
    except Exception:
        return False
