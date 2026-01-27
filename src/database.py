"""
ChromaDB Database Module

Handles all database operations for the Academic Literature Assistant using local ChromaDB.
"""

import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Constants
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy loaded singletons
_client = None
_collection = None
_embedding_model = None

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
            os.makedirs(DB_DIR)
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
                papers[title] = {
                    'title': title,
                    'authors': meta.get('authors', 'Unknown'),
                    'published': meta.get('published', 'Unknown'),
                    'doi': meta.get('doi', ''),
                    'entry_id': meta.get('entry_id', ''),
                    'chunk_count': 0
                }
            papers[title]['chunk_count'] += 1
            
    return list(papers.values())

def get_paper_count() -> int:
    return len(get_all_papers())

def get_chunk_count() -> int:
    collection = get_collection()
    return collection.count()

def get_all_chunks() -> Dict:
    collection = get_collection()
    return collection.get(include=["documents", "metadatas"])

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

# ============== Vector Search Operations ==============

def query_similar(query_text: str, n_results: int = 5) -> Dict:
    collection = get_collection()
    model = get_embedding_model()
    
    query_embedding = model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

def get_all_embeddings() -> Tuple[List[str], List[np.ndarray], List[Dict]]:
    """Get all embeddings for clustering."""
    collection = get_collection()
    result = collection.get(include=["embeddings", "metadatas"])
    
    titles = [m.get('title', 'Unknown') for m in result['metadatas']]
    embeddings = [np.array(e) for e in result['embeddings']]
    return titles, embeddings, result['metadatas']

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
