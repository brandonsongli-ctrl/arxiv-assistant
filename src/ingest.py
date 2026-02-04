import os
import hashlib
import re
from typing import List, Dict
from pypdf import PdfReader
from src import database, config
from src.metadata_utils import compute_canonical_id, extract_arxiv_id, extract_openalex_id, normalize_doi

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""
        
    text = ""
    try:
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                # Postgres cannot handle NUL characters
                text += extracted.replace('\x00', '') + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text

def _split_long_text(text: str, max_len: int) -> List[str]:
    """Fallback splitter for very long sentences."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks using paragraph and sentence boundaries.
    """
    if not text:
        return []
        
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
        
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, int(overlap))
    
    # Normalize line endings and de-hyphenate common PDF line wraps
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r'-\n', '', normalized)
    
    # Split into paragraphs by blank lines
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', normalized) if p.strip()]
    
    units: List[str] = []
    for para in raw_paragraphs:
        para = re.sub(r'\s*\n\s*', ' ', para)  # join single line breaks
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
            
        if len(para) <= chunk_size:
            units.append(para)
            continue
            
        # Split long paragraphs into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                units.extend(_split_long_text(sent, chunk_size))
    
    # Build chunks from units
    chunks: List[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        if len(current) + 1 + len(unit) <= chunk_size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
        
    # Apply overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            prefix = prev[-min(overlap, len(prev)):]
            merged = f"{prefix} {chunks[i]}".strip()
            overlapped.append(merged)
        chunks = overlapped
        
    return chunks

def ingest_paper(pdf_path: str, metadata: Dict):
    """
    Process a PDF and add it to the PostgreSQL vector DB.
    """
    print(f"Ingesting {pdf_path}...")
    
    # 1. Extract Text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return

    # 2. Add/Get Paper Record
    # Ensure metadata has 'source'
    if 'source' not in metadata or metadata['source'] == 'manual_upload':
        metadata['source'] = pdf_path
    
    # Normalize identifiers
    if metadata.get("doi"):
        metadata["doi"] = normalize_doi(metadata.get("doi"))
    if not metadata.get("arxiv_id") and metadata.get("entry_id"):
        metadata["arxiv_id"] = extract_arxiv_id(str(metadata.get("entry_id")))
    if not metadata.get("openalex_id") and metadata.get("entry_id"):
        metadata["openalex_id"] = extract_openalex_id(str(metadata.get("entry_id")))
    metadata["canonical_id"] = compute_canonical_id(metadata)

    # Skip duplicates for strong IDs
    canonical_id = metadata.get("canonical_id", "")
    if canonical_id.startswith(("doi:", "arxiv:", "openalex:")):
        if database.has_paper_by_metadata("canonical_id", canonical_id):
            print(f"Skipping duplicate paper (canonical_id={canonical_id})")
            return False
        
    paper_id = database.insert_paper(metadata)
    # print(f"Paper '{metadata.get('title')}' ID: {paper_id}")
    
    # 3. Create Chunks
    chunks = chunk_text(text)
    if not chunks:
        print(f"Warning: No text chunks created for {pdf_path}")
        return False
        
    # print(f"Generated {len(chunks)} chunks.")
    
    # 4. Insert Chunks (and generate embeddings automatically in database module)
    # For ChromaDB, we must pass metadata to attach to chunks
    database.insert_chunks(paper_id, chunks, metadata)
    # print(f"Successfully ingested {len(chunks)} chunks to ChromaDB.")
    return True


def ingest_directory(directory: str, resolve_metadata: bool = True, progress_callback=None) -> Dict:
    """
    Incrementally ingest new PDFs from a directory.
    Skips files already present in the database (by entry_id or source path).
    """
    if not os.path.exists(directory):
        return {
            "total": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "message": f"Directory not found: {directory}"
        }
    
    from src import scraper, dedupe
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    files.sort()
    
    existing = database.get_all_papers()
    existing_entry_ids = {str(p.get("entry_id", "")) for p in existing}
    existing_sources = {str(p.get("source", "")) for p in existing}
    existing_titles = {dedupe.normalize_title(p.get("title", "")) for p in existing}
    
    total = len(files)
    ingested = 0
    skipped = 0
    failed = 0
    
    for idx, fname in enumerate(files):
        file_path = os.path.join(directory, fname)
        entry_id = f"manual_{fname}"
        
        if entry_id in existing_entry_ids or file_path in existing_sources:
            skipped += 1
            if progress_callback:
                progress_callback(idx + 1, total, f"Skipping existing: {fname}")
            continue
        
        raw_title = fname.replace(".pdf", "").replace("_", " ").replace("-", " ")
        title_norm = dedupe.normalize_title(raw_title)
        if title_norm in existing_titles:
            skipped += 1
            if progress_callback:
                progress_callback(idx + 1, total, f"Skipping duplicate title: {fname}")
            continue
        
        final_title = raw_title
        final_authors = ["Unknown"]
        final_doi = "Unknown"
        final_summary = "Manual Upload"
        final_published = "Unknown"
        
        if resolve_metadata:
            resolved = scraper.resolve_paper_metadata(raw_title)
            if resolved.get("found"):
                final_title = resolved.get("title") or final_title
                final_authors = resolved.get("authors") or final_authors
                final_doi = resolved.get("doi") or final_doi
                final_summary = resolved.get("summary") or final_summary
                final_published = resolved.get("published") or final_published
        
        metadata = {
            "title": final_title,
            "authors": final_authors,
            "summary": final_summary,
            "published": final_published,
            "doi": final_doi,
            "entry_id": entry_id,
            "source": file_path
        }
        
        # Skip duplicates based on strong IDs when available
        canonical_id = compute_canonical_id(metadata)
        if canonical_id.startswith(("doi:", "arxiv:", "openalex:")):
            if database.has_paper_by_metadata("canonical_id", canonical_id):
                skipped += 1
                if progress_callback:
                    progress_callback(idx + 1, total, f"Skipping duplicate ID: {fname}")
                continue
        
        try:
            success = ingest_paper(file_path, metadata)
            if success:
                ingested += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        
        if progress_callback:
            progress_callback(idx + 1, total, f"Ingested: {fname}")
    
    return {
        "total": total,
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        "message": f"Ingested {ingested}, skipped {skipped}, failed {failed}."
    }

def get_all_papers() -> List[Dict]:
    """
    Retrieve listing of all unique papers in the DB.
    Wrapper for database.get_all_papers() to maintain API compatibility.
    """
    return database.get_all_papers()

