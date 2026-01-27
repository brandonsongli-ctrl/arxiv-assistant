import os
import hashlib
from typing import List, Dict
from pypdf import PdfReader
from src import database

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            # Postgres cannot handle NUL characters
            text += extracted.replace('\x00', '') + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        # Move forward by stride
        start += (chunk_size - overlap)
        
        # Avoid infinite update if overlap >= chunk_size (unlikely but safe)
        if (chunk_size - overlap) <= 0:
            start += chunk_size
            
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
    if 'source' not in metadata:
        metadata['source'] = 'manual_upload'
        
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

def get_all_papers() -> List[Dict]:
    """
    Retrieve listing of all unique papers in the DB.
    Wrapper for database.get_all_papers() to maintain API compatibility.
    """
    return database.get_all_papers()

