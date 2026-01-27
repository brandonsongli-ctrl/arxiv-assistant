"""
Migration Script: ChromaDB to PostgreSQL

Migrates existing paper data from ChromaDB to PostgreSQL + pgvector.
Run this once after setting up PostgreSQL.

Usage:
    python src/migrate_to_postgres.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EMBEDDING_DIMENSION


def migrate():
    """Migrate data from ChromaDB to PostgreSQL."""
    
    print("=" * 60)
    print("ChromaDB to PostgreSQL Migration")
    print("=" * 60)
    
    # Step 1: Test PostgreSQL connection
    print("\n[1/5] Testing PostgreSQL connection...")
    from src.database import test_connection, init_db
    
    if not test_connection():
        print("ERROR: Cannot connect to PostgreSQL!")
        print("Make sure PostgreSQL is running and check your config.py settings.")
        print("Default settings: host=localhost, port=5432, db=academic_assistant, user=postgres, password=postgres")
        return False
    print("✓ PostgreSQL connection successful")
    
    # Step 2: Initialize PostgreSQL schema
    print("\n[2/5] Initializing PostgreSQL schema...")
    init_db()
    print("✓ Schema initialized")
    
    # Step 3: Load ChromaDB data
    print("\n[3/5] Loading data from ChromaDB...")
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
        
        if not os.path.exists(DB_DIR):
            print(f"ChromaDB directory not found: {DB_DIR}")
            print("Nothing to migrate - start fresh with PostgreSQL!")
            return True
        
        client = chromadb.PersistentClient(path=DB_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name="papers", embedding_function=ef)
        
        # Get all data from ChromaDB
        result = collection.get(include=["documents", "embeddings", "metadatas"])
        
        total_chunks = len(result.get('ids', []))
        print(f"✓ Found {total_chunks} chunks in ChromaDB")
        
        if total_chunks == 0:
            print("No data to migrate!")
            return True
            
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        print("If ChromaDB is not installed, that's OK - start fresh with PostgreSQL!")
        return True
    
    # Step 4: Group chunks by paper
    print("\n[4/5] Grouping chunks by paper...")
    
    papers = {}  # title -> {metadata, chunks, embeddings}
    
    for i in range(total_chunks):
        embeddings = result.get('embeddings')
        
        # Safe check for embeddings existence (handle list or numpy array)
        has_embeddings = embeddings is not None and len(embeddings) > 0
        
        if params_check := False: pass # Dummy line to maintain indentation if needed by tool
        
        meta = result['metadatas'][i] if result.get('metadatas') else {}
        doc = result['documents'][i] if result.get('documents') else ""
        emb = embeddings[i] if has_embeddings else None
        
        title = meta.get('title', 'Unknown')
        
        if title not in papers:
            papers[title] = {
                'metadata': {
                    'title': title,
                    'authors': meta.get('authors', 'Unknown'),
                    'published': meta.get('published', 'Unknown'),
                    'entry_id': meta.get('entry_id', ''),
                    'doi': meta.get('doi', ''),
                    'summary': meta.get('summary', ''),
                    'source': meta.get('source', 'chromadb_migration')
                },
                'chunks': [],
                'embeddings': []
            }
        
        chunk_index = meta.get('chunk_index', len(papers[title]['chunks']))
        papers[title]['chunks'].append((chunk_index, doc))
        if emb is not None:
            papers[title]['embeddings'].append((chunk_index, emb))
    
    print(f"✓ Grouped into {len(papers)} unique papers")
    
    # Step 5: Insert into PostgreSQL
    print("\n[5/5] Inserting into PostgreSQL...")
    
    import psycopg2
    from psycopg2.extras import execute_values
    from pgvector.psycopg2 import register_vector
    from src.config import get_connection_params
    
    conn = psycopg2.connect(**get_connection_params())
    register_vector(conn)
    cur = conn.cursor()
    
    migrated_papers = 0
    migrated_chunks = 0
    
    try:
        for title, paper_data in papers.items():
            meta = paper_data['metadata']
            
            # Insert paper
            cur.execute("""
                INSERT INTO papers (title, authors, published, entry_id, doi, summary, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (title) DO UPDATE SET
                    authors = EXCLUDED.authors,
                    published = EXCLUDED.published
                RETURNING id
            """, (
                meta['title'],
                meta['authors'],
                meta['published'],
                meta['entry_id'],
                meta['doi'],
                meta['summary'],
                meta['source']
            ))
            paper_id = cur.fetchone()[0]
            migrated_papers += 1
            
            # Sort chunks by index
            sorted_chunks = sorted(paper_data['chunks'], key=lambda x: x[0])
            sorted_embeddings = sorted(paper_data['embeddings'], key=lambda x: x[0])
            
            # Insert chunks with embeddings
            if sorted_embeddings:
                # We have embeddings from ChromaDB
                for (chunk_idx, content), (_, embedding) in zip(sorted_chunks, sorted_embeddings):
                    cur.execute("""
                        INSERT INTO chunks (paper_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        ON CONFLICT (paper_id, chunk_index) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                    """, (paper_id, chunk_idx, content, embedding.tolist()))
                    migrated_chunks += 1
            else:
                # No embeddings - insert content only (will need to regenerate embeddings)
                for chunk_idx, content in sorted_chunks:
                    cur.execute("""
                        INSERT INTO chunks (paper_id, chunk_index, content)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (paper_id, chunk_index) DO UPDATE SET
                            content = EXCLUDED.content
                    """, (paper_id, chunk_idx, content))
                    migrated_chunks += 1
            
            if migrated_papers % 10 == 0:
                print(f"  Migrated {migrated_papers}/{len(papers)} papers...")
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        print(f"Error during migration: {e}")
        return False
    finally:
        cur.close()
        conn.close()
    
    print(f"✓ Migrated {migrated_papers} papers and {migrated_chunks} chunks")
    
    # Summary
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"Papers migrated: {migrated_papers}")
    print(f"Chunks migrated: {migrated_chunks}")
    print("\nYou can now use the app with PostgreSQL!")
    print("The original ChromaDB data is still in data/chroma_db/ as backup.")
    
    return True


if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
