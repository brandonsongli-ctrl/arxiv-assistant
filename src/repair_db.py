# Imports deferred to avoid early DB initialization
import os
import shutil
import sys
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def repair_database():
    print("Starting database repair...", flush=True)
    
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(base_dir, "data", "chroma_db")
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    
    # 2. Remove Corrupted DB
    if os.path.exists(db_dir):
        print(f"Removing corrupted database at: {db_dir}", flush=True)
        try:
            shutil.rmtree(db_dir)
            print("Database directory removed.", flush=True)
        except Exception as e:
            print(f"Error removing directory: {e}", flush=True)
            return
    else:
        print("No database directory found to remove.", flush=True)

    # 3. Find existing PDFs
    print(f"Checking PDF directory: {pdf_dir}", flush=True)
    if not os.path.exists(pdf_dir):
        print(f"No PDF directory found at {pdf_dir}. Creating it.", flush=True)
        os.makedirs(pdf_dir)
        return

    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to re-ingest.", flush=True)

    if not pdf_files:
        print("No PDFs found. Database has been reset and is empty.", flush=True)
        return

    # 4. Re-ingest
    # Now valid to import since DB is clean
    print("Importing src.ingest and src.scraper...", flush=True)
    try:
        from src import ingest, scraper
        print("Imports successful.", flush=True)
    except Exception as e:
        print(f"Error importing modules: {e}", flush=True)
        return

    print("Re-ingesting papers...", flush=True)
    
    # Get existing papers to skip them
    print("Fetching existing papers in DB to allow resuming...", flush=True)
    try:
        existing_papers = ingest.get_all_papers()
        existing_titles = set([p['title'] for p in existing_papers])
        print(f"Found {len(existing_titles)} already in DB.", flush=True)
    except Exception as e:
        print(f"Could not fetch existing papers (DB might be empty/broken): {e}", flush=True)
        existing_titles = set()

    success_count = 0
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            filename = os.path.basename(pdf_path)
            
            # Simple heuristic metadata recovery since we lost the original metadata DB
            # We try to use the scraper's resolve function if possible, or fallback to filename
            raw_title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")

            # Check if *likely* already in DB (heuristic title match)
            # Ideally we'd match exact titles, but since we resolve metadata dynamically, 
            # we might have minor mismatches if we don't resolve first.
            # To be safe/fast, we can check raw_title or check after resolution.
            # Let's do lazy resolution: only resolve if we think we might need it? 
            # No, correct is to resolve first then check.
            
            # Optimization: Check if filename-based entry_id exists? 
            # ingest.py uses md5(title) as ID base.
            # If we used "restored_filename" as entry_id in previous run, we could check that.
            # But get_all_papers() returns titles.
            
            # Let's resolve first (it's network heavy but safer) OR assume if we have 30 papers and 340 files, we just check title.
            # BUT: resolving 340 papers takes forever.
            # FAST PATH: If we used the same logic before, title should match.
            
            if raw_title in existing_titles:
                 print(f"[{i+1}/{len(pdf_files)}] Skipping {filename} (already in DB)", flush=True)
                 success_count += 1
                 continue
                 
            # If not exact match, maybe resolved title matches?
            # We'll check again after resolution.

            print(f"[{i+1}/{len(pdf_files)}] Processing {filename}...", flush=True)
            
            # Try to resolve real metadata (with timeout protection conceptually, though synchronous here)
            # print("  Resolving metadata...", flush=True)
            try:
                 res = scraper.resolve_paper_metadata(raw_title)
            except Exception as e:
                 print(f"  Metadata resolution failed: {e}", flush=True)
                 res = {}
            
            final_title = res.get("title", raw_title)
            
            if final_title in existing_titles:
                 print(f"  Skipping {final_title} (already in DB)", flush=True)
                 success_count += 1
                 continue
            
            metadata = {
                "title": final_title,
                "authors": res.get("authors", ["Unknown"]),
                "summary": res.get("summary", "Restored from local file"),
                "published": res.get("published", "Unknown"),
                "doi": res.get("doi", "Unknown"),
                "entry_id": "restored_" + filename
            }
            
            ingest.ingest_paper(pdf_path, metadata)
            success_count += 1
            
        except Exception as e:
            print(f"Failed to re-ingest {filename}: {e}", flush=True)

    print(f"\nRepair complete. Successfully restored {success_count}/{len(pdf_files)} papers.", flush=True)

if __name__ == "__main__":
    repair_database()
