import os
import glob
from tqdm import tqdm
from src import database, ingest, scraper
from src import enrich

def restore_db():
    print("==================================================")
    print("RESTORING DATABASE FROM PDFS (LOCAL MODE)")
    print("==================================================")
    
    # 1. Initialize DB
    database.init_db()
    
    # 2. Find PDFs
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("No PDFs found to ingest.")
        return
        
    print(f"Found {total_files} PDFs to ingest.")
    
    # 3. Ingest Loop (Fast)
    print("\nPhase 1: Ingesting into ChromaDB (Fast)...")
    success_count = 0
    
    failed_list = []
    
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Ingesting")):
        filename = os.path.basename(pdf_path)
        
        # Basic cleanup for title
        final_title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
        
        metadata = {
            "title": final_title,
            "authors": ["Unknown"],
            "published": "Unknown",
            "doi": "Unknown",
            "summary": "Restored from local PDF",
            "entry_id": "restored_" + filename,
            "source": pdf_path
        }
        
        try:
            if ingest.ingest_paper(pdf_path, metadata):
                success_count += 1
            else:
                failed_list.append(filename)
        except Exception as e:
            print(f"  ❌ Failed to ingest {filename}: {str(e)}")
            failed_list.append(f"{filename} ({str(e)})")

    print(f"\nPhase 1 Complete. Successfully ingested {success_count}/{total_files} papers.")
    if failed_list:
        print(f"\n⚠️ {len(failed_list)} papers failed to ingest (likely empty text/scanned):")
        for f in failed_list[:10]:
            print(f"  - {f}")
        if len(failed_list) > 10: print(f"  ...and {len(failed_list)-10} more.")
    
    # 4. Auto-Enrichment (Smart)
    print("\nPhase 2: Auto-Enriching Metadata (Smart Read)...")
    
    try:
        def progress(current, total, msg):
            print(f"[{current}/{total}] {msg}")
            
        results = enrich.enrich_all_papers(progress_callback=progress)
        print(f"\nPhase 2 Complete! {results.get('message', 'Done')}")
        
    except Exception as e:
        print(f"\nPhase 2 Warning: {e}")

if __name__ == "__main__":
    restore_db()
