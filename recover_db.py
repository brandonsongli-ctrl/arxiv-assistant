
import os
import sys
from dotenv import load_dotenv

# Load env before importing src
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from src import ingest, database

def main():
    data_dir = os.path.join("data", "pdfs")
    print(f"Scanning for PDFs in {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    # Force re-ingest logic by calling ingest_directory
    # We want to re-process even if it thinks it might be there (though DB is empty now)
    
    print("Starting recovery ingest...")
    
    def progress(current, total, msg):
        print(f"[{current}/{total}] {msg}")

    result = ingest.ingest_directory(
        data_dir,
        resolve_metadata=True,
        progress_callback=progress
    )
    
    print("\nRecovery Complete!")
    print(f"Total: {result['total']}")
    print(f"Ingested: {result['ingested']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    
    # Verify count
    count = database.get_paper_count()
    print(f"Final Database Count: {count}")

if __name__ == "__main__":
    main()
