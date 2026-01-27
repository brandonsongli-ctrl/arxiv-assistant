import os
import shutil
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def reset_database():
    print("Resetting database to empty state...")
    
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(base_dir, "data", "chroma_db")
    
    # 2. Remove Corrupted DB
    if os.path.exists(db_dir):
        print(f"Removing database at: {db_dir}")
        try:
            shutil.rmtree(db_dir)
            print("Database directory removed.")
        except Exception as e:
            print(f"Error removing directory: {e}")
            return
    else:
        print("No database directory found to remove.")

    # 3. Initialize fresh DB (implicitly creates the dir)
    print("Initializing fresh ChromaDB client...")
    import chromadb
    try:
        client = chromadb.PersistentClient(path=db_dir)
        print("Fresh database initialized successfully.")
    except Exception as e:
        print(f"Error initializing fresh database: {e}")

if __name__ == "__main__":
    reset_database()
