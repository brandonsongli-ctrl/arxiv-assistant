import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import ingest

def test_ingest_metadata_sanitization():
    print("Testing metadata sanitization...")
    
    # Mock extract_text_from_pdf to avoid pypdf issues with dummy file
    original_extract = ingest.extract_text_from_pdf
    ingest.extract_text_from_pdf = lambda path: "Dummy text content for testing metadata."
    
    dummy_pdf = "mock_path.pdf"
    
    try:
        # Metadata with None values designed to trigger the bug
        bad_metadata = {
            "title": "Test Paper",
            "authors": None, # Should become "Unknown"
            "summary": None, # Should become "Unknown"
            "published": None, # Should become "Unknown"
            "doi": None, # Should become "Unknown"
            "entry_id": "test_id_123"
        }
        
        print("Ingesting paper with None metadata...")
        ingest.ingest_paper(dummy_pdf, bad_metadata)
        print("Success! Ingestion did not crash.")
        
    except Exception as e:
        print(f"FAILED: Ingestion crashed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(dummy_pdf):
            os.remove(dummy_pdf)

if __name__ == "__main__":
    test_ingest_metadata_sanitization()
