import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import scraper

def test_google_scholar():
    print("Testing Google Scholar Search...")
    # Using a very specific query to minimize rate limit risk
    results = scraper.search_google_scholar("Athey Imbens Machine Learning", max_results=2)
    
    print(f"Found {len(results)} results.")
    for r in results:
        print(f" - {r['title']} ({r['published']}) [Source: {r['source']}]")
        if r['pdf_url']:
            print(f"   PDF: {r['pdf_url']}")
        else:
            print("   PDF: None")

if __name__ == "__main__":
    test_google_scholar()
