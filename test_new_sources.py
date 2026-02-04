import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import scraper

def test_sources():
    print("Testing NBER Search...")
    nber_results = scraper.search_nber("mechanism design", max_results=3)
    print(f"Found {len(nber_results)} NBER results.")
    for r in nber_results:
        print(f" - {r['title']} ({r['published']}) [PDF: {r['pdf_url'] is not None}]")

    print("\nTesting SSRN Search...")
    ssrn_results = scraper.search_ssrn("contract theory", max_results=3)
    print(f"Found {len(ssrn_results)} SSRN results.")
    for r in ssrn_results:
        print(f" - {r['title']} ({r['published']}) [PDF: {r['pdf_url'] is not None}]")

if __name__ == "__main__":
    test_sources()
