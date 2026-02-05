
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src import scraper, ingest, database

def main():
    print("Searching for 'Attention Is All You Need'...")
    results = scraper.search_arxiv("Attention Is All You Need", max_results=1)
    
    if not results:
        print("No results found from arXiv search.")
        return

    paper = results[0]
    print(f"Found: {paper['title']}")
    
    print("Downloading...")
    pdf_path, error = ingest.extract_text_from_pdf(paper) if False else (None, None) # Mocking? No, let's use the app logic
    
    # We need to replicate app.py logic roughly
    download_dir = os.path.join("data", "pdfs")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    pdf_path = scraper.download_from_url(paper["pdf_url"], paper["title"], download_dir)
    
    if pdf_path:
        print(f"Downloaded to {pdf_path}")
        print("Ingesting...")
        success = ingest.ingest_paper(pdf_path, paper)
        if success:
            print("Successfully ingested paper!")
            print(f"Total papers in DB now: {len(database.get_all_papers())}")
        else:
            print("Ingestion failed.")
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()
