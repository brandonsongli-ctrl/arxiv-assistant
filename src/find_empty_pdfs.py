import os
import glob
from pypdf import PdfReader

def check_empty_pdfs():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    print(f"Scanning {len(pdf_files)} PDFs for text content...")
    
    empty_files = []
    error_files = []
    
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text_len = 0
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_len += len(extracted.strip())
            
            if text_len < 50: # Arbitrary threshold for "empty"
                empty_files.append((os.path.basename(pdf), text_len))
                
        except Exception as e:
            error_files.append((os.path.basename(pdf), str(e)))
            
    print(f"\nFound {len(empty_files)} Empty/Scanned PDFs:")
    for f, l in empty_files:
        print(f"  [Empty] {f} ({l} chars)")
        
    print(f"\nFound {len(error_files)} Corrupted/Encrypted PDFs:")
    for f, e in error_files:
        print(f"  [Error] {f}: {e}")

if __name__ == "__main__":
    check_empty_pdfs()
