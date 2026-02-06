import os
import hashlib
import re
from typing import List, Dict, Tuple, Any
from pypdf import PdfReader
from src import database, config
from src.metadata_utils import compute_canonical_id, extract_arxiv_id, extract_openalex_id, normalize_doi


ENABLE_OCR_FALLBACK = os.getenv("ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK", "1") == "1"
OCR_MAX_PAGES = int(os.getenv("ARXIV_ASSISTANT_OCR_MAX_PAGES", "3"))
OCR_MIN_TEXT_CHARS = int(os.getenv("ARXIV_ASSISTANT_OCR_MIN_TEXT_CHARS", "120"))
PARSE_QUALITY_HIGH = float(os.getenv("ARXIV_ASSISTANT_PARSE_QUALITY_HIGH", "0.75"))
PARSE_QUALITY_MEDIUM = float(os.getenv("ARXIV_ASSISTANT_PARSE_QUALITY_MEDIUM", "0.45"))


def _clean_extracted_text(text: str) -> str:
    text = str(text or "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _looks_like_table_row(line: str) -> bool:
    line = str(line or "").strip()
    if len(line) < 10:
        return False
    if re.search(r"\s{2,}", line) and len(re.findall(r"\d", line)) >= 2:
        return True
    if line.count("|") >= 2:
        return True
    if re.match(r"^[A-Za-z].{0,30}\s+\d+(\.\d+)?\s+\d+(\.\d+)?", line):
        return True
    return False


def _looks_like_formula_line(line: str) -> bool:
    line = str(line or "").strip()
    if len(line) < 6:
        return False
    if re.search(r"\\(frac|sum|int|alpha|beta|gamma|theta|lambda)", line):
        return True
    symbol_count = len(re.findall(r"[=<>+\-*/^∑∫√]", line))
    digit_count = len(re.findall(r"\d", line))
    if symbol_count >= 2 and (digit_count > 0 or any(ch.isalpha() for ch in line)):
        return True
    if re.match(r"^\(?\d+(\.\d+)?\)?\s*[).-]?\s*[A-Za-z].*=", line):
        return True
    return False


def _annotate_layout_text(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Add lightweight layout-aware tags for table rows, formulas and figure captions.
    """
    stats = {
        "table_lines": 0,
        "formula_lines": 0,
        "figure_caption_lines": 0,
        "table_caption_lines": 0,
    }
    out_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue
        if re.match(r"^(figure|fig\.)\s*\d+[:.\-)]", line, re.IGNORECASE):
            stats["figure_caption_lines"] += 1
            out_lines.append(f"[FIGURE_CAPTION] {line}")
            continue
        if re.match(r"^table\s*\d+[:.\-)]", line, re.IGNORECASE):
            stats["table_caption_lines"] += 1
            out_lines.append(f"[TABLE_CAPTION] {line}")
            continue
        if _looks_like_table_row(line):
            stats["table_lines"] += 1
            out_lines.append(f"[TABLE] {line}")
            continue
        if _looks_like_formula_line(line):
            stats["formula_lines"] += 1
            out_lines.append(f"[FORMULA] {line}")
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip(), stats


def _ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """
    OCR one page only (1-based page_number).
    """
    if not ENABLE_OCR_FALLBACK:
        return ""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
    except Exception:
        return ""
    if not images:
        return ""
    try:
        txt = pytesseract.image_to_string(images[0])
        return _clean_extracted_text(txt)
    except Exception:
        return ""


def _score_parse_quality(page_texts: List[str], layout_stats: Dict[str, int], ocr_pages: int) -> Dict[str, Any]:
    pages = max(1, len(page_texts))
    merged_text = "\n\n".join(page_texts).strip()
    total_chars = len(merged_text)
    avg_chars_per_page = total_chars / pages
    low_text_pages = sum(1 for t in page_texts if len(str(t or "").strip()) < OCR_MIN_TEXT_CHARS)
    low_text_ratio = low_text_pages / pages
    ocr_ratio = ocr_pages / pages

    bad_char_count = len(re.findall(r"[^\x09\x0A\x0D\x20-\x7E]", merged_text))
    bad_char_ratio = (bad_char_count / max(1, total_chars)) if total_chars else 1.0

    score = 1.0
    if total_chars < 600:
        score -= 0.35
    if avg_chars_per_page < 250:
        score -= 0.20
    if low_text_ratio > 0.50:
        score -= 0.20
    if ocr_ratio > 0.70:
        score -= 0.10
    if bad_char_ratio > 0.08:
        score -= 0.10
    if layout_stats.get("table_lines", 0) > 0 or layout_stats.get("figure_caption_lines", 0) > 0:
        score += 0.05
    score = max(0.0, min(1.0, score))

    if score >= PARSE_QUALITY_HIGH:
        label = "high"
    elif score >= PARSE_QUALITY_MEDIUM:
        label = "medium"
    else:
        label = "low"

    return {
        "quality_score": round(score, 4),
        "quality_label": label,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars_per_page, 2),
        "low_text_pages": low_text_pages,
        "low_text_ratio": round(low_text_ratio, 4),
        "ocr_ratio": round(ocr_ratio, 4),
        "bad_char_ratio": round(bad_char_ratio, 4),
    }


def extract_text_from_pdf_with_report(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract PDF text with:
    1) layout-aware line tagging (table/formula/caption),
    2) segmented OCR fallback for low-text pages,
    3) parse quality scoring.
    """
    report: Dict[str, Any] = {
        "pages": 0,
        "ocr_pages": 0,
        "ocr_used": False,
        "ocr_replaced_pages": [],
        "extraction_mode": "embedded_only",
        "layout": {},
        "quality_score": 0.0,
        "quality_label": "low",
    }
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return "", report

    page_texts: List[str] = []
    report["pages"] = len(reader.pages)
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
            page_texts.append(_clean_extracted_text(extracted))
        except Exception:
            page_texts.append("")

    # Segmented OCR fallback only on pages that are low-text.
    if ENABLE_OCR_FALLBACK and page_texts:
        low_text_page_indices = [
            idx for idx, txt in enumerate(page_texts)
            if len(str(txt or "").strip()) < OCR_MIN_TEXT_CHARS
        ]
        max_ocr_pages = max(1, OCR_MAX_PAGES)
        for idx in low_text_page_indices[:max_ocr_pages]:
            page_number = idx + 1
            ocr_text = _ocr_pdf_page(pdf_path, page_number)
            if len(ocr_text.strip()) > len(str(page_texts[idx] or "").strip()):
                page_texts[idx] = ocr_text
                report["ocr_pages"] += 1
                report["ocr_replaced_pages"].append(page_number)

    report["ocr_used"] = report["ocr_pages"] > 0
    if report["ocr_used"] and report["ocr_pages"] >= report["pages"]:
        report["extraction_mode"] = "ocr_only"
    elif report["ocr_used"]:
        report["extraction_mode"] = "embedded_plus_segmented_ocr"

    combined_text = "\n\n".join([t for t in page_texts if t is not None]).strip()
    annotated_text, layout_stats = _annotate_layout_text(combined_text)
    quality = _score_parse_quality(page_texts, layout_stats, ocr_pages=int(report["ocr_pages"]))

    report["layout"] = layout_stats
    report.update(quality)
    return annotated_text, report


def _ocr_pdf_fallback(pdf_path: str, max_pages: int = 3) -> str:
    """
    Backward-compatible helper for older call sites.
    """
    texts = []
    for page_num in range(1, max_pages + 1):
        txt = _ocr_pdf_page(pdf_path, page_num)
        if txt:
            texts.append(txt)
    return "\n".join(texts).strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text, _ = extract_text_from_pdf_with_report(pdf_path)
    return text

def _split_long_text(text: str, max_len: int) -> List[str]:
    """Fallback splitter for very long sentences."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks using paragraph and sentence boundaries.
    """
    if not text:
        return []
        
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
        
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, int(overlap))
    
    # Normalize line endings and de-hyphenate common PDF line wraps
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r'-\n', '', normalized)
    
    # Split into paragraphs by blank lines
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', normalized) if p.strip()]
    
    units: List[str] = []
    for para in raw_paragraphs:
        para = re.sub(r'\s*\n\s*', ' ', para)  # join single line breaks
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
            
        if len(para) <= chunk_size:
            units.append(para)
            continue
            
        # Split long paragraphs into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                units.extend(_split_long_text(sent, chunk_size))
    
    # Build chunks from units
    chunks: List[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        if len(current) + 1 + len(unit) <= chunk_size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
        
    # Apply overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            prefix = prev[-min(overlap, len(prev)):]
            merged = f"{prefix} {chunks[i]}".strip()
            overlapped.append(merged)
        chunks = overlapped
        
    return chunks

def ingest_paper(pdf_path: str, metadata: Dict, run_incremental_pipeline: bool = True, run_metadata_fix: bool = True):
    """
    Process a PDF and add it to the PostgreSQL vector DB.
    """
    print(f"Ingesting {pdf_path}...")
    
    # 1. Extract Text
    text, parse_report = extract_text_from_pdf_with_report(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return False

    # 2. Add/Get Paper Record
    # Ensure metadata has 'source'
    if 'source' not in metadata or metadata['source'] == 'manual_upload':
        metadata['source'] = pdf_path
    metadata.setdefault("venue", "")
    metadata.setdefault("tags", "")
    metadata.setdefault("is_favorite", False)
    metadata.setdefault("in_reading_list", False)
    metadata["parse_pages"] = int(parse_report.get("pages", 0))
    metadata["parse_ocr_pages"] = int(parse_report.get("ocr_pages", 0))
    metadata["parse_extraction_mode"] = str(parse_report.get("extraction_mode", "embedded_only"))
    metadata["parse_quality_score"] = float(parse_report.get("quality_score", 0.0))
    metadata["parse_quality_label"] = str(parse_report.get("quality_label", "low"))
    layout_stats = parse_report.get("layout", {}) or {}
    metadata["parse_table_lines"] = int(layout_stats.get("table_lines", 0))
    metadata["parse_formula_lines"] = int(layout_stats.get("formula_lines", 0))
    metadata["parse_figure_caption_lines"] = int(layout_stats.get("figure_caption_lines", 0))
    
    # Normalize identifiers
    if metadata.get("doi"):
        metadata["doi"] = normalize_doi(metadata.get("doi"))
    if not metadata.get("arxiv_id") and metadata.get("entry_id"):
        metadata["arxiv_id"] = extract_arxiv_id(str(metadata.get("entry_id")))
    if not metadata.get("openalex_id") and metadata.get("entry_id"):
        metadata["openalex_id"] = extract_openalex_id(str(metadata.get("entry_id")))
    metadata["canonical_id"] = compute_canonical_id(metadata)

    # Skip duplicates for strong IDs
    canonical_id = metadata.get("canonical_id", "")
    if canonical_id.startswith(("doi:", "arxiv:", "openalex:")):
        if database.has_paper_by_metadata("canonical_id", canonical_id):
            print(f"Skipping duplicate paper (canonical_id={canonical_id})")
            return False
        
    paper_id = database.insert_paper(metadata)
    # print(f"Paper '{metadata.get('title')}' ID: {paper_id}")
    
    # 3. Create Chunks
    chunks = chunk_text(text)
    if not chunks:
        print(f"Warning: No text chunks created for {pdf_path}")
        return False
        
    # print(f"Generated {len(chunks)} chunks.")
    
    # 4. Insert Chunks (and generate embeddings automatically in database module)
    # For ChromaDB, we must pass metadata to attach to chunks
    database.insert_chunks(paper_id, chunks, metadata)
    # print(f"Successfully ingested {len(chunks)} chunks to ChromaDB.")

    if run_incremental_pipeline:
        try:
            from src import pipeline

            pipe_result = pipeline.run_incremental_indexing(
                metadata.get("title", ""),
                run_metadata_fix=bool(run_metadata_fix),
            )
            if pipe_result.get("errors"):
                print(f"Incremental pipeline warnings for {metadata.get('title', '')}: {pipe_result.get('errors')}")
        except Exception as e:
            print(f"Warning: incremental pipeline failed for {metadata.get('title', '')}: {e}")

    return True


def ingest_directory(directory: str, resolve_metadata: bool = True, progress_callback=None) -> Dict:
    """
    Incrementally ingest new PDFs from a directory.
    Skips files already present in the database (by entry_id or source path).
    """
    if not os.path.exists(directory):
        return {
            "total": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "message": f"Directory not found: {directory}"
        }
    
    from src import scraper, dedupe
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    files.sort()
    
    existing = database.get_all_papers()
    existing_entry_ids = {str(p.get("entry_id", "")) for p in existing}
    existing_sources = {str(p.get("source", "")) for p in existing}
    existing_titles = {dedupe.normalize_title(p.get("title", "")) for p in existing}
    
    total = len(files)
    ingested = 0
    skipped = 0
    failed = 0
    
    for idx, fname in enumerate(files):
        file_path = os.path.join(directory, fname)
        entry_id = f"manual_{fname}"
        
        if entry_id in existing_entry_ids or file_path in existing_sources:
            skipped += 1
            if progress_callback:
                progress_callback(idx + 1, total, f"Skipping existing: {fname}")
            continue
        
        raw_title = fname.replace(".pdf", "").replace("_", " ").replace("-", " ")
        title_norm = dedupe.normalize_title(raw_title)
        if title_norm in existing_titles:
            skipped += 1
            if progress_callback:
                progress_callback(idx + 1, total, f"Skipping duplicate title: {fname}")
            continue
        
        final_title = raw_title
        final_authors = ["Unknown"]
        final_doi = "Unknown"
        final_summary = "Manual Upload"
        final_published = "Unknown"
        final_venue = ""
        
        if resolve_metadata:
            resolved = scraper.resolve_paper_metadata(raw_title)
            if resolved.get("found"):
                final_title = resolved.get("title") or final_title
                final_authors = resolved.get("authors") or final_authors
                final_doi = resolved.get("doi") or final_doi
                final_summary = resolved.get("summary") or final_summary
                final_published = resolved.get("published") or final_published
                final_venue = resolved.get("venue") or final_venue
        
        metadata = {
            "title": final_title,
            "authors": final_authors,
            "summary": final_summary,
            "published": final_published,
            "doi": final_doi,
            "venue": final_venue,
            "entry_id": entry_id,
            "source": file_path,
            "tags": "",
            "is_favorite": False,
            "in_reading_list": False
        }
        
        # Skip duplicates based on strong IDs when available
        canonical_id = compute_canonical_id(metadata)
        if canonical_id.startswith(("doi:", "arxiv:", "openalex:")):
            if database.has_paper_by_metadata("canonical_id", canonical_id):
                skipped += 1
                if progress_callback:
                    progress_callback(idx + 1, total, f"Skipping duplicate ID: {fname}")
                continue
        
        try:
            success = ingest_paper(file_path, metadata)
            if success:
                ingested += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        
        if progress_callback:
            progress_callback(idx + 1, total, f"Ingested: {fname}")
    
    return {
        "total": total,
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        "message": f"Ingested {ingested}, skipped {skipped}, failed {failed}."
    }

def get_all_papers() -> List[Dict]:
    """
    Retrieve listing of all unique papers in the DB.
    Wrapper for database.get_all_papers() to maintain API compatibility.
    """
    return database.get_all_papers()

