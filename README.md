# Local Academic Assistant - Usage Guide

[English](README.md) | [简体中文](README.zh-CN.md)

## Setup

1.  **Install Dependencies**:
    Open a terminal in `p:\workspace\arxiv_assistant` and run:
    ```bash
    pip install -r requirements.txt
    ```
    Notes:
    - Google Scholar search requires `scholarly` (included in `requirements.txt`).
    - Sentence patterns require `spacy` plus the English model:
      ```bash
      python -m spacy download en_core_web_sm
      ```
    - Optional clustering requires `umap-learn` and `hdbscan`.

2.  **Install & Run Ollama**:
    - Download from [ollama.com](https://ollama.com).
    - Run `ollama run llama3` (or `mistral`) in a separate terminal to download and serve the model.
    - Ensure it is listening on port 11434 (default).

## Running the App

1.  Run the Streamlit app:
    ```bash
    streamlit run src/app.py
    ```
2.  The browser will open automatically.

## Features

1.  **Manage Database**: Use the sidebar to search arXiv for topics (e.g., "Mechanism Design") and download papers.
2.  **Search Literature**: In the "Semantic Search" tab, ask questions or search for concepts.
3.  **Academic Sentence Polisher**: In the "Rewrite" tab, paste a drafted sentence. The system will retrieve similar sentences from your downloaded papers and use the LLM to rewrite yours in a better style.
4.  **Research Idea Generator**: Enter a topic, and the system will read your library to suggest new directions.
5.  **Batch Import**: Run multi-keyword searches and auto-ingest deduped results.
6.  **Metadata Editing**: Edit title/authors/year/DOI directly from the UI.
7.  **Rich Exports**: Export library as BibTeX, JSON, CSL-JSON, RIS, or Zotero JSON.
8.  **Paper Summaries**: Generate paper-level summary cards.
9.  **Review References**: Literature reviews include a reference list.
10. **Style-Aware Rewriting**: Choose journal / working paper / grant styles.
11. **Bulk Download**: Select multiple search results and ingest in one click.
12. **Structured Ideas**: Structured output option for idea generation.
13. **Search History**: Persist recent searches and re-run quickly.
14. **Quick Filters**: Filter library by year, source, or author.
15. **Incremental Ingest**: Scan `data/pdfs` for new files and ingest only what’s missing.
16. **Diagnostics**: Startup checks for DB/LLM/dependencies.
17. **Retrieval Tuning UI**: Runtime controls for retrieval mode, alpha, reranker.
18. **Search UX Upgrade**: Query expansion, term highlighting, and evidence snippets with adjustable length.
19. **Metadata Coverage**: Venue propagation across search, enrichment, ingest, and exports.
20. **Knowledge Management**: Tags, favorites, and reading-list flags with quick filters and inline actions.
21. **Background Queue**: Async download -> ingest -> enrich tasks with cancel/retry/recovery.
22. **Conflict Merge UI**: Review duplicate groups and merge with metadata-preserving strategy.
23. **Evidence Answers**: Generate citation-tagged answers (`[S1]`, `[S2]`) with source confidence.
24. **Graph Analytics**: Citation, co-citation, bibliographic coupling, author collaboration views.
25. **Watchlist & Digests**: Daily/weekly keyword or author tracking with digest files.
26. **OCR Fallback**: Optional OCR extraction for scanned PDFs with low embedded text.

## Configuration

You can override defaults via environment variables:
- `ARXIV_ASSISTANT_DB_DIR`: custom ChromaDB storage path
- `ARXIV_ASSISTANT_EMBEDDING_MODEL`: sentence-transformers model name
- `ARXIV_ASSISTANT_CHUNK_SIZE`: chunk size (characters)
- `ARXIV_ASSISTANT_CHUNK_OVERLAP`: chunk overlap (characters)
- `ARXIV_ASSISTANT_RETRIEVAL_MODE`: `vector`, `bm25`, or `hybrid` (default)
- `ARXIV_ASSISTANT_HYBRID_ALPHA`: weight for vector score in hybrid mode (0-1)
- `ARXIV_ASSISTANT_RERANKER_MODEL`: optional CrossEncoder model name for reranking
- `ARXIV_ASSISTANT_RERANKER_TOP_K`: rerank top K candidates
- `ARXIV_ASSISTANT_HYBRID_CANDIDATE_MULTIPLIER`: candidate expansion for hybrid retrieval
- `ARXIV_ASSISTANT_HISTORY_PATH`: custom search history path
- `ARXIV_ASSISTANT_TASK_QUEUE_PATH`: background task queue JSON path
- `ARXIV_ASSISTANT_WATCHLIST_PATH`: watchlist JSON path
- `ARXIV_ASSISTANT_DIGEST_DIR`: digest output directory
- `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK`: `1`/`0` to enable OCR fallback
- `ARXIV_ASSISTANT_OCR_MAX_PAGES`: number of first pages to OCR when fallback triggers
- `ARXIV_ASSISTANT_OCR_MIN_TEXT_CHARS`: OCR fallback trigger threshold
- `SCHOLAR_PROXY`: single proxy URL for Google Scholar (e.g., http://user:pass@host:port)
- `SCHOLAR_USE_FREE_PROXY`: set to `1` to attempt free proxies
- `SCHOLAR_USE_TOR`: set to `1` to attempt Tor proxy

### Optional OCR Dependencies
- Python packages: `pdf2image`, `pytesseract`
- System tools: `poppler` and `tesseract`

### Docker Compose Notes
- `docker-compose.yml` persists ChromaDB at `/app/data/chroma_db` and search history at `/app/data/history/search_history.json`.
