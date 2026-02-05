import streamlit as st
import os
import sys
import time
import re
from datetime import date, timedelta

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import scraper, ingest, rag, retrieval
from src import bibtex, citations, patterns, clustering, database, dedupe, exports, summaries, history, diagnostics
from src.metadata_utils import compute_canonical_id, normalize_doi, extract_arxiv_id, extract_openalex_id
# review is imported lazily inside its tab to avoid circular import issues

st.set_page_config(page_title="Local Academic Assistant", layout="wide")

st.title("📚 Local Academic Literature Assistant")
st.markdown("Focused on **Microeconomic Theory, Information Design, Contract Theory, and Mechanism Design**.")

@st.cache_data(show_spinner=False)
def load_papers_cached():
    return ingest.get_all_papers()


def format_authors(authors) -> str:
    if not authors:
        return "Unknown"
    if isinstance(authors, list):
        return ", ".join([str(a) for a in authors if a])
    return str(authors)


def format_title(paper) -> str:
    if not paper:
        return "Untitled"
    title = paper.get("title") or paper.get("name")
    return str(title) if title else "Untitled"


def format_published(paper) -> str:
    if not paper:
        return "Unknown"
    return str(paper.get("published", "Unknown"))


def _split_top_level_alternation(text: str) -> list:
    parts = []
    buf = []
    depth = 0
    in_brackets = 0
    escaped = False
    
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == "[":
            in_brackets += 1
            buf.append(ch)
            continue
        if ch == "]" and in_brackets > 0:
            in_brackets -= 1
            buf.append(ch)
            continue
        if in_brackets > 0:
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            buf.append(ch)
            continue
        if ch == "|" and depth == 0 and in_brackets == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    
    parts.append("".join(buf))
    return parts


def _find_first_group(text: str):
    depth = 0
    in_brackets = 0
    escaped = False
    start = None
    
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "[":
            in_brackets += 1
            continue
        if ch == "]" and in_brackets > 0:
            in_brackets -= 1
            continue
        if in_brackets > 0:
            continue
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                end = i
                optional = False
                if end + 1 < len(text) and text[end + 1] == "?":
                    optional = True
                return start, end, optional
    return None


def _strip_group_prefix(inner: str) -> str:
    if inner.startswith("?:"):
        return inner[2:]
    if inner.startswith("?P<"):
        close = inner.find(">")
        if close != -1:
            return inner[close + 1:]
    return inner


def expand_pattern_variants(pattern_text: str, max_variants: int = 200) -> list:
    def _expand(text: str) -> list:
        group_info = _find_first_group(text)
        if not group_info:
            return [text]
        
        start, end, optional = group_info
        prefix = text[:start]
        inner = _strip_group_prefix(text[start + 1:end])
        suffix = text[end + 1:]
        
        if optional and suffix.startswith("?"):
            suffix = suffix[1:]
        
        alternatives = _split_top_level_alternation(inner)
        if not alternatives:
            alternatives = [inner]
        
        next_texts = []
        for alt in alternatives:
            next_texts.append(prefix + alt + suffix)
        if optional:
            next_texts.append(prefix + suffix)
        
        results = []
        for candidate in next_texts:
            results.extend(_expand(candidate))
            if len(results) >= max_variants:
                break
        return results[:max_variants]
    
    expanded = _expand(pattern_text)
    cleaned = []
    seen = set()
    for text in expanded:
        display = text.replace("(", "").replace(")", "")
        display = clean_pattern_display(display)
        if display and display not in seen:
            cleaned.append(display)
            seen.add(display)
    return cleaned


def strip_brackets(text: str) -> str:
    if not text:
        return ""
    return text.replace("[", "").replace("]", "")


def clean_pattern_display(text: str) -> str:
    if not text:
        return ""
    display = strip_brackets(text)
    display = re.sub(r"\\[wsd][+*?]?", "...", display)
    display = re.sub(r"\[[^\]]+\]", "...", display)
    display = re.sub(r"\{[^}]+\}", "", display)
    display = display.replace("\\", "")
    display = display.replace("?", "")
    display = display.replace("+", "")
    display = display.replace("*", "")
    display = re.sub(r"\s+", " ", display).strip()
    return display


def map_struct_category(raw_cat: str, template: str) -> str:
    raw_cat = (raw_cat or "").upper()
    template_lower = (template or "").lower()
    
    cat_map = {
        "INTRO": "INTRO",
        "MODEL": "MODEL",
        "ARGUMENT": "ARGUMENT",
        "LOGIC": "LOGIC",
        "RESULT": "RESULT",
        "ECON": "ECON",
        "DISC": "DISC",
        "LIT": "LIT",
        "META": "META",
        "CONCLUSION": "LOGIC",
        "CONTRAST": "LOGIC",
        "CONNECTOR": "LOGIC",
        "CONDITION": "LOGIC",
        "ATTENTION": "DISC",
        "INTUITION": "DISC",
        "REFERENCE": "LIT",
        "PROOF": "META",
        "ASSUMPTION": "MODEL",
        "STATE": "RESULT",
        "QUALIFIER": "RESULT",
        "ORD": "INTRO",
    }
    
    if raw_cat in cat_map:
        return cat_map[raw_cat]
    
    # Heuristic fallbacks based on template content
    if "we [verb]" in template_lower or "we show that" in template_lower or "we prove that" in template_lower:
        return "ARGUMENT"
    if template_lower.startswith("let [var]") or "there exists" in template_lower or template_lower.startswith("for [quantifier]"):
        return "MODEL"
    if "the [qualifier]" in template_lower or "in equilibrium" in template_lower or "proposition" in template_lower or "theorem" in template_lower or "[result]" in template_lower:
        return "RESULT"
    if template_lower.startswith("it is [adj] to [verb]") or "then [consequence]" in template_lower or "[conclusion]" in template_lower or "[contrast]" in template_lower or "[connector]" in template_lower:
        return "LOGIC"
    if "roadmap" in template_lower or "the [adj] [noun] is" in template_lower or "[intro]" in template_lower:
        return "INTRO"
    if "reference" in template_lower or "[reference]" in template_lower:
        return "LIT"
    if "proof" in template_lower or "[proof]" in template_lower:
        return "META"
    if "intuition" in template_lower or "attention" in template_lower:
        return "DISC"
    
    return "Other"


def download_paper_for_ingest(paper):
    download_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
    source = (paper.get("source") or "").lower()
    pdf_path = None
    error = None
    
    if source in ["semanticscholar", "nber", "ssrn", "google_scholar", "google_scholar_fallback"]:
        if paper.get("pdf_url"):
            pdf_path = scraper.download_from_url(paper["pdf_url"], paper.get("title", "paper"), download_dir)
        else:
            error = "No open access PDF available for this paper."
    else:
        if paper.get("obj"):
            pdf_path = scraper.download_paper(paper["obj"], download_dir)
        elif paper.get("pdf_url"):
            pdf_path = scraper.download_from_url(paper["pdf_url"], paper.get("title", "paper"), download_dir)
        else:
            error = "No PDF link found for this paper."
    
    return pdf_path, error

# Sidebar: Manage Database
with st.sidebar:
    st.header("Manage Database")
    
    st.subheader("Add Papers")
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "Mechanism Design"
    if "search_source" not in st.session_state:
        st.session_state["search_source"] = "ArXiv"
    if "search_sort" not in st.session_state:
        st.session_state["search_sort"] = "Relevance"
    if "search_category" not in st.session_state:
        st.session_state["search_category"] = ""
    if "search_date_filter" not in st.session_state:
        st.session_state["search_date_filter"] = False
    if "search_start_date" not in st.session_state:
        st.session_state["search_start_date"] = date.today() - timedelta(days=365)
    if "search_end_date" not in st.session_state:
        st.session_state["search_end_date"] = date.today()
    
    source = st.selectbox("Source", ["ArXiv", "Semantic Scholar", "NBER", "SSRN", "Google Scholar"], key="search_source")

    # Search controls
    col_search, col_sort = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("Search Keywords or ID", key="search_query")
    with col_sort:
        if source == "ArXiv":
            sort_option = st.selectbox("Sort By", ["Relevance", "Last Updated", "Submitted Date"], key="search_sort")
        else:
            sort_option = "Relevance"
            st.selectbox("Sort By", ["Relevance"], disabled=True, key="search_sort_disabled")
    
    arxiv_category = ""
    date_filter_enabled = False
    start_date = None
    end_date = None
    if source == "ArXiv":
        arxiv_category = st.text_input("arXiv Category optional", key="search_category", help="Examples: econ.TH, cs.LG. You can enter multiple separated by commas.")
        date_filter_enabled = st.checkbox("Filter by date range", key="search_date_filter")
        if date_filter_enabled:
            col_date_1, col_date_2 = st.columns(2)
            with col_date_1:
                start_date = st.date_input("Start date", key="search_start_date")
            with col_date_2:
                end_date = st.date_input("End date", key="search_end_date")
    
    max_results = st.slider("Max Results", min_value=1, max_value=20, value=5, help="Higher values may be slower or rate-limited.")
    
    sort_map = {
        "Relevance": "relevance",
        "Last Updated": "last_updated",
        "Submitted Date": "submitted_date"
    }

    if 'search_seed' not in st.session_state:
        st.session_state['search_seed'] = 0
    
    if st.button("Search"):
        with st.spinner(f"Searching {source}..."):
            if source == "ArXiv":
                date_field = "submitted" if sort_option != "Last Updated" else "last_updated"
                results = scraper.search_arxiv(
                    search_query,
                    max_results=max_results,
                    sort_by=sort_map[sort_option],
                    category=arxiv_category.strip() or None,
                    date_from=start_date if date_filter_enabled else None,
                    date_to=end_date if date_filter_enabled else None,
                    date_field=date_field
                )
            elif source == "Semantic Scholar":
                results = scraper.search_semantic_scholar(search_query, max_results=max_results)
            elif source == "NBER":
                results = scraper.search_nber(search_query, max_results=max_results)
            elif source == "SSRN":
                results = scraper.search_ssrn(search_query, max_results=max_results)
            elif source == "Google Scholar":
                results = scraper.search_google_scholar(search_query, max_results=max_results)
            raw_count = len(results)
            results = scraper.dedupe_results(results)
            st.session_state['search_results'] = results
            st.session_state['search_results_raw_count'] = raw_count
            st.session_state['search_seed'] += 1
            
            history.append_history({
                "query": search_query,
                "source": source,
                "sort": sort_option,
                "category": arxiv_category.strip() if source == "ArXiv" else "",
                "date_filter": bool(date_filter_enabled) if source == "ArXiv" else False,
                "date_from": str(start_date) if (source == "ArXiv" and date_filter_enabled) else "",
                "date_to": str(end_date) if (source == "ArXiv" and date_filter_enabled) else ""
            })
    
    st.subheader("🕘 Search History")
    history_items = history.load_history(limit=20)
    if not history_items:
        st.caption("No search history yet.")
    else:
        for idx, item in enumerate(reversed(history_items)):
            label = f"{item.get('source', '')}: {item.get('query', '')[:50]}"
            if st.button(f"Use {label}", key=f"hist_{idx}"):
                st.session_state["search_query"] = item.get("query", "")
                st.session_state["search_source"] = item.get("source", "ArXiv")
                st.session_state["search_sort"] = item.get("sort", "Relevance")
                if st.session_state["search_sort"] == "Relevance (Default)":
                    st.session_state["search_sort"] = "Relevance"
                st.session_state["search_category"] = item.get("category", "")
                st.session_state["search_date_filter"] = bool(item.get("date_filter"))
                if item.get("date_from"):
                    try:
                        st.session_state["search_start_date"] = date.fromisoformat(item.get("date_from"))
                    except Exception:
                        pass
                if item.get("date_to"):
                    try:
                        st.session_state["search_end_date"] = date.fromisoformat(item.get("date_to"))
                    except Exception:
                        pass
        if st.button("Clear History"):
            history.clear_history()
            st.success("Search history cleared.")
            st.rerun()
            
    if 'search_results' in st.session_state:
        raw_count = st.session_state.get('search_results_raw_count', len(st.session_state['search_results']))
        if raw_count != len(st.session_state['search_results']):
            st.write(f"Found {len(st.session_state['search_results'])} unique papers, deduped from {raw_count} results:")
        else:
            st.write(f"Found {len(st.session_state['search_results'])} papers:")
        
        search_seed = st.session_state.get('search_seed', 0)
        select_all = st.checkbox("Select all results", key=f"select_all_{search_seed}")
        if select_all:
            for i in range(len(st.session_state['search_results'])):
                st.session_state[f"select_{search_seed}_{i}"] = True
        
        selected_indices = []
        for i, paper in enumerate(st.session_state['search_results']):
            title = format_title(paper)
            with st.expander(title):
                selected = st.checkbox("Select", key=f"select_{search_seed}_{i}")
                if selected:
                    selected_indices.append(i)
                st.caption(f"Authors: {format_authors(paper.get('authors'))}")
                st.caption(f"Published: {format_published(paper)}")
                st.write(paper.get('summary', 'No summary available.'))
                if st.button(f"Download and Ingest {i}", key=f"btn_{i}"):
                    with st.spinner("Downloading and ingesting..."):
                        # Download
                        pdf_path, error = download_paper_for_ingest(paper)
                        if error:
                            st.error(error)
                        
                        if pdf_path:
                            st.success(f"Downloaded to {pdf_path}")
                            
                            # Ingest
                            ingest.ingest_paper(pdf_path, paper)
                            st.success("Added to Database!")
                            st.cache_data.clear()
            st.divider()
        
        if selected_indices:
            if st.button("⬇️ Bulk Download & Ingest Selected"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                successes = 0
                failures = 0
                
                for idx, i in enumerate(selected_indices):
                    paper = st.session_state['search_results'][i]
                    status_text.text(f"Downloading and ingesting {idx+1}/{len(selected_indices)}: {paper.get('title', 'Unknown')[:60]}")
                    pdf_path, _ = download_paper_for_ingest(paper)
                    
                    if pdf_path:
                        ingest.ingest_paper(pdf_path, paper)
                        successes += 1
                    else:
                        failures += 1
                    
                    progress_bar.progress((idx + 1) / len(selected_indices))
                
                status_text.text("")
                st.success(f"Bulk ingest complete. Added {successes} papers. Failed {failures}.")
                st.cache_data.clear()

    st.markdown("---")
    st.subheader("Batch Import")
    batch_source = st.selectbox("Batch Source", ["ArXiv", "Semantic Scholar", "NBER", "SSRN", "Google Scholar"], key="batch_source")
    batch_queries = st.text_area("Queries one per line", height=100, key="batch_queries")
    batch_max_results = st.slider("Results per query", min_value=1, max_value=10, value=3, key="batch_results_per_query")
    batch_max_total = st.slider("Max total results", min_value=1, max_value=100, value=30, key="batch_max_total")
    batch_auto_ingest = st.checkbox("Auto-ingest deduped results", value=True, key="batch_auto_ingest")
    
    if st.button("Run Batch Search"):
        queries = [q.strip() for q in batch_queries.splitlines() if q.strip()]
        if not queries:
            st.warning("No queries provided.")
        else:
            aggregated = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner("Running batch search..."):
                for idx, q in enumerate(queries):
                    status_text.text(f"Searching {idx+1}/{len(queries)}: {q[:60]}")
                    if batch_source == "ArXiv":
                        results = scraper.search_arxiv(q, max_results=batch_max_results, sort_by="relevance")
                    elif batch_source == "Semantic Scholar":
                        results = scraper.search_semantic_scholar(q, max_results=batch_max_results)
                    elif batch_source == "NBER":
                        results = scraper.search_nber(q, max_results=batch_max_results)
                    elif batch_source == "SSRN":
                        results = scraper.search_ssrn(q, max_results=batch_max_results)
                    elif batch_source == "Google Scholar":
                        results = scraper.search_google_scholar(q, max_results=batch_max_results)
                    else:
                        results = []
                    
                    aggregated.extend(results)
                    progress_bar.progress((idx + 1) / len(queries))
            
            status_text.text("")
            progress_bar.empty()
            
            for q in queries:
                history.append_history({
                    "query": q,
                    "source": batch_source,
                    "sort": "relevance",
                    "batch": True
                })
            
            raw_count = len(aggregated)
            deduped = scraper.dedupe_results(aggregated)
            
            # Skip results already in local DB
            existing = load_papers_cached()
            existing_titles = {dedupe.normalize_title(p.get('title', '')) for p in existing if p.get('title')}
            existing_dois = {str(p.get('doi', '')).lower() for p in existing if p.get('doi') not in ['Unknown', '', None, 'None']}
            
            filtered = []
            skipped = 0
            for paper in deduped:
                doi = (paper.get('doi') or "").lower()
                title_norm = dedupe.normalize_title(paper.get('title', ''))
                if doi and doi in existing_dois:
                    skipped += 1
                    continue
                if title_norm and title_norm in existing_titles:
                    skipped += 1
                    continue
                filtered.append(paper)
                if len(filtered) >= batch_max_total:
                    break
            
            st.session_state['batch_results'] = filtered
            st.session_state['batch_stats'] = {
                'raw_count': raw_count,
                'deduped_count': len(deduped),
                'filtered_count': len(filtered),
                'skipped_existing': skipped
            }
            st.session_state['batch_seed'] = st.session_state.get('batch_seed', 0) + 1
            
            stats = st.session_state['batch_stats']
            st.info(
                f"Batch results: {stats['raw_count']} raw, "
                f"{stats['deduped_count']} deduped, "
                f"{stats['filtered_count']} ready, "
                f"{stats['skipped_existing']} skipped, already in database."
            )
            
            if batch_auto_ingest and filtered:
                progress_bar = st.progress(0)
                status_text = st.empty()
                successes = 0
                failures = 0
                
                for i, paper in enumerate(filtered):
                    status_text.text(f"Downloading and ingesting {i+1}/{len(filtered)}: {paper.get('title', 'Unknown')[:60]}")
                    pdf_path, _ = download_paper_for_ingest(paper)
                    
                    if pdf_path:
                        ingest.ingest_paper(pdf_path, paper)
                        successes += 1
                    else:
                        failures += 1
                    
                    progress_bar.progress((i + 1) / len(filtered))
                
                status_text.text("")
                st.success(f"Batch ingest complete. Added {successes} papers. Failed {failures}.")
                st.cache_data.clear()

    if not batch_auto_ingest and st.session_state.get('batch_results'):
        st.markdown("#### Batch Results")
        batch_seed = st.session_state.get('batch_seed', 0)
        select_all_batch = st.checkbox("Select all batch results", key=f"select_all_batch_{batch_seed}")
        if select_all_batch:
            for i in range(len(st.session_state['batch_results'])):
                st.session_state[f"batch_select_{batch_seed}_{i}"] = True
        
        batch_selected = []
        for i, paper in enumerate(st.session_state['batch_results']):
            title = format_title(paper)
            with st.expander(title):
                selected = st.checkbox("Select", key=f"batch_select_{batch_seed}_{i}")
                if selected:
                    batch_selected.append(i)
                st.caption(f"Authors: {format_authors(paper.get('authors'))}")
                st.caption(f"Published: {format_published(paper)}")
                st.write(paper.get('summary', 'No summary available.'))
            st.divider()
        
        if batch_selected:
            if st.button("⬇️ Bulk Download & Ingest Batch Selection"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                successes = 0
                failures = 0
                
                for idx, i in enumerate(batch_selected):
                    paper = st.session_state['batch_results'][i]
                    status_text.text(f"Downloading and ingesting {idx+1}/{len(batch_selected)}: {paper.get('title', 'Unknown')[:60]}")
                    pdf_path, _ = download_paper_for_ingest(paper)
                    
                    if pdf_path:
                        ingest.ingest_paper(pdf_path, paper)
                        successes += 1
                    else:
                        failures += 1
                    
                    progress_bar.progress((idx + 1) / len(batch_selected))
                
                status_text.text("")
                st.success(f"Batch ingest complete. Added {successes} papers. Failed {failures}.")
                st.cache_data.clear()
    st.markdown("---")
    st.subheader("Incremental Ingest data and PDFs")
    if st.button("Scan & Ingest New PDFs"):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, message):
            progress_bar.progress(current / total if total > 0 else 0)
            status_text.text(message)
        
        with st.spinner("Scanning local PDFs..."):
            result = ingest.ingest_directory(
                data_dir,
                resolve_metadata=True,
                progress_callback=update_progress
            )
        
        status_text.text("")
        progress_bar.empty()
        st.success(result.get("message", "Incremental ingest complete."))
        st.cache_data.clear()

    st.subheader("Upload Local PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"selected {len(uploaded_files)} files")
        
        if st.button(f"Ingest {len(uploaded_files)} Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} {i+1}/{len(uploaded_files)}...")
                
                # 1. Save File
                download_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                
                pdf_path = os.path.join(download_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Heuristic Title from filename
                raw_title = uploaded_file.name.replace(".pdf", "").replace("_", " ").replace("-", " ")
                
                # 3. Resolve Metadata
                resolved = scraper.resolve_paper_metadata(raw_title)
                
                final_title = raw_title
                final_authors = ["Unknown"]
                final_doi = "Unknown"
                final_summary = "Manual Upload"
                final_published = "Unknown"
                
                if resolved.get("found"):
                    final_title = resolved.get('title') or final_title
                    final_authors = resolved.get('authors') or final_authors
                    final_doi = resolved.get('doi', 'Unknown')
                    final_summary = resolved.get('summary', 'Manual Upload')
                    final_published = resolved.get('published', 'Unknown')
                
                # 4. Ingest
                metadata = {
                    "title": final_title,
                    "authors": final_authors,
                    "summary": final_summary,
                    "published": final_published,
                    "doi": final_doi,
                    "entry_id": "manual_" + uploaded_file.name
                }
                ingest.ingest_paper(pdf_path, metadata)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success(f"Successfully processed {len(uploaded_files)} files!")
            st.cache_data.clear()

# Metadata Enrichment Section
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Enrich Metadata")
st.sidebar.caption("Auto-fill missing DOI, authors, and year for papers with 'Unknown' metadata.")

# Add checkbox for force re-check
force_recheck = st.sidebar.checkbox("Force re-check all manual/local papers", value=False, help="Enable this to re-scan all manually uploaded papers, even if they already have metadata. Useful for fixing incorrect matches.")

if st.sidebar.button("🔄 Enrich Papers"):
    from src import enrich
    
    # Get count of papers needing enrichment
    papers_to_enrich = enrich.get_papers_needing_enrichment(force_recheck=force_recheck)
    
    if len(papers_to_enrich) == 0:
        st.sidebar.success("All papers already have complete metadata!")
    else:
        st.sidebar.info(f"Found {len(papers_to_enrich)} papers to process. Starting enrichment...")
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def update_progress(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)
        
        results = enrich.enrich_all_papers(force_recheck=force_recheck, progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        st.sidebar.success(f"✅ Done! Updated {results['success']}/{results['total']} papers.")
        
        if results['failed'] > 0:
            st.sidebar.warning(f"⚠️ {results['failed']} papers could not be matched.")
            with st.sidebar.expander("Details"):
                for detail in results['details']:
                    if not detail['success']:
                        st.caption(f"**{(detail.get('title') or 'Unknown')[:20]}...**: {detail.get('message', '')}")
        
        # Removed auto-rerun so user can read the details
        # time.sleep(2) 
        # st.rerun()

# Deduplication Section
st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Manage Library")
if st.sidebar.button("Find & Remove Duplicates"):
    from src import dedupe
    
    st.sidebar.info("Scanning for duplicates...")
    duplicates = dedupe.find_duplicates()
    
    if not duplicates:
        st.sidebar.success("No duplicates found!")
    else:
        st.sidebar.warning(f"Found {len(duplicates)} sets of duplicates.")
        
        total_removed = 0
        for group in duplicates:
            keep, removed = dedupe.merge_duplicates(group)
            total_removed += len(removed)
            # Log what happened
            st.sidebar.text(f"Kept: {(keep.get('title') or 'Unknown')[:20]}...")
            for items in removed:
                st.sidebar.text(f"  Deleted: {items[0][:20]}...")
        
        st.sidebar.success(f"Cleanup complete! Removed {total_removed} duplicate entries.")

# Database Management Section
st.sidebar.markdown("---")
with st.sidebar.expander("🗑️ Manage Database"):
    st.sidebar.caption("Fix metadata errors or delete unwanted papers.")

    # Get all current papers
    all_papers_dict = load_papers_cached()

    if not all_papers_dict:
        st.info("Database is empty.")
    else:
        # Create unique labels: "Title - ID ..."
        # We use a mapping from Label -> Paper Object
        paper_options = {}
        for p in all_papers_dict:
             # Use a short ID suffix to ensure uniqueness
             short_id = str(p.get('entry_id', 'unknown'))[-6:]
             label = f"{p.get('title', 'Untitled')} - ID ...{short_id}"
             paper_options[label] = p
         
        sorted_labels = sorted(paper_options.keys())
    
        selected_label = st.selectbox("Select paper to manage:", ["-- Select --"] + sorted_labels)
    
        if selected_label != "-- Select --":
            target_paper = paper_options[selected_label]
        
            st.code(f"Selected: {target_paper.get('title', 'Untitled')}\nID: {target_paper.get('entry_id')}\nSource: {target_paper.get('source')}")
        
            col_manage_1, col_manage_2 = st.columns(2)
        
            with col_manage_1:
                if st.button("✏️ Fix Metadata"):
                    with st.spinner("Re-analyzing paper..."):
                        # Force re-enrichment with robust logic
                        new_title, success, msg = enrich.enrich_single_paper(target_paper)
                        if success:
                            st.success(f"Fixed! New title: {new_title}")
                            time.sleep(1)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.warning(f"Could not improve metadata: {msg}")
        
            with col_manage_2:
                if st.button("❌ Delete"):
                    # Delete by Title is risky if duplicates exist, but currently database.py only supports it easily
                    # Improving safety by verifying
                    if database.delete_paper_by_title(target_paper.get('title', '')):
                        st.success("Deleted!")
                        time.sleep(1)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Failed.")
        
            st.subheader("🛠️ Edit Metadata")
            edit_title = st.text_input("Title", value=target_paper.get('title', ''))
            edit_authors = st.text_input("Authors comma separated", value=target_paper.get('authors', ''))
            edit_year = st.text_input("Year", value=str(target_paper.get('published', '')))
            edit_doi = st.text_input("DOI", value=str(target_paper.get('doi', '')))
            edit_entry_id = st.text_input("Entry ID optional", value=str(target_paper.get('entry_id', '')))
            edit_source = st.text_input("Source", value=str(target_paper.get('source', '')))
        
            if st.button("💾 Save Metadata"):
                authors_list = [a.strip() for a in str(edit_authors).split(",") if a.strip()]
                normalized_doi = normalize_doi(edit_doi)
                arxiv_id = extract_arxiv_id(edit_entry_id)
                openalex_id = extract_openalex_id(edit_entry_id)
            
                updated = {
                    "title": edit_title.strip() or target_paper.get('title'),
                    "authors": authors_list if authors_list else edit_authors,
                    "published": edit_year.strip() or target_paper.get('published'),
                    "doi": normalized_doi or edit_doi,
                    "entry_id": edit_entry_id.strip() or target_paper.get('entry_id'),
                    "source": edit_source.strip() or target_paper.get('source'),
                    "arxiv_id": arxiv_id,
                    "openalex_id": openalex_id
                }
                updated["canonical_id"] = compute_canonical_id(updated)
            
                updated_count = database.update_paper_metadata_by_title(target_paper.get('title', ''), updated)
                if updated_count > 0:
                    st.success(f"Updated {updated_count} chunks.")
                    time.sleep(1)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("No records updated.")
    
        st.markdown("---")
        if st.button("⚠️ RESET DATABASE DANGER"):
            database.drop_tables()
            st.success("Database wiped clean!")
            time.sleep(1)
            st.cache_data.clear()
            st.rerun()

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📚 Local Library", 
    "🔍 Semantic Search", 
    "✍️ Rewrite / Polish", 
    "💡 Idea Generation",
    "📝 Sentence Patterns",
    "🔬 Topic Clusters",
    "📎 Citation Finder",
    "📝 Literature Review",
    "📊 Citation Graph",
    "🧰 Diagnostics"
])

with tab1:
    st.header("Local Knowledge Base")
    st.caption("These are the papers currently analyzed and stored in your local vector database.")
    
    # Action buttons row
    col_refresh, col_export = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Library"):
            st.cache_data.clear()
    with col_export:
        if st.button("📥 Export All as BibTeX"):
            bibtex_content = bibtex.export_library_bibtex()
            st.download_button(
                label="💾 Download BibTeX File",
                data=bibtex_content,
                file_name="library.bib",
                mime="text/plain",
                key="download_bibtex"
            )
        if st.button("📥 Export All as JSON"):
            json_content = exports.export_library_json()
            st.download_button(
                label="💾 Download JSON",
                data=json_content,
                file_name="library.json",
                mime="application/json",
                key="download_json"
            )
        if st.button("📥 Export All as CSL-JSON"):
            csl_content = exports.export_library_csl_json()
            st.download_button(
                label="💾 Download CSL-JSON",
                data=csl_content,
                file_name="library.csl.json",
                mime="application/json",
                key="download_csl"
            )
        
    papers = load_papers_cached()
    
    if not papers:
        st.info("No papers found. Use the sidebar to add some!")
    else:
        st.metric("Total Papers", len(papers))
        
        # Quick filters
        years = []
        sources = set()
        for p in papers:
            sources.add(str(p.get("source", "Unknown")))
            published = p.get("published", "")
            match = re.search(r"(\d{4})", str(published))
            if match:
                years.append(int(match.group(1)))
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            if years:
                min_year = min(years)
                max_year = max(years)
                if min_year == max_year:
                    st.caption(f"Year Filter: {min_year}")
                    year_range = (min_year, max_year)
                else:
                    year_range = st.slider("Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
            else:
                year_range = None
                st.caption("Year Range: N/A")
        with col_f2:
            source_options = ["All"] + sorted([s for s in sources if s])
            source_filter = st.selectbox("Source", source_options)
        with col_f3:
            author_filter = st.text_input("Author Contains", value="")
        
        filtered_papers = []
        for p in papers:
            if year_range:
                published = p.get("published", "")
                match = re.search(r"(\d{4})", str(published))
                if match:
                    y = int(match.group(1))
                    if y < year_range[0] or y > year_range[1]:
                        continue
            if source_filter != "All" and str(p.get("source", "")) != source_filter:
                continue
            if author_filter:
                author_text = format_authors(p.get("authors", ""))
                if author_filter.lower() not in author_text.lower():
                    continue
            filtered_papers.append(p)
        
        st.caption(f"Showing {len(filtered_papers)} papers after filters.")
        
        st.markdown("#### 🧾 Paper Summary Cards")
        col_sum_1, col_sum_2 = st.columns([1, 1])
        with col_sum_1:
            if st.button("✨ Generate Missing Summaries"):
                llm_status = rag.llm_status()
                if not llm_status.get("available"):
                    st.warning("LLM not available. Set OPENAI_API_KEY or run Ollama to generate summaries.")
                    st.stop()
                missing = [p for p in filtered_papers if not p.get('summary') or str(p.get('summary')).strip() in ["", "Unknown", "None"]]
                if not missing:
                    st.info("All papers already have summaries.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    updated = 0
                    
                    for i, paper in enumerate(missing):
                        status_text.text(f"Summarizing {i+1}/{len(missing)}: {paper.get('title', 'Unknown')[:60]}")
                        summary_text = summaries.generate_paper_summary(paper.get('title', ''))
                        if summary_text:
                            database.update_paper_metadata_by_title(paper.get('title', ''), {"summary": summary_text})
                            updated += 1
                        progress_bar.progress((i + 1) / len(missing))
                    
                    status_text.text("")
                    st.success(f"Generated {updated} summaries.")
                    st.cache_data.clear()
                    st.rerun()
        
        # Display summary cards
        for idx, paper in enumerate(filtered_papers):
            target_col = col_sum_1 if idx % 2 == 0 else col_sum_2
            with target_col:
                summary_text = paper.get('summary') or "No summary available."
                with st.container():
                    st.markdown(f"**{format_title(paper)}**")
                    st.caption(f"Authors: {format_authors(paper.get('authors'))} | Year: {format_published(paper)}")
                    st.write(summary_text)
                    st.markdown("---")
        
        # Display as a table or cards
        for idx, paper in enumerate(filtered_papers):
            with st.expander(format_title(paper)):
                st.write(f"Authors: {format_authors(paper.get('authors'))}")
                st.write(f"Published: {format_published(paper)}")
                st.write(f"Source: {paper.get('source', 'Unknown')}")
                st.write(f"Summary: {paper.get('summary', 'No summary available.')}")
                st.write(f"DOI: {paper.get('doi', 'N/A')}")
                st.write(f"Vector Chunks: {paper['chunk_count']}")
                st.caption(f"Entry ID: {paper['entry_id']}")
                
                # Individual BibTeX copy
                paper_bibtex = bibtex.generate_bibtex_entry(paper)
                st.code(paper_bibtex, language="bibtex")
            st.divider()

with tab2:
    st.header("Search Literature")
    query = st.text_input("Enter a concept or question:", placeholder="e.g., Optimal mechanism for multi-item auctions")
    if query:
        results = rag.query_db(query, n_results=5)
        
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            score = results['distances'][0][i]
            
            with st.container():
                st.markdown(f"**Source:** {meta.get('title', 'Unknown')} — Chunk {meta.get('chunk_index')}")
                st.info(doc)
                st.divider()

with tab3:
    st.header("Academic Sentence Polisher")
    sentence = st.text_area("Draft Sentence:", placeholder="I want to say that the agent will lie if the incentive is not good.")
    
    col1, col2 = st.columns(2)
    with col1:
        use_context = st.checkbox("Use Retrieval Context?", value=True)
    with col2:
        style_choice = st.selectbox("Style", ["Journal", "Working Paper", "Grant"], index=0)
        
    if st.button("Rewrite"):
        context = ""
        if use_context:
            with st.spinner("Retrieving context..."):
                results = rag.query_db(sentence, n_results=3)
                context = rag.format_context(results)
                st.subheader("Retrieved Context")
                st.write(context)
        
        with st.spinner("Polishing..."):
            rewritten = rag.rewrite_sentence(sentence, context, style=style_choice.lower())
            st.markdown("### Suggestion:")
            st.success(rewritten)

with tab4:
    st.header("Research Idea Generator")
    topic = st.text_input("Research Topic:", placeholder="Information Design in Networks")
    structured_output = st.checkbox("Structured Output", value=True)
    
    if st.button("Generate Ideas"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Analyzing literature and thinking..."):
            status_text.text("Retrieving relevant context...")
            results = rag.query_db(topic, n_results=10)
            context = rag.format_context(results)
            progress_bar.progress(0.5)
            
            status_text.text("Generating ideas...")
            ideas = rag.generate_ideas(topic, context, structured=structured_output)
            progress_bar.progress(1.0)
            
            status_text.text("")
            progress_bar.empty()
            st.markdown(ideas)

# ============== NEW TABS ==============

with tab5:
    st.header("📝 Academic Sentence Patterns")
    st.caption("Discover common academic expressions and sentence structures from your paper library.")
    
    col_build, col_info = st.columns([1, 2])
    with col_build:
        if st.button("🔨 Build/Refresh Pattern Library"):
            with st.spinner("Analyzing papers for patterns... This may take a minute."):
                library = patterns.build_pattern_library()
                st.success(f"Built library with {len(library.get('patterns', {}))} patterns!")
                st.session_state['pattern_library'] = library
    
    # Load existing library
    if 'pattern_library' not in st.session_state:
        st.session_state['pattern_library'] = patterns.load_pattern_library()
    
    library = st.session_state['pattern_library']
    
    if len(library.get('patterns', {})) == 0:
        st.info("No pattern library found. Click 'Build/Refresh Pattern Library' to analyze your papers.")
    else:
        st.metric("Total Unique Patterns", len(library.get('patterns', {})))
        
        # Search patterns
        search_query = st.text_input("Search patterns:", placeholder="e.g., 'we show' or category like 'Argumentation'")
        
        if search_query:
            results = patterns.search_patterns(search_query, library)
            st.write(f"Found {len(results)} matching patterns:")
            for pattern_text, data in results:
                variants = expand_pattern_variants(pattern_text)
                header = variants[0] if variants else clean_pattern_display(pattern_text)
                with st.expander(f"{data['category']} — {header}"):
                    for variant in variants:
                        st.write(f"{variant} — {data['count']} uses")
                        if data.get('examples'):
                            for ex in data.get('examples', [])[:2]:
                                st.info(f"_{ex['sentence']}_")
                                st.caption(f"Source: {ex['source']}")
        else:
            # Show patterns grouped by category with expanders
            categories = patterns.get_patterns_by_category(library)
            for category, cat_patterns in categories.items():
                with st.expander(f"{category} — {len(cat_patterns)} patterns"):
                    for p in cat_patterns:
                        variants = expand_pattern_variants(p['text'])
                        for variant in variants:
                            st.write(f"{variant} — {p['count']} uses")
                            if p.get('examples'):
                                for ex in p.get('examples', [])[:2]:
                                    st.info(f"_{ex['sentence']}_")
                                    st.caption(f"Source: {ex['source']}")
                        st.divider()
                            
            st.markdown("---")
            st.subheader("🛡️ Structural Analysis Dynamic")
            st.caption("Common sentence templates discovered automatically from your papers.")
            
            struct_patterns = library.get('structural_patterns', {})
            if struct_patterns:
                grouped = {}
                for template, data in struct_patterns.items():
                    cat = map_struct_category("", template)
                    grouped.setdefault(cat, []).append((template, data))
                
                cat_order = ["INTRO", "MODEL", "ARGUMENT", "LOGIC", "RESULT", "ECON", "DISC", "LIT", "META", "Other"]
                ordered = [c for c in cat_order if c in grouped]
                ordered += sorted([c for c in grouped.keys() if c not in ordered])
                
                for cat in ordered:
                    with st.expander(f"{cat} — {len(grouped[cat])} types"):
                        # Sort by count desc
                        for template, data in sorted(grouped[cat], key=lambda x: x[1].get('count', 0), reverse=True):
                            match = re.match(r"^\[(\w+)\][, ]+(.+)", template)
                            display_template = match.group(2) if match else template
                            display_template = clean_pattern_display(display_template)
                            variants = expand_pattern_variants(display_template) if display_template else []
                            shown_variant = False
                            for variant in variants:
                                if variant.isupper():
                                    continue
                                st.write(f"{variant} — {data['count']} matches")
                                examples = data.get('examples', [])
                                if examples:
                                    for ex in examples[:2]:
                                        st.info(f"_{ex['sentence']}_")
                                        st.caption(f"Source: {ex['source']}")
                                shown_variant = True
                            if not shown_variant:
                                st.write(f"{cat} — {data['count']} matches")
                                examples = data.get('examples', [])
                                if examples:
                                    for ex in examples[:2]:
                                        st.info(f"_{ex['sentence']}_")
                                        st.caption(f"Source: {ex['source']}")
                            st.divider()
            else:
                st.info("No structural patterns found yet. Click 'Build/Refresh Pattern Library' above to analyze.")
with tab6:
    st.header("🔬 Topic Clusters")
    st.caption("Visualize how your papers cluster by research topic using embeddings.")
    
    col_cluster, col_label = st.columns([1, 1])
    with col_cluster:
        if st.button("🔄 Generate Clusters"):
            with st.spinner("Computing clusters... This may take a moment."):
                cluster_result = clustering.cluster_papers()
                st.session_state['cluster_result'] = cluster_result
                
                if "error" not in cluster_result:
                    st.success(f"Found {cluster_result['n_clusters']} clusters in {cluster_result['paper_count']} papers!")
                else:
                    st.error(cluster_result['error'])
    
    with col_label:
        if st.button("🏷️ Generate Cluster Labels LLM"):
            if 'cluster_result' in st.session_state:
                with st.spinner("Generating labels with LLM..."):
                    labels = clustering.label_clusters_with_llm(st.session_state['cluster_result'])
                    st.session_state['cluster_labels'] = labels
                    st.success("Labels generated!")
            else:
                st.warning("Generate clusters first!")
    
    if 'cluster_result' in st.session_state:
        cluster_result = st.session_state['cluster_result']
        cluster_labels = st.session_state.get('cluster_labels', None)
        
        if "error" not in cluster_result:
            # Display interactive plot
            fig = clustering.plot_clusters(cluster_result, cluster_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cluster summary
            st.subheader("Cluster Details")
            st.markdown(clustering.get_cluster_summary(cluster_result))
        else:
            st.error(cluster_result['error'])
    else:
        st.info("Click 'Generate Clusters' to analyze your paper collection.")

with tab7:
    st.header("📎 Citation Finder")
    st.caption("Find relevant papers to cite based on your text. Paste a paragraph and get citation suggestions.")
    
    input_text = st.text_area(
        "Your text (paragraph or sentence):",
        placeholder="When designing mechanisms for multi-agent settings, the principal must account for strategic behavior...",
        height=150
    )
    
    n_citations = st.slider("Number of suggestions:", min_value=1, max_value=10, value=5)
    
    if st.button("🔍 Find Citations"):
        if input_text.strip():
            with st.spinner("Finding relevant papers..."):
                recommendations = citations.recommend_citations(input_text, n_results=n_citations)
                
                if not recommendations:
                    st.warning("No relevant papers found. Try adding more papers to your library.")
                else:
                    st.success(f"Found {len(recommendations)} relevant papers:")
                    
                    for i, paper in enumerate(recommendations, 1):
                        score_pct = paper['similarity'] * 100
                        
                        st.markdown(f"**{i}. {format_title(paper)}** - Relevance {score_pct:.1f}%")
                        st.write(f"Authors: {format_authors(paper.get('authors'))}")
                        st.write(f"Published: {format_published(paper)}")
                        
                        # Show BibTeX
                        paper_bibtex = bibtex.generate_bibtex_entry(paper)
                        st.code(paper_bibtex, language="bibtex")
                        st.divider()
        else:
            st.warning("Please enter some text to find citations for.")

with tab8:
    st.header("📝 Literature Review Generator")
    st.caption("Generate a comparative literature review for selected papers.")
    
    # Lazy import to avoid circular import issues
    import src.review as review
    
    # Get all papers for selection
    all_papers = load_papers_cached()
    paper_titles = [p.get('title') for p in all_papers if p.get('title')]
    paper_titles.sort()
    
    selected_papers = st.multiselect(
        "Select papers to review 2-10 recommended:",
        options=paper_titles,
        max_selections=10
    )
    
    col_review_1, col_review_2 = st.columns(2)
    with col_review_1:
        use_agentic = st.checkbox("🚀 Use Agentic Workflow Deeper Analysis", value=False)
    with col_review_2:
        deep_topic = st.text_input("Deep Review Focus optional:", placeholder="e.g. Robustness in auctions")

    if st.button("Generate Review"):
        if not selected_papers and not use_agentic:
            st.warning("Please select at least one paper or enable Agentic Workflow.")
        elif use_agentic and not deep_topic and not selected_papers:
            st.warning("Please provide a topic or select papers for deep review.")
        else:
            with st.spinner(f"Running Agentic Workflow..." if use_agentic else f"Reading {len(selected_papers)} papers and generating review..."):
                if use_agentic:
                    review_text = review.generate_deep_literature_review(selected_papers, deep_topic)
                else:
                    review_text = review.generate_literature_review(selected_papers)
                
                st.markdown(review_text)
                
                # Download button for the review
                st.download_button(
                    label="💾 Download Review",
                    data=review_text,
                    file_name="literature_review.md",
                    mime="text/markdown"
                )

with tab9:
    st.header("📊 Citation Graph")
    st.caption("Visualize citation relationships between papers in your library.")
    
    # Lazy import
    from src import citation_graph
    
    st.info("""
    This feature builds a citation network by:
    1. Looking up each paper in Semantic Scholar
    2. Finding citation links between your library papers
    3. Displaying an interactive graph
    
    ⚠️ Note: Only papers found in Semantic Scholar will appear. This may take a few minutes for large libraries.
    """)
    
    if st.button("🔄 Build Citation Graph"):
        all_papers = load_papers_cached()
        
        if len(all_papers) == 0:
            st.warning("No papers in library. Add some papers first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress_bar.progress(current / total if total > 0 else 0)
                status_text.text(f"{message} {current}/{total}")
            
            with st.spinner("Building citation graph..."):
                G, paper_info = citation_graph.build_citation_graph(
                    all_papers, 
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Found {len(G.nodes())} papers with {len(G.edges())} citation links.")
                
                # Display the graph
                fig = citation_graph.graph_to_plotly(G, paper_info)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                if len(G.nodes()) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers Found", len(G.nodes()))
                    with col2:
                        st.metric("Citation Links", len(G.edges()))
                    with col3:
                        density = len(G.edges()) / (len(G.nodes()) * (len(G.nodes()) - 1)) if len(G.nodes()) > 1 else 0
                        st.metric("Graph Density", f"{density:.2%}")

with tab10:
    st.header("🧰 Diagnostics & Tuning")
    st.caption("Startup checks, dependency status, and retrieval tuning.")
    
    st.subheader("Startup Checks")
    if st.button("Run Startup Checks"):
        report = diagnostics.run_startup_checks()
        for item in report.get("checks", []):
            name = item.get("name", "Unknown")
            status = item.get("status", "unknown")
            detail = item.get("detail", "")
            if status == "ok":
                st.success(f"{name}: {detail}")
            elif status == "warning":
                st.warning(f"{name}: {detail}")
            else:
                st.error(f"{name}: {detail}")
    
    st.subheader("Retrieval Tuning Runtime")
    defaults = diagnostics.get_retrieval_defaults()
    mode = st.selectbox("Retrieval Mode", ["vector", "bm25", "hybrid"], index=["vector","bm25","hybrid"].index(defaults["mode"]))
    alpha = st.slider("Hybrid Alpha vector weight", min_value=0.0, max_value=1.0, value=float(defaults["alpha"]), step=0.05)
    reranker_model = st.text_input("Reranker Model optional", value=defaults["reranker_model"])
    reranker_top_k = st.number_input("Reranker Top-K", min_value=1, max_value=100, value=int(defaults["reranker_top_k"]))
    candidate_multiplier = st.number_input("Hybrid Candidate Multiplier", min_value=1, max_value=10, value=int(defaults["candidate_multiplier"]))
    
    if st.button("Apply Retrieval Settings"):
        retrieval.set_overrides(
            mode=mode,
            alpha=alpha,
            reranker_model=reranker_model.strip(),
            reranker_top_k=int(reranker_top_k),
            candidate_multiplier=int(candidate_multiplier)
        )
        st.success("Applied runtime retrieval settings for this session.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Local LLMs & ChromaDB")
