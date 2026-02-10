import streamlit as st
import os
import sys
import time
import re
from datetime import date, timedelta

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import scraper, ingest, rag, retrieval, search_utils, task_queue, watchlist
from src import bibtex, citations, patterns, clustering, database, dedupe, exports, summaries, history, diagnostics
from src.metadata_utils import compute_canonical_id, normalize_doi, extract_arxiv_id, extract_openalex_id
# review is imported lazily inside its tab to avoid circular import issues

st.set_page_config(page_title="Local Academic Assistant", layout="wide")

st.title("📚 Local Academic Literature Assistant")

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


def parse_tags(tags_value) -> list:
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        raw = [str(t).strip() for t in tags_value if str(t).strip()]
    else:
        raw = [t.strip() for t in re.split(r"[;,]", str(tags_value)) if t.strip()]
    seen = set()
    tags = []
    for t in raw:
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        tags.append(t)
    return tags


def format_tags(tags_value) -> str:
    tags = parse_tags(tags_value)
    return ", ".join(tags)


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


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


queue_manager = task_queue.get_queue()


def _init_llm_settings_state():
    if "llm_setting_backend_pref" not in st.session_state:
        st.session_state["llm_setting_backend_pref"] = os.getenv("ARXIV_ASSISTANT_LLM_BACKEND", "auto")
    if "llm_setting_openai_key" not in st.session_state:
        # Keep runtime key session-only; do not mirror environment key in UI.
        st.session_state["llm_setting_openai_key"] = ""
    if "llm_setting_ollama_host" not in st.session_state:
        st.session_state["llm_setting_ollama_host"] = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
    if "llm_setting_openai_model" not in st.session_state:
        st.session_state["llm_setting_openai_model"] = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    if "llm_setting_ollama_model" not in st.session_state:
        st.session_state["llm_setting_ollama_model"] = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "llama3.2"))


def _apply_runtime_llm_settings(force: bool = False) -> dict:
    signature = (
        str(st.session_state.get("llm_setting_backend_pref", "auto")),
        str(st.session_state.get("llm_setting_openai_key", "")),
        str(st.session_state.get("llm_setting_ollama_host", "")),
        str(st.session_state.get("llm_setting_openai_model", "")),
        str(st.session_state.get("llm_setting_ollama_model", "")),
    )
    if (not force) and st.session_state.get("_llm_runtime_signature") == signature:
        return rag.llm_status()

    status = rag.configure_llm(
        openai_api_key=(st.session_state.get("llm_setting_openai_key", "").strip() or None),
        ollama_host=st.session_state.get("llm_setting_ollama_host", ""),
        backend_preference=st.session_state.get("llm_setting_backend_pref", "auto"),
        openai_model=st.session_state.get("llm_setting_openai_model", ""),
        ollama_model=st.session_state.get("llm_setting_ollama_model", ""),
    )
    st.session_state["_llm_runtime_signature"] = signature
    return status


_init_llm_settings_state()
_apply_runtime_llm_settings()

# Sidebar: Manage Database
with st.sidebar:
    st.header("Manage Database")

    st.subheader("LLM Settings")
    with st.expander("Configure API / Backend", expanded=False):
        current_llm_status = rag.llm_status()
        if current_llm_status.get("available"):
            st.caption(
                f"Current: `{current_llm_status.get('backend')}` / "
                f"`{current_llm_status.get('model')}`"
            )
        else:
            st.caption("Current: unavailable")

        with st.form("llm_settings_form"):
            st.selectbox(
                "Backend Preference",
                options=["auto", "openai", "ollama"],
                key="llm_setting_backend_pref",
                help="auto: OpenAI first when key exists, otherwise fallback to Ollama.",
            )
            st.text_input(
                "OpenAI API Key optional",
                key="llm_setting_openai_key",
                type="password",
                help="Stored only in current app session.",
            )
            st.text_input("OpenAI Model", key="llm_setting_openai_model")
            st.text_input("Ollama Host", key="llm_setting_ollama_host")
            st.text_input("Ollama Model", key="llm_setting_ollama_model")
            apply_llm = st.form_submit_button("Apply LLM Settings")

        if apply_llm:
            current_llm_status = _apply_runtime_llm_settings(force=True)
            if current_llm_status.get("available"):
                st.success(
                    f"LLM ready: {current_llm_status.get('backend')} "
                    f"({current_llm_status.get('model')})"
                )
            else:
                st.warning("LLM unavailable. Check API key, backend choice, or Ollama host.")

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
        use_bg_queue = st.checkbox("Use background queue for ingest", value=True, key="search_use_bg_queue")
        auto_enrich_bg = st.checkbox("Background tasks run metadata enrichment", value=True, key="search_auto_enrich_bg")

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
                if paper.get("venue"):
                    st.caption(f"Venue: {paper.get('venue')}")
                st.write(paper.get('summary', 'No summary available.'))
                if st.button(f"Download and Ingest {i}", key=f"btn_{i}"):
                    if use_bg_queue:
                        task_id = queue_manager.enqueue_ingest_from_paper(paper, run_enrichment=auto_enrich_bg)
                        st.success(f"Queued as background task: {task_id[:8]}")
                    else:
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
                if use_bg_queue:
                    queued = 0
                    for i in selected_indices:
                        paper = st.session_state['search_results'][i]
                        queue_manager.enqueue_ingest_from_paper(paper, run_enrichment=auto_enrich_bg)
                        queued += 1
                    st.success(f"Queued {queued} background tasks.")
                else:
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
                queued = 0
                for paper in filtered:
                    queue_manager.enqueue_ingest_from_paper(paper, run_enrichment=True)
                    queued += 1
                st.success(f"Batch ingest queued: {queued} background tasks.")

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
                if paper.get("venue"):
                    st.caption(f"Venue: {paper.get('venue')}")
                st.write(paper.get('summary', 'No summary available.'))
            st.divider()
        
        if batch_selected:
            if st.button("⬇️ Bulk Download & Ingest Batch Selection"):
                queued = 0
                for i in batch_selected:
                    paper = st.session_state['batch_results'][i]
                    queue_manager.enqueue_ingest_from_paper(paper, run_enrichment=True)
                    queued += 1
                st.success(f"Batch selection queued: {queued} background tasks.")
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
    upload_bg_queue = st.checkbox("Queue uploaded PDFs in background", value=True, key="upload_bg_queue")
    
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
                final_venue = ""
                
                if resolved.get("found"):
                    final_title = resolved.get('title') or final_title
                    final_authors = resolved.get('authors') or final_authors
                    final_doi = resolved.get('doi', 'Unknown')
                    final_summary = resolved.get('summary', 'Manual Upload')
                    final_published = resolved.get('published', 'Unknown')
                    final_venue = resolved.get('venue') or final_venue
                
                # 4. Ingest
                metadata = {
                    "title": final_title,
                    "authors": final_authors,
                    "summary": final_summary,
                    "published": final_published,
                    "doi": final_doi,
                    "venue": final_venue,
                    "entry_id": "manual_" + uploaded_file.name,
                    "tags": "",
                    "is_favorite": False,
                    "in_reading_list": False
                }
                if upload_bg_queue:
                    queue_manager.enqueue_local_file(pdf_path, metadata, run_enrichment=True)
                else:
                    ingest.ingest_paper(pdf_path, metadata)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if upload_bg_queue:
                status_text.success(f"Queued {len(uploaded_files)} uploaded files.")
            else:
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
dedupe_threshold = st.sidebar.slider("Duplicate sensitivity", min_value=0.75, max_value=0.95, value=0.84, step=0.01)
if st.sidebar.button("Find Duplicate Conflicts"):
    from src import dedupe

    st.sidebar.info("Scanning for duplicate conflicts...")
    st.session_state["duplicate_groups"] = dedupe.find_duplicates(min_score=dedupe_threshold)
    st.session_state["duplicate_threshold"] = dedupe_threshold

groups = st.session_state.get("duplicate_groups", [])
if groups:
    st.sidebar.warning(f"Found {len(groups)} duplicate groups.")
    with st.sidebar.expander("Review & Merge Conflicts", expanded=False):
        for idx, group in enumerate(groups):
            titles = [str(p.get("title", "Unknown"))[:28] for p in group]
            label = f"Group {idx + 1} ({len(group)}): {titles[0]}"
            with st.expander(label, expanded=False):
                from src import dedupe
                merged = dedupe.propose_merged_metadata(group)
                st.caption("Proposed merged metadata:")
                st.write({
                    "title": merged.get("title"),
                    "doi": merged.get("doi"),
                    "venue": merged.get("venue"),
                    "published": merged.get("published"),
                    "tags": merged.get("tags"),
                })
                for j, p in enumerate(group):
                    st.caption(
                        f"{j + 1}. {p.get('title', 'Unknown')} | DOI={p.get('doi', '')} | "
                        f"Year={p.get('published', '')} | Entry={p.get('entry_id', '')}"
                    )
                if st.button("Merge This Group", key=f"merge_dup_group_{idx}"):
                    result = dedupe.merge_duplicates_group(group)
                    removed = result.get("removed", [])
                    st.success(f"Merged group {idx + 1}. Removed {len([r for r in removed if r[1]])} entries.")
                    # Refresh conflict list after merge
                    st.session_state["duplicate_groups"] = dedupe.find_duplicates(
                        min_score=st.session_state.get("duplicate_threshold", dedupe_threshold)
                    )
                    st.cache_data.clear()
                    st.rerun()
else:
    if "duplicate_groups" in st.session_state:
        st.sidebar.success("No duplicate conflicts found.")

# Background Task Queue Section
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Background Tasks")
if st.sidebar.button("Refresh Task Status"):
    st.rerun()

task_summary = queue_manager.get_summary()
col_tq_1, col_tq_2 = st.sidebar.columns(2)
with col_tq_1:
    st.metric("Pending", task_summary.get("pending", 0))
with col_tq_2:
    st.metric("Running", task_summary.get("running", 0))
col_tq_3, col_tq_4 = st.sidebar.columns(2)
with col_tq_3:
    st.metric("Done", task_summary.get("completed", 0))
with col_tq_4:
    st.metric("Failed", task_summary.get("failed", 0))
st.sidebar.caption(f"Cancelled: {task_summary.get('cancelled', 0)}")

with st.sidebar.expander("Queue Operations", expanded=False):
    include_cancelled_retry = st.checkbox(
        "Retry cancelled too",
        value=False,
        key="queue_include_cancelled_retry",
        help="If enabled, bulk retry includes cancelled tasks in addition to failed tasks.",
    )
    keep_latest_terminal = st.number_input(
        "Keep latest terminal tasks",
        min_value=10,
        max_value=5000,
        value=200,
        step=10,
        key="queue_keep_latest_terminal",
        help="Older completed/failed/cancelled tasks are pruned first.",
    )
    col_q_1, col_q_2 = st.columns(2)
    with col_q_1:
        if st.button("Cancel Pending", key="queue_cancel_pending_bulk"):
            cancelled = queue_manager.cancel_all_pending()
            if cancelled > 0:
                st.success(f"Cancelled {cancelled} pending tasks.")
            else:
                st.info("No pending tasks to cancel.")
            st.rerun()
    with col_q_2:
        if st.button("Retry Failed", key="queue_retry_failed_bulk"):
            retried = queue_manager.retry_all_failed(include_cancelled=include_cancelled_retry)
            if retried > 0:
                st.success(f"Queued {retried} retry tasks.")
            else:
                st.info("No eligible tasks to retry.")
            st.rerun()
    if st.button("Prune Task History", key="queue_prune_terminal_bulk"):
        removed = queue_manager.prune_terminal(keep_latest=int(keep_latest_terminal))
        if removed > 0:
            st.success(f"Pruned {removed} terminal tasks.")
        else:
            st.info("No terminal tasks were pruned.")
        st.rerun()

with st.sidebar.expander("Task Details", expanded=False):
    task_rows = queue_manager.list_tasks(limit=30)
    if not task_rows:
        st.caption("No tasks queued yet.")
    else:
        for idx, t in enumerate(task_rows):
            tid = t.get("id", "")
            status = t.get("status", "")
            msg = t.get("message", "")
            st.caption(f"{status.upper()} | {tid[:8]} | {msg}")
            col_a, col_b = st.columns(2)
            with col_a:
                if status in ["pending", "running"]:
                    if st.button("Cancel", key=f"cancel_task_{idx}_{tid[:8]}"):
                        queue_manager.cancel_task(tid)
                        st.rerun()
            with col_b:
                if status in ["failed", "cancelled"]:
                    if st.button("Retry", key=f"retry_task_{idx}_{tid[:8]}"):
                        queue_manager.retry_task(tid)
                        st.rerun()

# Watchlist & Digest Section
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Watchlist")
due_count = watchlist.due_watch_count()
if due_count > 0:
    st.sidebar.warning(f"{due_count} watch items are due. Run digest now.")
watch_auto_enqueue = st.sidebar.checkbox(
    "Auto-enqueue new papers",
    value=True,
    key="watch_auto_enqueue_new",
    help="When running due watches, queue new papers for background ingest.",
)
watch_auto_enrich = st.sidebar.checkbox(
    "Enqueue with enrichment",
    value=True,
    key="watch_auto_enrich_new",
    help="Apply metadata enrichment in background ingest tasks created by watchlist runs.",
)
watch_auto_run_due = st.sidebar.checkbox(
    "Auto-run due watches on refresh",
    value=False,
    key="watch_auto_run_due",
    help="Automatically trigger due watch runs once when this page refreshes.",
)
if due_count == 0:
    st.session_state["watch_auto_run_fired"] = False
if not watch_auto_run_due:
    st.session_state["watch_auto_run_fired"] = False
if watch_auto_run_due and due_count > 0 and not st.session_state.get("watch_auto_run_fired", False):
    auto_result = watchlist.run_due_watches(
        max_results_per_watch=8,
        enqueue_new=watch_auto_enqueue,
        enqueue_fn=queue_manager.enqueue_ingest_from_paper if watch_auto_enqueue else None,
        run_enrichment=watch_auto_enrich,
    )
    st.session_state["watch_auto_run_fired"] = True
    st.session_state["watch_last_result"] = auto_result
    st.rerun()

watch_query = st.sidebar.text_input("Watch query", key="watch_query_input", placeholder="e.g., mechanism design")
col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    watch_type = st.selectbox("Type", ["keyword", "author"], index=0, key="watch_type_select")
with col_w2:
    watch_frequency = st.selectbox("Frequency", ["daily", "weekly"], index=0, key="watch_freq_select")
watch_source = st.sidebar.selectbox("Source", ["arxiv", "semantic_scholar"], index=0, key="watch_source_select")
if st.sidebar.button("Add Watch"):
    if not watch_query.strip():
        st.sidebar.warning("Watch query is required.")
    else:
        wid = watchlist.add_watch(
            watch_query.strip(),
            watch_type=watch_type,
            source=watch_source,
            frequency=watch_frequency,
        )
        st.sidebar.success(f"Watch added: {wid[:8]}")
        st.rerun()

if st.sidebar.button("Run Due Watches"):
    result = watchlist.run_due_watches(
        max_results_per_watch=8,
        enqueue_new=watch_auto_enqueue,
        enqueue_fn=queue_manager.enqueue_ingest_from_paper if watch_auto_enqueue else None,
        run_enrichment=watch_auto_enrich,
    )
    st.session_state["watch_last_result"] = result
    st.sidebar.success(
        f"Watches run: {result.get('ran', 0)} | new papers: {result.get('new_papers', 0)} | queued: {result.get('queued_tasks', 0)}"
    )
    if result.get("digest_path"):
        st.sidebar.caption(f"Digest: {result.get('digest_path')}")

last_watch_run = st.session_state.get("watch_last_result")
if last_watch_run:
    st.sidebar.caption(
        f"Last run summary: ran={last_watch_run.get('ran', 0)}, "
        f"new={last_watch_run.get('new_papers', 0)}, queued={last_watch_run.get('queued_tasks', 0)}"
    )

with st.sidebar.expander("Watch Items", expanded=False):
    watches = watchlist.list_watches()
    if not watches:
        st.caption("No watch entries.")
    else:
        for i, w in enumerate(watches):
            st.caption(
                f"{w.get('query')} | {w.get('source')} | {w.get('frequency')} | "
                f"next={w.get('next_run', '')[:10]}"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                label = "Disable" if w.get("enabled", True) else "Enable"
                if st.button(label, key=f"toggle_watch_{i}_{w.get('id', '')[:8]}"):
                    watchlist.toggle_watch(w.get("id"), not w.get("enabled", True))
                    st.rerun()
            with col_b:
                if st.button("Delete", key=f"delete_watch_{i}_{w.get('id', '')[:8]}"):
                    watchlist.remove_watch(w.get("id"))
                    st.rerun()

with st.sidebar.expander("Latest Digest", expanded=False):
    digest_path = watchlist.get_latest_digest_path()
    if digest_path and os.path.exists(digest_path):
        st.caption(digest_path)
        try:
            with open(digest_path, "r", encoding="utf-8") as f:
                digest_text = f.read()
            st.text_area("Digest", value=digest_text, height=220, key="latest_digest_text")
        except Exception as e:
            st.warning(f"Failed to read digest: {e}")
    else:
        st.caption("No digest generated yet.")

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
            edit_venue = st.text_input("Venue", value=str(target_paper.get('venue', '')))
            edit_tags = st.text_input("Tags (comma separated)", value=format_tags(target_paper.get('tags', '')))
            edit_favorite = st.checkbox("Favorite", value=coerce_bool(target_paper.get('is_favorite')))
            edit_reading_list = st.checkbox("Reading list", value=coerce_bool(target_paper.get('in_reading_list')))
        
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
                    "venue": edit_venue.strip() or target_paper.get('venue', ''),
                    "entry_id": edit_entry_id.strip() or target_paper.get('entry_id'),
                    "source": edit_source.strip() or target_paper.get('source'),
                    "tags": edit_tags,
                    "is_favorite": edit_favorite,
                    "in_reading_list": edit_reading_list,
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
        if st.button("📥 Export All as RIS"):
            ris_content = exports.export_library_ris()
            st.download_button(
                label="💾 Download RIS",
                data=ris_content,
                file_name="library.ris",
                mime="application/x-research-info-systems",
                key="download_ris"
            )
        if st.button("📥 Export All as Zotero JSON"):
            zotero_content = exports.export_library_zotero_json()
            st.download_button(
                label="💾 Download Zotero JSON",
                data=zotero_content,
                file_name="library.zotero.json",
                mime="application/json",
                key="download_zotero_json"
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

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            tag_filter = st.text_input("Tag Contains", value="")
        with col_t2:
            favorites_only = st.checkbox("Favorites only", value=False)
        with col_t3:
            reading_list_only = st.checkbox("Reading list only", value=False)
        
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
            if tag_filter:
                tags_text = format_tags(p.get("tags", ""))
                if tag_filter.lower() not in tags_text.lower():
                    continue
            if favorites_only and not coerce_bool(p.get("is_favorite")):
                continue
            if reading_list_only and not coerce_bool(p.get("in_reading_list")):
                continue
            filtered_papers.append(p)
        
        st.caption(f"Showing {len(filtered_papers)} papers after filters.")
        
        st.markdown("#### 🧾 Paper Summary Cards")
        col_sum_1, col_sum_2 = st.columns([1, 1])
        with col_sum_1:
            if st.button("✨ Generate Missing Summaries"):
                llm_status = rag.llm_status()
                if not llm_status.get("available"):
                    st.warning(
                        "LLM not available. Configure in sidebar LLM Settings, "
                        "set OPENAI_API_KEY, or run Ollama."
                    )
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
                    if paper.get("venue"):
                        st.caption(f"Venue: {paper.get('venue')}")
                    tags_text = format_tags(paper.get("tags", ""))
                    if tags_text:
                        st.caption(f"Tags: {tags_text}")
                    if coerce_bool(paper.get("is_favorite")):
                        st.caption("Favorite: Yes")
                    if coerce_bool(paper.get("in_reading_list")):
                        st.caption("Reading list: Yes")
                    st.write(summary_text)
                    st.markdown("---")
        
        # Display as a table or cards
        for idx, paper in enumerate(filtered_papers):
            with st.expander(format_title(paper)):
                st.write(f"Authors: {format_authors(paper.get('authors'))}")
                st.write(f"Published: {format_published(paper)}")
                st.write(f"Venue: {paper.get('venue', 'N/A')}")
                st.write(f"Tags: {format_tags(paper.get('tags', '')) or 'N/A'}")
                st.write(f"Favorite: {'Yes' if coerce_bool(paper.get('is_favorite')) else 'No'}")
                st.write(f"Reading list: {'Yes' if coerce_bool(paper.get('in_reading_list')) else 'No'}")
                st.write(f"Source: {paper.get('source', 'Unknown')}")
                st.write(f"Summary: {paper.get('summary', 'No summary available.')}")
                st.write(f"DOI: {paper.get('doi', 'N/A')}")
                st.write(f"Vector Chunks: {paper['chunk_count']}")
                parse_score = float(paper.get("parse_quality_score", 0.0) or 0.0)
                parse_label = str(paper.get("parse_quality_label", "unknown"))
                parse_pages = int(paper.get("parse_pages", 0) or 0)
                parse_ocr_pages = int(paper.get("parse_ocr_pages", 0) or 0)
                st.write(
                    f"Parse Quality: {parse_score:.3f} ({parse_label}) | "
                    f"Pages: {parse_pages} | OCR pages: {parse_ocr_pages}"
                )
                st.caption(
                    f"Layout tags: table={int(paper.get('parse_table_lines', 0) or 0)}, "
                    f"formula={int(paper.get('parse_formula_lines', 0) or 0)}, "
                    f"figure_caption={int(paper.get('parse_figure_caption_lines', 0) or 0)}"
                )
                st.caption(f"Entry ID: {paper['entry_id']}")

                paper_key_raw = str(paper.get("canonical_id") or paper.get("entry_id") or f"paper_{idx}")
                paper_key = re.sub(r"[^A-Za-z0-9_]+", "_", paper_key_raw)
                title_for_update = paper.get("title", "")

                col_action_1, col_action_2 = st.columns(2)
                with col_action_1:
                    fav_label = "Unset Favorite" if coerce_bool(paper.get("is_favorite")) else "Mark Favorite"
                    if st.button(fav_label, key=f"toggle_fav_{idx}_{paper_key}"):
                        database.update_paper_metadata_by_title(
                            title_for_update,
                            {"is_favorite": not coerce_bool(paper.get("is_favorite"))}
                        )
                        st.cache_data.clear()
                        st.rerun()
                with col_action_2:
                    list_label = "Remove From Reading List" if coerce_bool(paper.get("in_reading_list")) else "Add To Reading List"
                    if st.button(list_label, key=f"toggle_list_{idx}_{paper_key}"):
                        database.update_paper_metadata_by_title(
                            title_for_update,
                            {"in_reading_list": not coerce_bool(paper.get("in_reading_list"))}
                        )
                        st.cache_data.clear()
                        st.rerun()

                tag_value = st.text_input(
                    "Edit tags (comma separated)",
                    value=format_tags(paper.get("tags", "")),
                    key=f"edit_tags_{idx}_{paper_key}"
                )
                if st.button("Save Tags", key=f"save_tags_{idx}_{paper_key}"):
                    database.update_paper_metadata_by_title(
                        title_for_update,
                        {"tags": tag_value}
                    )
                    st.cache_data.clear()
                    st.rerun()
                
                # Individual BibTeX copy
                paper_bibtex = bibtex.generate_bibtex_entry(paper)
                st.code(paper_bibtex, language="bibtex")
            st.divider()

with tab2:
    st.header("Search Literature")
    query = st.text_input("Enter a concept or question:", placeholder="e.g., Optimal mechanism for multi-item auctions")

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1, 1, 1, 1, 1])
    with col_s1:
        n_results = st.slider("Results", min_value=3, max_value=15, value=5)
    with col_s2:
        use_expansion = st.checkbox("Query expansion", value=True)
    with col_s3:
        show_snippets = st.checkbox("Evidence snippets", value=True)
    with col_s4:
        highlight_matches = st.checkbox("Highlight matches", value=True)
    with col_s5:
        retrieval_mode = st.selectbox("Retrieval Mode", ["auto", "hybrid", "vector", "bm25"], index=0)

    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c1:
        confidence_policy = st.selectbox(
            "Confidence Policy",
            ["auto", "off"],
            index=0,
            help="auto: low confidence refuse, medium confidence downgrade to extractive evidence-only output.",
        )
    with col_c2:
        refuse_threshold = st.slider(
            "Refuse threshold",
            min_value=0.10,
            max_value=0.70,
            value=0.28,
            step=0.01,
            help="Below this score, the assistant refuses to synthesize an answer.",
        )
    with col_c3:
        downgrade_threshold = st.slider(
            "Downgrade threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.48,
            step=0.01,
            help="Between refuse and downgrade thresholds, output is evidence-only observations.",
        )

    snippet_len = 360
    if show_snippets:
        snippet_len = st.slider("Snippet length", min_value=160, max_value=600, value=360)

    if query:
        expanded = {"expanded_query": query, "extra_terms": [], "rules_matched": []}
        if use_expansion:
            expanded = search_utils.expand_query(query)
            if expanded.get("extra_terms"):
                st.caption(f"Expanded terms: {', '.join(expanded['extra_terms'])}")

        tuned_query = expanded.get("expanded_query", query)
        if retrieval_mode == "auto":
            tuned = retrieval.auto_tune(tuned_query, n_results=n_results)
            st.caption(
                f"Auto mode -> {tuned.get('mode')} | top-k={tuned.get('n_results')} | reason={tuned.get('reason')}"
            )
        results = rag.query_db(tuned_query, n_results=n_results, mode=retrieval_mode)

        terms = search_utils.extract_query_terms(query)
        if expanded.get("extra_terms"):
            extra_term_str = " ".join(expanded.get("extra_terms", []))
            terms.extend(search_utils.extract_query_terms(extra_term_str))

        docs = results.get('documents', [[]])[0] if results.get('documents') else []
        metas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []

        if not docs:
            st.info("No results found.")
        else:
            for i in range(len(docs)):
                doc = docs[i]
                meta = metas[i] if i < len(metas) else {}
                distance = distances[i] if i < len(distances) else None
                score = None
                if distance is not None:
                    score = 1.0 / (1.0 + float(distance))

                with st.container():
                    title = meta.get('title', 'Unknown')
                    chunk_idx = meta.get('chunk_index')
                    header = f"**Source:** {title}"
                    if chunk_idx is not None:
                        header += f" — Chunk {chunk_idx}"
                    if score is not None:
                        header += f" — Score {score:.3f}"
                    st.markdown(header)
                    venue = meta.get("venue")
                    published = meta.get("published")
                    if venue or published:
                        line_parts = []
                        if venue:
                            line_parts.append(f"Venue: {venue}")
                        if published:
                            line_parts.append(f"Published: {published}")
                        st.caption(" | ".join(line_parts))

                    if show_snippets:
                        snippet = search_utils.extract_snippet(doc, terms, window=snippet_len)
                        if highlight_matches:
                            display = search_utils.highlight_text(snippet, terms)
                            st.markdown(display, unsafe_allow_html=True)
                        else:
                            st.write(snippet)
                        with st.expander("Show full chunk"):
                            if highlight_matches:
                                full_text = search_utils.highlight_text(doc, terms)
                                st.markdown(full_text, unsafe_allow_html=True)
                            else:
                                st.write(doc)
                    else:
                        if highlight_matches:
                            display = search_utils.highlight_text(doc, terms)
                            st.markdown(display, unsafe_allow_html=True)
                        else:
                            st.write(doc)
                    st.divider()

        st.markdown("#### Evidence-Grounded Answer")
        if st.button("Generate Answer With Citations", key="btn_answer_with_citations"):
            with st.spinner("Generating evidence-grounded answer..."):
                answer_pack = rag.answer_with_evidence(
                    tuned_query,
                    n_results=max(8, n_results),
                    mode=retrieval_mode,
                    confidence_policy=confidence_policy,
                    refuse_threshold=refuse_threshold,
                    downgrade_threshold=downgrade_threshold,
                )
                st.markdown(answer_pack.get("answer", "No answer generated."))

                confidence_info = answer_pack.get("confidence", {})
                if confidence_info:
                    st.caption(
                        "Confidence: "
                        f"{confidence_info.get('score', 0.0):.3f} | "
                        f"band={confidence_info.get('band', 'unknown')} | "
                        f"decision={answer_pack.get('decision', 'unknown')}"
                    )
                    reasons = confidence_info.get("reasons", [])
                    if reasons:
                        st.caption(f"Confidence reasons: {', '.join(reasons)}")

                evidence_rows = answer_pack.get("evidence", [])
                if evidence_rows:
                    st.caption("Source confidence")
                    for ev in evidence_rows:
                        st.caption(
                            f"[{ev.get('label')}] {ev.get('title')} | chunk {ev.get('chunk_index')} | confidence={ev.get('score', 0.0):.3f}"
                        )

                st.code(answer_pack.get("copy_markdown", ""), language="markdown")

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
    st.caption("Visualize citation/co-citation/coupling/author networks and identify unread recommendations.")
    
    # Lazy import
    from src import citation_graph
    
    st.info("""
    This feature builds multiple networks by:
    1. Matching your papers to Semantic Scholar
    2. Fetching references for each paper
    3. Building citation, co-citation, bibliographic coupling, and author collaboration graphs
    
    ⚠️ Only papers found in Semantic Scholar will appear. Large libraries may take several minutes.
    """)

    network_type = st.selectbox(
        "Network Type",
        ["citation", "co-citation", "bibliographic-coupling", "author-collaboration"],
        index=0
    )
    
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
                pack = citation_graph.build_enhanced_networks(
                    all_papers, 
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                status_text.empty()

                paper_info = pack.get("paper_info", {})
                if network_type == "citation":
                    G = pack.get("citation_graph")
                elif network_type == "co-citation":
                    G = pack.get("cocitation_graph")
                elif network_type == "bibliographic-coupling":
                    G = pack.get("coupling_graph")
                else:
                    G = pack.get("author_graph")

                st.success(f"Built {network_type} network: {len(G.nodes())} nodes, {len(G.edges())} links.")

                # Display graph
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

                st.markdown("#### Recommended Unread Papers")
                recs = pack.get("recommendations", [])
                if not recs:
                    st.caption("No unread recommendations available.")
                else:
                    for r in recs[:10]:
                        st.caption(
                            f"{r.get('title')} | score={r.get('score')} | "
                            f"citation_links={r.get('citation_links')} | "
                            f"co-citation_links={r.get('cocitation_links')} | "
                            f"coupling_links={r.get('coupling_links')}"
                        )

                st.markdown("#### Key Bridge Papers")
                bridge_papers = pack.get("bridge_papers", [])
                if not bridge_papers:
                    st.caption("No bridge papers identified.")
                else:
                    for b in bridge_papers[:10]:
                        st.caption(
                            f"{b.get('title')} | bridge_score={b.get('bridge_score')} | "
                            f"betweenness={b.get('betweenness')} | degree={b.get('degree')} | "
                            f"articulation={b.get('is_articulation')}"
                        )
                        examples = b.get("path_examples", []) or []
                        for ex in examples[:2]:
                            st.caption(f"Path: {ex}")

                st.markdown("#### Potential Missing Papers")
                missing_recs = pack.get("missing_recommendations", [])
                if not missing_recs:
                    st.caption("No potential missing papers detected from shared external references.")
                else:
                    for m in missing_recs[:12]:
                        st.caption(
                            f"{m.get('title')} | score={m.get('score')} | support_count={m.get('support_count')} | "
                            f"citation_count={m.get('citation_count')} | year={m.get('year')}"
                        )
                        supporters = m.get("supporting_titles", []) or []
                        if supporters:
                            st.caption(f"Supported by: {', '.join(supporters[:4])}")
                        for ex in (m.get("path_examples", []) or [])[:2]:
                            st.caption(f"Path: {ex}")

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
    mode_options = ["auto", "vector", "bm25", "hybrid"]
    default_mode = defaults["mode"] if defaults["mode"] in mode_options else "hybrid"
    mode = st.selectbox("Retrieval Mode", mode_options, index=mode_options.index(default_mode))
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
