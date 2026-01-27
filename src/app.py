import streamlit as st
import os
import sys
import time

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import scraper, ingest, rag
from src import bibtex, citations, patterns, clustering, database
# review is imported lazily inside its tab to avoid circular import issues

st.set_page_config(page_title="Local Academic Assistant", layout="wide")

st.title("📚 Local Academic Literature Assistant")
st.markdown("Focused on **Microeconomic Theory, Information Design, Contract Theory, and Mechanism Design**.")

# Sidebar: Manage Database
with st.sidebar:
    st.header("Manage Database")
    
    st.subheader("Add Papers")
    
    source = st.selectbox("Source", ["ArXiv", "Semantic Scholar"])

    # Search controls
    col_search, col_sort = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("Search (Keywords or ID)", value="Mechanism Design")
    with col_sort:
        if source == "ArXiv":
            sort_option = st.selectbox("Sort By", ["Relevance", "Last Updated", "Submitted Date"])
        else:
            sort_option = st.selectbox("Sort By", ["Relevance (Default)"], disabled=True)
    
    sort_map = {
        "Relevance": "relevance",
        "Last Updated": "last_updated",
        "Submitted Date": "submitted_date",
        "Relevance (Default)": "relevance"
    }

    if st.button("Search"):
        with st.spinner(f"Searching {source}..."):
            if source == "ArXiv":
                results = scraper.search_arxiv(search_query, max_results=5, sort_by=sort_map[sort_option])
            else:
                results = scraper.search_semantic_scholar(search_query, max_results=5)
            st.session_state['search_results'] = results
            
    if 'search_results' in st.session_state:
        st.write(f"Found {len(st.session_state['search_results'])} papers:")
        for i, paper in enumerate(st.session_state['search_results']):
            with st.expander(f"{paper['title']}"):
                st.caption(f"Authors: {', '.join(paper['authors'])}")
                st.caption(f"Published: {paper['published']}")
                st.write(paper['summary'])
                if st.button(f"Download & Ingest {i}", key=f"btn_{i}"):
                    with st.spinner("Downloading and Ingesting..."):
                        # Download
                        download_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
                        
                        pdf_path = None
                        if paper.get("source") == "semanticscholar":
                            if paper.get("pdf_url"):
                                pdf_path = scraper.download_from_url(paper['pdf_url'], paper['title'], download_dir)
                            else:
                                st.error("No open access PDF available for this paper.")
                        else:
                            # ArXiv
                             pdf_path = scraper.download_paper(paper['obj'], download_dir)
                        
                        if pdf_path:
                            st.success(f"Downloaded to {pdf_path}")
                            
                            # Ingest
                            ingest.ingest_paper(pdf_path, paper)
                            st.success("Added to Database!")

    st.markdown("---")
    st.subheader("Upload Local PDF")
    st.subheader("Upload Local PDF(s)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"selected {len(uploaded_files)} files")
        
        if st.button(f"Ingest {len(uploaded_files)} Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                
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
                    final_title = resolved['title']
                    final_authors = resolved['authors']
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

# Metadata Enrichment Section
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Enrich Metadata")
st.sidebar.caption("Auto-fill missing DOI, authors, and year for papers with 'Unknown' metadata.")

if st.sidebar.button("🔄 Enrich All Papers"):
    from src import enrich
    
    # Get count of papers needing enrichment
    papers_to_enrich = enrich.get_papers_needing_enrichment()
    
    if len(papers_to_enrich) == 0:
        st.sidebar.success("All papers already have complete metadata!")
    else:
        st.sidebar.info(f"Found {len(papers_to_enrich)} papers with missing metadata. Starting enrichment...")
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def update_progress(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)
        
        results = enrich.enrich_all_papers(progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        st.sidebar.success(f"✅ Done! Updated {results['success']}/{results['total']} papers.")
        
        if results['failed'] > 0:
            st.sidebar.warning(f"⚠️ {results['failed']} papers could not be matched.")
        
        time.sleep(1)
        st.rerun()

# Database Management Section
st.sidebar.markdown("---")
with st.sidebar.expander("🗑️ Manage Database"):
    st.caption("Fix metadata errors or delete unwanted papers.")
    
    # Get all current titles
    all_papers_dict = ingest.get_all_papers()
    all_titles = sorted([p['title'] for p in all_papers_dict])
    
    if not all_titles:
        st.info("Database is empty.")
    else:
        paper_to_delete = st.selectbox("Select paper to manage:", ["-- Select --"] + all_titles)
        
        if paper_to_delete != "-- Select --":
            col_manage_1, col_manage_2 = st.columns(2)
            
            with col_manage_1:
                if st.button("✏️ Fix Metadata"):
                    # Find the full paper object
                    target_paper = next((p for p in all_papers_dict if p['title'] == paper_to_delete), None)
                    if target_paper:
                        with st.spinner("Re-analyzing paper..."):
                            # Force re-enrichment
                            new_title, success, msg = enrich.enrich_single_paper(target_paper)
                            if success:
                                st.success(f"Fixed! New title: {new_title}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning(f"Could not improve metadata: {msg}")
            
            with col_manage_2:
                if st.button("❌ Delete"):
                    if database.delete_paper_by_title(paper_to_delete):
                        st.success("Deleted!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed.")
        
        st.markdown("---")
        if st.button("⚠️ RESET DATABASE (Danger)"):
            database.drop_tables()
            st.success("Database wiped clean!")
            time.sleep(1)
            st.rerun()

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📚 Local Library", 
    "🔍 Semantic Search", 
    "✍️ Rewrite / Polish", 
    "💡 Idea Generation",
    "📝 Sentence Patterns",
    "🔬 Topic Clusters",
    "📎 Citation Finder",
    "📝 Literature Review",
    "📊 Citation Graph"
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
        
    papers = ingest.get_all_papers()
    
    if not papers:
        st.info("No papers found. Use the sidebar to add some!")
    else:
        st.metric("Total Papers", len(papers))
        
        # Display as a table or cards
        for idx, paper in enumerate(papers):
            with st.expander(f"📄 {paper['title']}"):
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Published:** {paper['published']}")
                st.write(f"**DOI:** {paper.get('doi', 'N/A')}")
                st.write(f"**Vector Chunks:** {paper['chunk_count']}")
                st.caption(f"Entry ID: {paper['entry_id']}")
                
                # Individual BibTeX copy
                paper_bibtex = bibtex.generate_bibtex_entry(paper)
                st.code(paper_bibtex, language="bibtex")

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
                st.markdown(f"**Source:** {meta.get('title', 'Unknown')} (Chunk {meta.get('chunk_index')})")
                st.info(doc)
                st.divider()

with tab3:
    st.header("Academic Sentence Polisher")
    sentence = st.text_area("Draft Sentence:", placeholder="I want to say that the agent will lie if the incentive is not good.")
    
    col1, col2 = st.columns(2)
    with col1:
        use_context = st.checkbox("Use Retrieval Context?", value=True)
        
    if st.button("Rewrite"):
        context = ""
        if use_context:
            with st.spinner("Retrieving context..."):
                results = rag.query_db(sentence, n_results=3)
                context = rag.format_context(results)
                with st.expander("View Retrieved Context"):
                    st.write(context)
        
        with st.spinner("Polishing..."):
            rewritten = rag.rewrite_sentence(sentence, context)
            st.markdown("### Suggestion:")
            st.success(rewritten)

with tab4:
    st.header("Research Idea Generator")
    topic = st.text_input("Research Topic:", placeholder="Information Design in Networks")
    
    if st.button("Generate Ideas"):
        with st.spinner("Analyzing literature and thinking..."):
            # First retrieve relevant papers
            results = rag.query_db(topic, n_results=10)
            context = rag.format_context(results)
            
            # Then ask LLM
            ideas = rag.generate_ideas(topic, context)
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
            for pattern_text, data in results[:20]:
                with st.expander(f"**{pattern_text}** (used {data['count']} times) - {data['category']}"):
                    st.write("**Examples from your papers:**")
                    for ex in data.get('examples', [])[:3]:
                        st.info(f"_{ex['sentence']}_")
                        st.caption(f"Source: {ex['source']}")
        else:
            # Show patterns grouped by category
            categories = patterns.get_patterns_by_category(library)
            for category, cat_patterns in categories.items():
                with st.expander(f"**{category}** ({len(cat_patterns)} patterns)"):
                    for p in cat_patterns[:10]:
                        st.write(f"• **{p['text']}** ({p['count']} uses)")
                        if p.get('examples'):
                            st.caption(f"  Example: _{p['examples'][0]['sentence'][:100]}..._")
                            
            st.markdown("---")
            st.subheader("🛡️ Structural Analysis (Dynamic)")
            st.caption("Common sentence templates discovered automatically from your papers.")
            
            struct_patterns = library.get('structural_patterns', {})
            if struct_patterns:
                # Group by Category (parsed from [CAT] prefix)
                grouped_templates = {}
                for template, data in struct_patterns.items():
                    # Extract [CATEGORY] if present
                    import re
                    match = re.match(r"^\[(\w+)\] (.+)", template)
                    if match:
                        cat = match.group(1)
                        clean_template = match.group(2)
                        if cat not in grouped_templates:
                            grouped_templates[cat] = []
                        grouped_templates[cat].append((clean_template, data))
                    else:
                        if "Other" not in grouped_templates:
                            grouped_templates["Other"] = []
                        grouped_templates["Other"].append((template, data))
                
                # Display by Category
                # Defined sort order for categories
                cat_order = ["INTRO", "MODEL", "ARGUMENT", "LOGIC", "RESULT", "ECON", "DISC", "LIT", "META", "Other"]
                
                for cat in cat_order:
                    if cat in grouped_templates:
                        with st.expander(f"📚 {cat} Patterns ({len(grouped_templates[cat])} types)"):
                            # Sort by count
                            sorted_patterns = sorted(grouped_templates[cat], key=lambda x: x[1]['count'], reverse=True)
                            
                            for template, data in sorted_patterns:
                                if data['count'] > 0:
                                    st.markdown(f"**{template}** — `{data['count']} matches`")
                                    
                                    # Show examples in a small scrollable area or just list them
                                    examples = data.get('examples', [])
                                    if examples:
                                        for i, ex in enumerate(examples):
                                            # Highlight the matched part? For now just italics
                                            st.info(f"_{ex['sentence']}_")
                                            st.caption(f"Source: {ex['source']}")
                                            if i >= 4: # Show max 5 by default per template in this view to save space
                                                if len(examples) > 5:
                                                    st.caption(f"... and {len(examples)-5} more")
                                                break
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
        if st.button("🏷️ Generate Cluster Labels (LLM)"):
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
            with st.expander("📊 Cluster Details"):
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
                        
                        with st.expander(f"**{i}. {paper['title']}** (Relevance: {score_pct:.1f}%)"):
                            st.write(f"**Authors:** {paper['authors']}")
                            st.write(f"**Published:** {paper['published']}")
                            
                            # Show BibTeX
                            paper_bibtex = bibtex.generate_bibtex_entry(paper)
                            st.code(paper_bibtex, language="bibtex")
        else:
            st.warning("Please enter some text to find citations for.")

with tab8:
    st.header("📝 Literature Review Generator")
    st.caption("Generate a comparative literature review for selected papers.")
    
    # Lazy import to avoid circular import issues
    import src.review as review
    
    # Get all papers for selection
    all_papers = ingest.get_all_papers()
    paper_titles = [p['title'] for p in all_papers]
    paper_titles.sort()
    
    selected_papers = st.multiselect(
        "Select papers to review (2-10 recommended):",
        options=paper_titles,
        max_selections=10
    )
    
    if st.button("Generate Review"):
        if not selected_papers:
            st.warning("Please select at least one paper.")
        else:
            with st.spinner(f"Reading {len(selected_papers)} papers and generating review..."):
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
        all_papers = ingest.get_all_papers()
        
        if len(all_papers) == 0:
            st.warning("No papers in library. Add some papers first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress_bar.progress(current / total if total > 0 else 0)
                status_text.text(f"{message} ({current}/{total})")
            
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Local LLMs & ChromaDB")
