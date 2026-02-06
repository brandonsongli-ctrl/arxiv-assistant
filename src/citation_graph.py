"""
Citation Graph Module

Builds and visualizes citation relationships between papers using Semantic Scholar API.
"""

import requests
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
import math


def search_paper_by_title(title: str) -> Optional[Dict]:
    """
    Search for a paper in Semantic Scholar by title.
    Returns paper ID and basic info if found.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "paperId,title,authors,year,citationCount,referenceCount"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                return data['data'][0]
    except Exception as e:
        print(f"Error searching for {title}: {e}")
    
    return None


def get_paper_references(paper_id: str) -> List[Dict]:
    """
    Get papers that this paper cites (references).
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params = {
        "limit": 100,
        "fields": "paperId,title,authors,year,citationCount,venue"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [r['citedPaper'] for r in data.get('data', []) if r.get('citedPaper')]
    except Exception as e:
        print(f"Error fetching references for {paper_id}: {e}")
    
    return []


def get_paper_citations(paper_id: str) -> List[Dict]:
    """
    Get papers that cite this paper.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {
        "limit": 100,
        "fields": "paperId,title,authors,year,citationCount,venue"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [c['citingPaper'] for c in data.get('data', []) if c.get('citingPaper')]
    except Exception as e:
        print(f"Error fetching citations for {paper_id}: {e}")
    
    return []


def _author_name_list(author_items) -> List[str]:
    names = []
    for a in author_items or []:
        if isinstance(a, dict):
            n = a.get("name") or a.get("displayName")
            if n:
                names.append(str(n))
        elif isinstance(a, str):
            names.append(a)
    return names


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _paper_title(paper_info: Dict, paper_id: str) -> str:
    info = paper_info.get(paper_id, {})
    title = info.get("title")
    if title:
        return str(title)
    return str(paper_id)


def _format_path(path: List[str], paper_info: Dict) -> str:
    return " -> ".join([_paper_title(paper_info, pid) for pid in path])


def _build_combined_library_graph(
    citation_graph: nx.DiGraph,
    cocitation_graph: nx.Graph,
    coupling_graph: nx.Graph,
) -> nx.Graph:
    """
    Merge multiple relation graphs into one weighted undirected graph for centrality/path analysis.
    """
    combined = nx.Graph()

    for pid, attrs in citation_graph.nodes(data=True):
        combined.add_node(pid, **attrs)

    def _bump(u: str, v: str, w: float) -> None:
        if combined.has_edge(u, v):
            combined[u][v]["weight"] += float(w)
        else:
            combined.add_edge(u, v, weight=float(w))

    for u, v in citation_graph.edges():
        _bump(u, v, 1.0)

    for u, v, data in cocitation_graph.edges(data=True):
        # soften raw counts to avoid overweighting dense co-citation edges
        raw = float(data.get("weight", 1.0))
        _bump(u, v, 0.8 + min(2.0, raw * 0.15))

    for u, v, data in coupling_graph.edges(data=True):
        raw = float(data.get("weight", 1.0))
        _bump(u, v, 0.7 + min(2.0, raw * 0.12))

    return combined


def _bridge_path_examples(
    node_id: str,
    graph: nx.Graph,
    paper_info: Dict,
    max_examples: int = 2,
) -> List[str]:
    neighbors = list(graph.neighbors(node_id)) if node_id in graph else []
    if len(neighbors) < 2:
        return []

    examples = []
    try:
        g_removed = graph.copy()
        g_removed.remove_node(node_id)
        components = list(nx.connected_components(g_removed))
    except Exception:
        components = []

    # Try explanations where removing the bridge disconnects neighbor groups.
    comp_neighbors = defaultdict(list)
    if components:
        for nb in neighbors:
            for idx, comp in enumerate(components):
                if nb in comp:
                    comp_neighbors[idx].append(nb)
                    break
        comp_ids = list(comp_neighbors.keys())
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                src = comp_neighbors[comp_ids[i]][0]
                dst = comp_neighbors[comp_ids[j]][0]
                try:
                    path = nx.shortest_path(graph, source=src, target=dst)
                except Exception:
                    continue
                if node_id not in path:
                    continue
                examples.append(_format_path(path, paper_info))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break

    # Fallback: simple two-hop bridge pattern.
    if not examples:
        for i in range(min(len(neighbors), 4)):
            for j in range(i + 1, min(len(neighbors), 5)):
                path = [neighbors[i], node_id, neighbors[j]]
                examples.append(_format_path(path, paper_info))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break

    dedup = []
    seen = set()
    for x in examples:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup[:max_examples]


def _identify_bridge_papers(
    paper_info: Dict,
    citation_graph: nx.DiGraph,
    cocitation_graph: nx.Graph,
    coupling_graph: nx.Graph,
    top_k: int = 10,
) -> List[Dict]:
    combined = _build_combined_library_graph(citation_graph, cocitation_graph, coupling_graph)
    if combined.number_of_nodes() < 3:
        return []

    betweenness = nx.betweenness_centrality(combined, weight="weight", normalized=True)
    articulation_nodes = set(nx.articulation_points(combined))

    bridges = []
    for pid in combined.nodes():
        degree = int(combined.degree(pid))
        if degree < 2:
            continue

        b = float(betweenness.get(pid, 0.0))
        is_articulation = pid in articulation_nodes
        path_examples = _bridge_path_examples(pid, combined, paper_info, max_examples=2)

        if b <= 0 and not is_articulation and not path_examples:
            continue

        info = paper_info.get(pid, {})
        bridge_score = (2.0 * b) + (0.20 if is_articulation else 0.0) + min(0.25, degree * 0.03)
        bridges.append(
            {
                "paper_id": pid,
                "title": info.get("title", "Unknown"),
                "bridge_score": round(bridge_score, 4),
                "betweenness": round(b, 4),
                "degree": degree,
                "is_articulation": bool(is_articulation),
                "path_examples": path_examples,
                "year": info.get("year"),
                "citation_count": _safe_int(info.get("citation_count", 0)),
            }
        )

    bridges.sort(key=lambda x: x.get("bridge_score", 0.0), reverse=True)
    return bridges[:top_k]


def _missing_path_explanations(
    missing_title: str,
    supporter_ids: List[str],
    paper_info: Dict,
    max_examples: int = 3,
) -> List[str]:
    supporters = [
        _paper_title(paper_info, pid)
        for pid in supporter_ids
        if pid in paper_info
    ]
    supporters = [s for s in supporters if s]
    supporters = supporters[:6]

    examples = []
    for i in range(len(supporters)):
        for j in range(i + 1, len(supporters)):
            examples.append(f"{supporters[i]} -> {missing_title} <- {supporters[j]}")
            if len(examples) >= max_examples:
                return examples
    if not examples and supporters:
        examples.append(f"{supporters[0]} -> {missing_title}")
    return examples


def _recommend_missing_papers(
    external_ref_sources: Dict[str, set],
    external_ref_info: Dict[str, Dict],
    paper_info: Dict,
    top_k: int = 12,
) -> List[Dict]:
    """
    Recommend potentially missing papers from external references shared across library papers.
    """
    recs = []
    for pid, source_ids in external_ref_sources.items():
        if pid in paper_info:
            continue
        support_count = len(source_ids)
        if support_count < 2:
            continue

        info = external_ref_info.get(pid, {})
        title = info.get("title") or pid
        citation_count = _safe_int(info.get("citationCount", 0))
        year = _safe_int(info.get("year", 0))
        venue = info.get("venue", "")

        score = (2.5 * support_count) + min(3.0, math.log1p(max(0, citation_count)))
        if year >= 2018:
            score += 0.4

        supporters = sorted(
            list(source_ids),
            key=lambda sid: _paper_title(paper_info, sid).lower(),
        )
        supporting_titles = [_paper_title(paper_info, sid) for sid in supporters[:6]]
        path_examples = _missing_path_explanations(
            str(title),
            supporters,
            paper_info,
            max_examples=3,
        )

        recs.append(
            {
                "paper_id": pid,
                "title": title,
                "score": round(score, 4),
                "support_count": support_count,
                "supporting_titles": supporting_titles,
                "path_examples": path_examples,
                "year": year if year > 0 else None,
                "venue": venue,
                "citation_count": citation_count,
            }
        )

    recs.sort(key=lambda x: (x.get("score", 0.0), x.get("support_count", 0)), reverse=True)
    return recs[:top_k]


def build_enhanced_networks(papers: List[Dict], progress_callback=None) -> Dict:
    """
    Build citation/co-citation/bibliographic-coupling/author networks and reading recommendations.
    """
    paper_info = {}
    library_paper_ids = []
    title_to_local = {str(p.get("title", "")).strip().lower(): p for p in papers}

    def process_paper(paper):
        title = paper.get("title", "")
        if not title or title == "Unknown":
            return None
        result = search_paper_by_title(title)
        if result:
            pid = result.get("paperId")
            if not pid:
                return None
            return {
                "paper_id": pid,
                "title": result.get("title", title),
                "authors": result.get("authors", []),
                "year": result.get("year"),
                "citation_count": result.get("citationCount", 0),
                "reference_count": result.get("referenceCount", 0),
                "in_library": True,
                "local": title_to_local.get(str(title).strip().lower(), paper),
            }
        return None

    total = len(papers)
    completed = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_paper, p): p for p in papers}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                pid = result["paper_id"]
                paper_info[pid] = result
                library_paper_ids.append(pid)
            if progress_callback:
                progress_callback(completed, total, "Matching library papers")
            time.sleep(0.05)

    citation_graph = nx.DiGraph()
    cocitation_graph = nx.Graph()
    coupling_graph = nx.Graph()
    author_graph = nx.Graph()

    for pid in library_paper_ids:
        citation_graph.add_node(pid, **paper_info[pid])
        cocitation_graph.add_node(pid, **paper_info[pid])
        coupling_graph.add_node(pid, **paper_info[pid])

    # reference cache for each library paper
    refs_by_paper: Dict[str, set] = {}
    external_ref_sources: Dict[str, set] = defaultdict(set)
    external_ref_info: Dict[str, Dict] = {}
    completed = 0
    for pid in library_paper_ids:
        refs = get_paper_references(pid)
        ref_ids = {r.get("paperId") for r in refs if r.get("paperId")}
        refs_by_paper[pid] = ref_ids

        for r in refs:
            rid = r.get("paperId")
            if not rid:
                continue
            if rid not in paper_info:
                external_ref_sources[rid].add(pid)
                if rid not in external_ref_info:
                    external_ref_info[rid] = {
                        "paperId": rid,
                        "title": r.get("title"),
                        "year": r.get("year"),
                        "venue": r.get("venue"),
                        "citationCount": r.get("citationCount", 0),
                        "authors": r.get("authors", []),
                    }

        # Directed citation edges within library
        for rid in ref_ids:
            if rid in paper_info:
                citation_graph.add_edge(pid, rid, weight=1)

        completed += 1
        if progress_callback:
            progress_callback(completed, len(library_paper_ids), "Fetching references")
        time.sleep(0.05)

    # Co-citation: two library papers cited together by same library paper.
    for pid, ref_ids in refs_by_paper.items():
        local_refs = [rid for rid in ref_ids if rid in paper_info]
        for i in range(len(local_refs)):
            for j in range(i + 1, len(local_refs)):
                a = local_refs[i]
                b = local_refs[j]
                if cocitation_graph.has_edge(a, b):
                    cocitation_graph[a][b]["weight"] += 1
                else:
                    cocitation_graph.add_edge(a, b, weight=1)

    # Bibliographic coupling: two library papers share references (can be outside library).
    pids = list(refs_by_paper.keys())
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            p1, p2 = pids[i], pids[j]
            inter = refs_by_paper[p1] & refs_by_paper[p2]
            w = len(inter)
            if w > 0:
                coupling_graph.add_edge(p1, p2, weight=w)

    # Author network
    for pid in library_paper_ids:
        names = _author_name_list(paper_info.get(pid, {}).get("authors", []))
        for n in names:
            if not author_graph.has_node(n):
                author_graph.add_node(n, label=n)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]
                if author_graph.has_edge(a, b):
                    author_graph[a][b]["weight"] += 1
                else:
                    author_graph.add_edge(a, b, weight=1)

    recommendations = _recommend_unread_papers(paper_info, citation_graph, cocitation_graph, coupling_graph)
    bridge_papers = _identify_bridge_papers(
        paper_info,
        citation_graph,
        cocitation_graph,
        coupling_graph,
        top_k=10,
    )
    missing_recommendations = _recommend_missing_papers(
        external_ref_sources,
        external_ref_info,
        paper_info,
        top_k=12,
    )

    return {
        "paper_info": paper_info,
        "citation_graph": citation_graph,
        "cocitation_graph": cocitation_graph,
        "coupling_graph": coupling_graph,
        "author_graph": author_graph,
        "recommendations": recommendations,
        "bridge_papers": bridge_papers,
        "missing_recommendations": missing_recommendations,
    }


def _recommend_unread_papers(
    paper_info: Dict,
    citation_graph: nx.DiGraph,
    cocitation_graph: nx.Graph,
    coupling_graph: nx.Graph,
    top_k: int = 10,
) -> List[Dict]:
    recs = []
    for pid, info in paper_info.items():
        local = info.get("local", {}) or {}
        in_reading_list = bool(local.get("in_reading_list"))
        if in_reading_list:
            continue
        deg_cit = citation_graph.in_degree(pid) + citation_graph.out_degree(pid)
        deg_cocit = cocitation_graph.degree(pid) if pid in cocitation_graph else 0
        deg_coupling = coupling_graph.degree(pid) if pid in coupling_graph else 0
        score = (2.0 * deg_cit) + (1.2 * deg_cocit) + (1.0 * deg_coupling)
        recs.append({
            "paper_id": pid,
            "title": info.get("title", "Unknown"),
            "score": round(score, 3),
            "citation_links": deg_cit,
            "cocitation_links": deg_cocit,
            "coupling_links": deg_coupling,
        })
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:top_k]


def build_citation_graph(papers: List[Dict], progress_callback=None) -> Tuple[nx.DiGraph, Dict]:
    """
    Build a citation graph for papers in the library.
    
    Returns:
        - NetworkX DiGraph (edges go from citing paper TO cited paper)
        - Dict mapping paper_id -> paper info
    """
    G = nx.DiGraph()
    paper_info = {}
    
    # Step 1: Find Semantic Scholar IDs for library papers
    library_paper_ids = set()
    total = len(papers)
    
    def process_paper(paper):
        title = paper.get('title', '')
        if not title or title == 'Unknown':
            return None
        
        result = search_paper_by_title(title)
        if result:
            return {
                'paper_id': result['paperId'],
                'title': result.get('title', title),
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'citation_count': result.get('citationCount', 0),
                'reference_count': result.get('referenceCount', 0),
                'in_library': True
            }
        return None
    
    # Parallel lookup of paper IDs
    completed = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_paper, p): p for p in papers}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                pid = result['paper_id']
                paper_info[pid] = result
                library_paper_ids.add(pid)
                G.add_node(pid, **result)
            
            if progress_callback:
                progress_callback(completed, total, "Finding papers...")
            
            time.sleep(0.1)  # Rate limiting
    
    # Step 2: Fetch citations between library papers
    if progress_callback:
        progress_callback(0, len(library_paper_ids), "Building citation links...")
    
    completed = 0
    for pid in library_paper_ids:
        # Get references (papers this one cites)
        refs = get_paper_references(pid)
        for ref in refs:
            ref_id = ref.get('paperId')
            if ref_id and ref_id in library_paper_ids:
                # This library paper cites another library paper
                G.add_edge(pid, ref_id)
        
        completed += 1
        if progress_callback:
            progress_callback(completed, len(library_paper_ids), "Building links...")
        
        time.sleep(0.1)  # Rate limiting
    
    return G, paper_info


def graph_to_plotly(G: nx.DiGraph, paper_info: Dict) -> go.Figure:
    """
    Convert NetworkX graph to Plotly figure for visualization.
    """
    if len(G.nodes()) == 0:
        # Empty graph
        fig = go.Figure()
        fig.add_annotation(
            text="No citation relationships found between library papers.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Citation Graph",
            showlegend=False
        )
        return fig
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        info = paper_info.get(node, {})
        title = info.get('title', str(node))[:50]
        year = info.get('year', '?')
        citations = info.get('citation_count', 0)
        if citations is None:
            citations = 0
        
        # Truncate title for display
        display_title = title[:40] + "..." if len(title) > 40 else title
        if info:
            node_text.append(f"{display_title}<br>Year: {year}<br>Citations: {citations}")
        else:
            deg = G.degree(node) if hasattr(G, "degree") else 0
            node_text.append(f"{display_title}<br>Degree: {deg}")
        
        # Size based on citation count (log scale)
        import math
        if info:
            size = 10 + min(30, math.log1p(citations) * 5)
        else:
            size = 10 + min(30, math.log1p(max(1, G.degree(node))) * 8)
        node_size.append(size)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=[
                paper_info.get(n, {}).get('citation_count', 0) if paper_info.get(n, {}) else (G.degree(n) if hasattr(G, "degree") else 0)
                for n in G.nodes()
            ],
            colorbar=dict(
                thickness=15,
                title='Node Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Citation Graph ({len(G.nodes())} papers, {len(G.edges())} links)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    return fig
