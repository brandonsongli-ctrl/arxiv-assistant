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
        "fields": "paperId,title,authors,year"
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
        "fields": "paperId,title,authors,year"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [c['citingPaper'] for c in data.get('data', []) if c.get('citingPaper')]
    except Exception as e:
        print(f"Error fetching citations for {paper_id}: {e}")
    
    return []


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
        title = info.get('title', 'Unknown')[:50]
        year = info.get('year', '?')
        citations = info.get('citation_count', 0)
        
        # Truncate title for display
        display_title = title[:40] + "..." if len(title) > 40 else title
        node_text.append(f"{display_title}<br>Year: {year}<br>Citations: {citations}")
        
        # Size based on citation count (log scale)
        import math
        size = 10 + min(30, math.log1p(citations) * 5)
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
            color=[paper_info.get(n, {}).get('citation_count', 0) for n in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Citations',
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
