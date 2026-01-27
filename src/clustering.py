"""
Paper Clustering Module

Clusters papers by topic using embeddings and provides visualization.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from src import database

def get_paper_embeddings() -> Tuple[List[str], np.ndarray]:
    """
    Get average embedding for each paper in the database.
    
    Returns:
        Tuple of (paper_titles, embeddings_matrix)
    """
    return database.get_paper_embeddings_aggregated()



def cluster_papers(n_clusters: Optional[int] = None, min_cluster_size: int = 2) -> Dict:
    """
    Cluster papers using UMAP for dimensionality reduction and HDBSCAN for clustering.
    
    Args:
        n_clusters: Not used for HDBSCAN (auto-determined), kept for API compatibility
        min_cluster_size: Minimum number of papers to form a cluster
    
    Returns:
        Dictionary with clustering results
    """
    titles, embeddings = get_paper_embeddings()
    
    if len(titles) < 3:
        return {
            "error": "Need at least 3 papers for clustering",
            "paper_count": len(titles)
        }
    
    try:
        import umap
        import hdbscan
    except ImportError:
        return {
            "error": "Please install: pip install umap-learn hdbscan",
            "paper_count": len(titles)
        }
    
    # Reduce to 2D for visualization
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(titles) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Cluster
    # Adjust min_cluster_size based on dataset size
    effective_min_size = min(min_cluster_size, max(2, len(titles) // 5))
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    
    # Organize results
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_name = f"Cluster {label}" if label >= 0 else "Unclustered"
        clusters[cluster_name].append({
            'title': titles[i],
            'x': float(embeddings_2d[i, 0]),
            'y': float(embeddings_2d[i, 1])
        })
    
    return {
        "clusters": dict(clusters),
        "n_clusters": len([k for k in clusters.keys() if k != "Unclustered"]),
        "paper_count": len(titles),
        "embeddings_2d": embeddings_2d.tolist(),
        "titles": titles,
        "labels": cluster_labels.tolist()
    }


def label_clusters_with_llm(clusters: Dict) -> Dict[str, str]:
    """
    Use LLM to generate descriptive labels for each cluster based on paper titles.
    
    Args:
        clusters: Output from cluster_papers()
    
    Returns:
        Dictionary mapping cluster names to descriptive labels
    """
    from src.rag import ask_llm
    
    labels = {}
    
    for cluster_name, papers in clusters.get('clusters', {}).items():
        if cluster_name == "Unclustered":
            labels[cluster_name] = "Miscellaneous / Uncategorized"
            continue
        
        # Get titles in this cluster
        titles = [p['title'] for p in papers[:10]]  # Limit to 10 for prompt size
        titles_text = "\n".join(f"- {t}" for t in titles)
        
        prompt = f"""Based on these academic paper titles, provide a SHORT (3-5 words) descriptive topic label for this research cluster. 
Only respond with the label, no explanation.

Paper titles:
{titles_text}

Topic label:"""
        
        try:
            label = ask_llm(prompt).strip()
            # Clean up the label
            label = label.replace('"', '').replace("'", "").strip()
            if len(label) > 50:
                label = label[:50] + "..."
            labels[cluster_name] = label
        except Exception as e:
            labels[cluster_name] = cluster_name
    
    return labels


def plot_clusters(cluster_result: Dict, cluster_labels: Dict[str, str] = None) -> "go.Figure":
    """
    Create an interactive Plotly visualization of the paper clusters.
    
    Args:
        cluster_result: Output from cluster_papers()
        cluster_labels: Optional descriptive labels from label_clusters_with_llm()
    
    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")
    
    if "error" in cluster_result:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=cluster_result["error"],
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Prepare data
    titles = cluster_result['titles']
    embeddings_2d = np.array(cluster_result['embeddings_2d'])
    labels = cluster_result['labels']
    
    # Create color map
    unique_labels = list(set(labels))
    colors = px.colors.qualitative.Set2
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Special color for unclustered (-1)
    if -1 in color_map:
        color_map[-1] = 'lightgray'
    
    fig = go.Figure()
    
    # Add traces for each cluster
    for label in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == label]
        
        if cluster_labels:
            cluster_name = f"Cluster {label}" if label >= 0 else "Unclustered"
            display_name = cluster_labels.get(cluster_name, cluster_name)
        else:
            display_name = f"Cluster {label}" if label >= 0 else "Unclustered"
        
        cluster_titles = [titles[i] for i in mask]
        # Truncate titles for hover
        hover_titles = [t[:80] + "..." if len(t) > 80 else t for t in cluster_titles]
        
        fig.add_trace(go.Scatter(
            x=embeddings_2d[mask, 0],
            y=embeddings_2d[mask, 1],
            mode='markers',
            name=display_name,
            text=hover_titles,
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker=dict(
                size=10,
                color=color_map[label],
                line=dict(width=1, color='white')
            )
        ))
    
    fig.update_layout(
        title="Paper Topic Clusters",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        template="plotly_white",
        width=800,
        height=600
    )
    
    return fig


def get_cluster_summary(cluster_result: Dict) -> str:
    """
    Generate a text summary of the clustering results.
    
    Args:
        cluster_result: Output from cluster_papers()
    
    Returns:
        Formatted markdown string
    """
    if "error" in cluster_result:
        return f"**Error:** {cluster_result['error']}"
    
    lines = []
    lines.append(f"## Clustering Summary")
    lines.append(f"- **Total Papers:** {cluster_result['paper_count']}")
    lines.append(f"- **Number of Clusters:** {cluster_result['n_clusters']}")
    lines.append("")
    
    for cluster_name, papers in cluster_result.get('clusters', {}).items():
        lines.append(f"### {cluster_name} ({len(papers)} papers)")
        for paper in papers[:5]:  # Show up to 5 papers per cluster
            lines.append(f"- {paper['title']}")
        if len(papers) > 5:
            lines.append(f"- *... and {len(papers) - 5} more*")
        lines.append("")
    
    return '\n'.join(lines)


def find_similar_papers(title: str, n_results: int = 5) -> List[Dict]:
    """
    Find papers most similar to a given paper.
    
    Args:
        title: Title of the reference paper
        n_results: Number of similar papers to return
    
    Returns:
        List of similar papers with similarity scores
    """
    titles, embeddings = get_paper_embeddings()
    
    if title not in titles:
        return []
    
    ref_idx = titles.index(title)
    ref_embedding = embeddings[ref_idx]
    
    # Calculate cosine similarities
    similarities = []
    for i, emb in enumerate(embeddings):
        if i == ref_idx:
            continue
        sim = np.dot(ref_embedding, emb) / (np.linalg.norm(ref_embedding) * np.linalg.norm(emb))
        similarities.append((titles[i], float(sim)))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [{'title': t, 'similarity': s} for t, s in similarities[:n_results]]
