import networkx as nx

from src import citation_graph


def test_identify_bridge_papers_with_path_examples():
    paper_info = {
        "A": {"title": "Paper A", "citation_count": 10},
        "B": {"title": "Paper B", "citation_count": 20},
        "C": {"title": "Paper C", "citation_count": 30},
    }

    citation = nx.DiGraph()
    for pid, info in paper_info.items():
        citation.add_node(pid, **info)
    citation.add_edge("A", "B", weight=1)
    citation.add_edge("B", "C", weight=1)

    cocitation = nx.Graph()
    for pid, info in paper_info.items():
        cocitation.add_node(pid, **info)

    coupling = nx.Graph()
    for pid, info in paper_info.items():
        coupling.add_node(pid, **info)

    bridges = citation_graph._identify_bridge_papers(
        paper_info,
        citation,
        cocitation,
        coupling,
        top_k=5,
    )

    assert bridges
    top = bridges[0]
    assert top["paper_id"] == "B"
    assert top["is_articulation"] is True
    assert top["path_examples"]
    assert "Paper A -> Paper B -> Paper C" in top["path_examples"][0]


def test_recommend_missing_papers_with_support_and_paths():
    paper_info = {
        "A": {"title": "Alpha"},
        "B": {"title": "Beta"},
    }
    external_ref_sources = {
        "X": {"A", "B"},
        "Y": {"A"},
    }
    external_ref_info = {
        "X": {"title": "Missing X", "year": 2022, "citationCount": 120},
        "Y": {"title": "Missing Y", "year": 2021, "citationCount": 45},
    }

    recs = citation_graph._recommend_missing_papers(
        external_ref_sources,
        external_ref_info,
        paper_info,
        top_k=10,
    )

    assert len(recs) == 1
    assert recs[0]["paper_id"] == "X"
    assert recs[0]["support_count"] == 2
    assert recs[0]["path_examples"]
    assert "Alpha -> Missing X <- Beta" in recs[0]["path_examples"][0]
