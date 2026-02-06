from src import pipeline


def test_incremental_pipeline_runs_metadata_fix_and_reindex(monkeypatch):
    papers = [
        {
            "title": "Paper A",
            "authors": "Unknown",
            "published": "Unknown",
            "doi": "",
            "venue": "",
            "entry_id": "id-a",
            "canonical_id": "cid-a",
        }
    ]

    monkeypatch.setattr(pipeline.database, "get_all_papers", lambda: list(papers))
    monkeypatch.setattr(
        pipeline.scraper,
        "resolve_paper_metadata",
        lambda title: {
            "found": True,
            "title": "Paper A",
            "authors": ["Alice"],
            "published": "2024",
            "doi": "10.1000/a",
            "venue": "Journal A",
        },
    )
    monkeypatch.setattr(pipeline.enrich, "update_paper_metadata", lambda title, payload: 3)
    monkeypatch.setattr(pipeline, "_find_duplicate_group_for_title", lambda title, min_score=0.84: [])
    monkeypatch.setattr(pipeline.database, "reindex_chunks_by_title", lambda title: 4)

    result = pipeline.run_incremental_indexing(
        "Paper A",
        run_metadata_fix=True,
        dedupe_threshold=0.84,
        rebuild_vectors=True,
    )
    assert result["metadata_updates"] == 3
    assert result["dedupe_groups"] == 0
    assert result["reindexed_chunks"] == 4
    assert "Paper A" in result["reindexed_titles"]


def test_incremental_pipeline_merges_duplicates(monkeypatch):
    papers = [
        {"title": "Paper A", "authors": ["Alice"], "published": "2024", "doi": "10.1/a", "venue": "J", "entry_id": "id-a", "canonical_id": "cid-a"},
        {"title": "Paper A v2", "authors": ["Alice"], "published": "2024", "doi": "10.1/a", "venue": "J", "entry_id": "id-b", "canonical_id": "cid-b"},
    ]

    monkeypatch.setattr(pipeline.database, "get_all_papers", lambda: list(papers))
    monkeypatch.setattr(pipeline.scraper, "resolve_paper_metadata", lambda title: {"found": False})

    group = list(papers)
    monkeypatch.setattr(
        pipeline,
        "_find_duplicate_group_for_title",
        lambda title, min_score=0.84: group if title == "Paper A" else [],
    )
    monkeypatch.setattr(
        pipeline.dedupe,
        "merge_duplicates_group",
        lambda g: {"keep": {"title": "Paper A"}, "removed": [("Paper A v2", True)], "merged_metadata": {}},
    )
    monkeypatch.setattr(pipeline.database, "reindex_chunks_by_title", lambda title: 2)

    result = pipeline.run_incremental_indexing(
        "Paper A",
        run_metadata_fix=False,
        dedupe_threshold=0.84,
        rebuild_vectors=True,
    )
    assert result["dedupe_groups"] == 1
    assert result["dedupe_removed"] == 1
    assert result["reindexed_chunks"] >= 2
