import os

from src import watchlist


def test_run_due_watches_enqueues_new_items(monkeypatch, tmp_path):
    monkeypatch.setattr(watchlist, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    monkeypatch.setattr(watchlist, "DIGEST_DIR", str(tmp_path / "digests"))

    watchlist.add_watch("mechanism design", watch_type="keyword", source="arxiv", frequency="daily")

    fake_results = [
        {"title": "Paper A", "published": "2024", "venue": "Test Journal", "doi": "10.1000/a"},
        {"title": "Paper B", "published": "2023", "venue": "Test Conf", "doi": "10.1000/b"},
    ]
    monkeypatch.setattr(watchlist, "_search_watch", lambda watch, max_results=10: list(fake_results))
    monkeypatch.setattr(watchlist, "_filter_new_results", lambda results: list(results))

    queued = []

    def fake_enqueue(paper, run_enrichment=True):
        queued.append((paper.get("title"), run_enrichment))
        return f"task-{len(queued)}"

    result = watchlist.run_due_watches(
        max_results_per_watch=8,
        enqueue_new=True,
        enqueue_fn=fake_enqueue,
        run_enrichment=False,
    )

    assert result["ran"] == 1
    assert result["new_papers"] == 2
    assert result["queued_tasks"] == 2
    assert len(queued) == 2
    assert all(not enrich_flag for _, enrich_flag in queued)

    digest_path = result["digest_path"]
    assert digest_path
    assert os.path.exists(digest_path)
    with open(digest_path, "r", encoding="utf-8") as f:
        digest = f.read()
    assert "Queued 2 new papers for background ingest." in digest


def test_run_due_watches_without_enqueue(monkeypatch, tmp_path):
    monkeypatch.setattr(watchlist, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    monkeypatch.setattr(watchlist, "DIGEST_DIR", str(tmp_path / "digests"))

    watchlist.add_watch("information design", watch_type="keyword", source="arxiv", frequency="daily")

    fake_results = [{"title": "Paper C", "published": "2025", "venue": "Venue", "doi": "10.1000/c"}]
    monkeypatch.setattr(watchlist, "_search_watch", lambda watch, max_results=10: list(fake_results))
    monkeypatch.setattr(watchlist, "_filter_new_results", lambda results: list(results))

    result = watchlist.run_due_watches(max_results_per_watch=8, enqueue_new=False)

    assert result["ran"] == 1
    assert result["new_papers"] == 1
    assert result["queued_tasks"] == 0
