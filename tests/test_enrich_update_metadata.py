from src import enrich


def test_update_paper_metadata_recomputes_canonical(monkeypatch):
    captured = {}

    def fake_get_all_papers():
        return [
            {
                "title": "Test Paper",
                "entry_id": "https://arxiv.org/abs/1234.5678v1",
                "doi": "",
                "canonical_id": "",
            }
        ]

    def fake_update(title, payload):
        captured["title"] = title
        captured["payload"] = payload
        return 2

    monkeypatch.setattr(enrich.database, "get_all_papers", fake_get_all_papers)
    monkeypatch.setattr(enrich.database, "update_paper_metadata_by_title", fake_update)

    result = enrich.update_paper_metadata(
        "Test Paper",
        {
            "doi": "10.1000/ABC",
            "entry_id": "https://arxiv.org/abs/1234.5678v1",
        },
    )

    assert result == 2
    assert captured["title"] == "Test Paper"
    assert captured["payload"]["doi"] == "10.1000/abc"
    assert captured["payload"]["canonical_id"].startswith("doi:10.1000/abc")
