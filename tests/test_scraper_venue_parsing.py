from src import scraper


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_search_semantic_scholar_parses_venue(monkeypatch):
    payload = {
        "data": [
            {
                "title": "Paper A",
                "authors": [{"name": "Alice"}],
                "abstract": "Abstract",
                "paperId": "pid-1",
                "year": 2024,
                "externalIds": {"DOI": "10.1000/test"},
                "publicationVenue": {"name": "Journal X"},
            }
        ]
    }

    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=200, payload=payload)

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    out = scraper.search_semantic_scholar("paper", max_results=1)
    assert len(out) == 1
    assert out[0]["venue"] == "Journal X"


def test_openalex_simple_parses_host_venue(monkeypatch):
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Paper B",
                "authorships": [{"author": {"display_name": "Bob"}}],
                "publication_year": 2023,
                "doi": "https://doi.org/10.2000/abc",
                "abstract_inverted_index": {"hello": [0], "world": [1]},
                "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
                "host_venue": {"display_name": "Venue Y"},
                "primary_location": {"source": {"display_name": "Fallback Venue"}},
            }
        ]
    }

    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=200, payload=payload)

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    out = scraper._search_openalex_simple("query", max_results=1, source_name="OpenAlex")
    assert len(out) == 1
    assert out[0]["venue"] == "Venue Y"
    assert out[0]["doi"] == "10.2000/abc"
