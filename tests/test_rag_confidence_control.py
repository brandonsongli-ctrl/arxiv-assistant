from src import rag


def _evidence(scores):
    out = []
    for i, s in enumerate(scores, start=1):
        out.append(
            {
                "label": f"S{i}",
                "id": f"id-{i}",
                "title": f"Paper {i}",
                "chunk_index": i,
                "score": float(s),
                "text": f"Evidence text {i}",
            }
        )
    return out


def test_assess_evidence_confidence_low_band():
    confidence = rag.assess_evidence_confidence(
        _evidence([0.11, 0.19, 0.15]),
        refuse_threshold=0.30,
        downgrade_threshold=0.55,
    )
    assert confidence["band"] == "low"
    assert confidence["decision"] == "refuse"
    assert confidence["score"] < 0.30


def test_answer_with_evidence_refuse_and_downgrade(monkeypatch):
    monkeypatch.setattr(rag, "get_evidence", lambda *args, **kwargs: _evidence([0.12, 0.18, 0.17]))
    refused = rag.answer_with_evidence(
        "q",
        confidence_policy="auto",
        refuse_threshold=0.30,
        downgrade_threshold=0.55,
    )
    assert refused["decision"] == "refuse"
    assert "Insufficient high-confidence evidence" in refused["answer"]

    monkeypatch.setattr(rag, "get_evidence", lambda *args, **kwargs: _evidence([0.50, 0.43, 0.36]))
    downgraded = rag.answer_with_evidence(
        "q",
        confidence_policy="auto",
        refuse_threshold=0.30,
        downgrade_threshold=0.60,
    )
    assert downgraded["decision"] == "downgrade"
    assert "downgraded to direct, source-supported observations" in downgraded["answer"]
