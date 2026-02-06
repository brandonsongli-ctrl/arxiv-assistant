from src import retrieval


def test_auto_tune_identifier_prefers_bm25():
    tuned = retrieval.auto_tune("10.1000/xyz123", n_results=5)
    assert tuned["mode"] == "bm25"
    assert tuned["n_results"] >= 3


def test_auto_tune_short_query_prefers_bm25():
    tuned = retrieval.auto_tune("auction design", n_results=5)
    assert tuned["mode"] == "bm25"


def test_auto_tune_long_question_prefers_hybrid():
    tuned = retrieval.auto_tune("How does robust mechanism design compare to Bayesian persuasion under model misspecification?", n_results=5)
    assert tuned["mode"] == "hybrid"
    assert tuned["n_results"] >= 8
