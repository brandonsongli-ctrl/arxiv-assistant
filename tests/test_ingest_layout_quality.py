from src import ingest


def test_annotate_layout_text_marks_structures():
    text = "\n".join(
        [
            "Figure 1: Mechanism overview.",
            "Table 2: Main regression results.",
            "x_i = y_i + z_i",
            "BidderA    10.2    0.33",
            "A normal sentence.",
        ]
    )
    annotated, stats = ingest._annotate_layout_text(text)

    assert "[FIGURE_CAPTION]" in annotated
    assert "[TABLE_CAPTION]" in annotated
    assert "[FORMULA]" in annotated
    assert "[TABLE]" in annotated
    assert stats["figure_caption_lines"] == 1
    assert stats["table_caption_lines"] == 1
    assert stats["formula_lines"] >= 1
    assert stats["table_lines"] >= 1


def test_score_parse_quality_detects_low_quality():
    poor = ingest._score_parse_quality(page_texts=["", ""], layout_stats={}, ocr_pages=2)
    assert poor["quality_label"] == "low"
    assert 0.0 <= poor["quality_score"] <= 1.0

    good_text = ["This is a long page. " * 120, "Another long page. " * 120]
    better = ingest._score_parse_quality(
        page_texts=good_text,
        layout_stats={"table_lines": 1, "formula_lines": 1, "figure_caption_lines": 1, "table_caption_lines": 0},
        ocr_pages=0,
    )
    assert better["quality_score"] > poor["quality_score"]
