import pandas as pd
import pytest

from evaluation.faithfulness_metrics import (
    compute_faithfulness_report,
    grounding_score,
    hallucination_rate,
    sentence_supported,
    split_sentences,
)

EVIDENCE = (
    "Clip 1: V_12.mp4, similarity score: 0.812, dataset: rlvs, label: violence\n"
    "  Description: rlvs clip labelled 'violence'\n"
    "Clip 2: NV_7.mp4, similarity score: 0.704, dataset: rlvs, label: non_violence\n"
    "  Description: rlvs clip labelled 'non_violence'"
)


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []


def test_supported_sentence_passes():
    ok, tag = sentence_supported("The clip is labelled violence with score 0.812", EVIDENCE)
    assert ok is True
    assert tag == "supported"


def test_unsupported_number_fails():
    ok, tag = sentence_supported("The incident happened at 03:45 near gate 9", EVIDENCE)
    assert ok is False
    assert tag == "number_or_date_mismatch"


def test_unsupported_content_fails():
    ok, tag = sentence_supported("A purple elephant danced on the roof of the shed", EVIDENCE)
    assert ok is False
    assert tag == "misgeneration"


def test_grounding_and_hallucination():
    report = "The evidence contains a clip labelled violence. A purple elephant danced."
    score, flags = grounding_score(report, EVIDENCE)
    assert len(flags) == 2
    assert score == pytest.approx(0.5)
    assert hallucination_rate(report, EVIDENCE) == pytest.approx(0.5)
    assert hallucination_rate("", EVIDENCE) == 0.0


def test_compute_faithfulness_report():
    reports = ["The clip is labelled violence.", "Officers arrived at 05:30."]
    df = compute_faithfulness_report(reports, [EVIDENCE, EVIDENCE])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.loc[0, "hallucination_rate"] < df.loc[1, "hallucination_rate"]
    assert "unsupported_tags" in df.columns
