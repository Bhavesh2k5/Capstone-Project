import re

import pandas as pd

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are",
    "was", "were", "with", "for", "by", "that", "this", "it", "as", "be",
    "been", "from", "not", "no", "his", "her", "their",
}


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _content_tokens(text: str):
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 2 and t not in _STOP]


def sentence_supported(sentence: str, evidence_text: str, overlap_threshold: float = 0.5):
    ev_tokens = set(_content_tokens(evidence_text))
    ev_numbers = set(_NUMBER.findall(evidence_text or ""))
    sent_numbers = _NUMBER.findall(sentence or "")
    if sent_numbers and not set(sent_numbers).issubset(ev_numbers):
        return False, "number_or_date_mismatch"
    sent_tokens = _content_tokens(sentence)
    if not sent_tokens:
        return True, "empty"
    overlap = sum(1 for t in sent_tokens if t in ev_tokens) / len(sent_tokens)
    if overlap + 1e-9 >= overlap_threshold:
        return True, "supported"
    return False, "misgeneration"


def grounding_score(report: str, evidence_text: str, overlap_threshold: float = 0.5):
    sentences = split_sentences(report)
    if not sentences:
        return 1.0, []
    flags = [sentence_supported(s, evidence_text, overlap_threshold) for s in sentences]
    supported = sum(1 for ok, _ in flags if ok)
    return supported / len(sentences), flags


def hallucination_rate(report: str, evidence_text: str, overlap_threshold: float = 0.5) -> float:
    score, _ = grounding_score(report, evidence_text, overlap_threshold)
    return 1.0 - score


def bertscore_faithfulness(generated_report: str, evidence_text: str, model_type=None) -> float:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError("bert-score is not installed; pip install bert-score") from exc
    _, _, F1 = bert_score(
        [generated_report], [evidence_text], model_type=model_type, verbose=False
    )
    return float(F1[0])


def compute_faithfulness_report(reports, evidence_texts, use_bertscore=False, overlap_threshold=0.5):
    rows = []
    for i, (report, evidence) in enumerate(zip(reports, evidence_texts)):
        score, flags = grounding_score(report, evidence, overlap_threshold)
        row = {
            "report_id": i,
            "n_sentences": len(flags),
            "n_supported": sum(1 for ok, _ in flags if ok),
            "grounding_score": score,
            "hallucination_rate": 1.0 - score,
            "unsupported_tags": ";".join(tag for ok, tag in flags if not ok),
        }
        if use_bertscore:
            try:
                row["bertscore_f1"] = bertscore_faithfulness(report, evidence)
            except ImportError:
                row["bertscore_f1"] = None
        rows.append(row)
    return pd.DataFrame(rows)
