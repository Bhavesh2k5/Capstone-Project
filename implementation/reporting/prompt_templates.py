from pathlib import Path

REPORT_PROMPT = """You are a surveillance analyst. Based on the following retrieved CCTV evidence clips,
write a concise factual incident report. Only include information directly supported
by the evidence. Do not add speculation.

Evidence:
{evidence}

Query: {query}

Generate a structured incident report with: Incident Type, Time/Location (if available),
Observed Behaviors, Severity Assessment, and Recommended Action."""

UNGROUNDED_PROMPT = """You are a surveillance analyst. Write a concise incident report for the query below.
Do not invent specific details such as names, times or locations.

Query: {query}

Generate a structured incident report with: Incident Type, Observed Behaviors,
Severity Assessment, and Recommended Action."""

MAX_EVIDENCE_CHARS = 6000


def format_evidence(evidence, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    lines = []
    for i, item in enumerate(evidence, 1):
        label = item.get("label", "unknown")
        if isinstance(label, (list, tuple)):
            label = ", ".join(str(x) for x in label)
        desc = item.get("description") or (
            f"{item.get('dataset', 'cctv')} clip labelled '{label}'"
        )
        path = Path(str(item.get("path", "unknown"))).name
        lines.append(
            f"- Clip {i}: {path}, similarity score: {float(item.get('score', 0.0)):.3f}, "
            f"dataset: {item.get('dataset', 'unknown')}, label: {label}\n"
            f"  Description: {desc}"
        )
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n...[truncated]"
    return block


def build_report_prompt(evidence, query: str, grounded: bool = True) -> str:
    if grounded:
        return REPORT_PROMPT.format(evidence=format_evidence(evidence), query=query)
    return UNGROUNDED_PROMPT.format(query=query)


def evidence_to_text(evidence) -> str:
    return format_evidence(evidence)
