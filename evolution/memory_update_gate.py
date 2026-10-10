"""Safety gates for writes produced by MemE."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from answer_scoring import extract_answer_candidates, normalize_answer_text


_ANSWER_HEADER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\*{1,2})?"
    r"(ground\s*truth|reference\s*answer|gold\s*answer|correct\s*answer|"
    r"final\s*answer|answer\s*given|predicted\s*answer|prediction|answer|this\s*run)"
    r"(?:\*{1,2})?\s*[:：=]"
)
_OUTCOME_METADATA_RE = re.compile(
    r"(?i)(?:<training_only_ground_truth>|\bevaluation\s*=|\brun_id\s*=)"
)
# External trajectories are intentionally supplied without labels.  A model
# can nevertheless reproduce benchmark scaffolding from its prior training or
# from an unclean tool response.  These phrases are never useful in a reusable
# memory and are therefore a hard write barrier for the external-bootstrap
# path (the normal RL/offline-evo gate below remains unchanged).
_EXTERNAL_OUTCOME_RE = re.compile(
    r"(?i)(?:\bground\s*truth\b|\breference\s+answer\b|"
    r"\bfinal\s+answer\b|\bcorrect\s+answer\b|\bgold\s+answer\b|"
    r"\bpredicted\s+answer\b|\bprediction\b|\bevaluation\b|"
    r"\bthis\s+run\b|\brun[_ -]?id\b|\bscore\s*[:=])"
)
_EXTERNAL_TEMPLATE_RE = re.compile(
    r"(?i)(?:structured\s*reusable\s*procedure|sample[_ -]?id\b|"
    r"task\s+signature\s*:|\btarget\s*=|\bgroup\s*=|\bdataset\s*=|"
    r"<query>|<table>|<answer>|</answer>|\.\.\.\s*etc\.?)"
)
_EXTERNAL_CONCRETE_ID_RE = re.compile(
    r"(?i)\b[A-Z]{1,4}\d{2,}[A-Z0-9-]{3,}\b|"
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b"
)
_DIRECT_ANSWER_RE = re.compile(
    r"(?is)^\s*(?:the\s+)?(?:final\s+)?answer\s+is\b|"
    r"^\s*(?:it\s+is|it's)\b|"
    r"^\s*(?:yes|no|true|false)\s*[.!]?\s*$"
)


def _distinctive_answer(candidate: str) -> bool:
    normalized = normalize_answer_text(candidate)
    if not normalized or normalized in {"yes", "no", "true", "false"}:
        return False
    compact = normalized.replace(" ", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        # Single digits appear naturally in numbered reusable recipes.  They
        # are only rejected when an explicit answer header is present.
        return len(compact.lstrip("+-")) >= 2 or "." in compact
    return len(compact) >= 4


def _contains_candidate(text: str, candidate: str) -> bool:
    haystack = normalize_answer_text(text)
    needle = normalize_answer_text(candidate)
    if not haystack or not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _degenerate_repetition(text: str) -> bool:
    """Catch decoder loops before they are persisted as external skills."""
    # These loops are usually visible as ``N: N: ...`` or ``score=1``
    # repeated hundreds of times.  Common prose stopwords are ignored so a
    # legitimate procedural memory is not rejected merely for saying "the".
    if re.search(r"(?:\b[A-Za-z]\s*:\s*){8,}", text):
        return True
    # Some decoder failures contain one short digit/punctuation pattern with
    # no whitespace (for example ``660066...``), so the token-level test below
    # cannot see the repetition.  Keep this conservative: require a long
    # string and a very small alphabet or an exact repeated chunk.
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) >= 160:
        if len(set(compact)) <= 14:
            return True
        for width in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32):
            if len(compact) < width * 8:
                continue
            grams = [compact[i : i + width]
                     for i in range(len(compact) - width + 1)]
            repeated = Counter(grams).most_common(1)[0][1]
            if repeated >= 8 and repeated / len(grams) >= 0.06:
                return True
    alnum = re.sub(r"[^a-z0-9]", "", compact)
    if len(alnum) >= 160:
        digits = sum(ch.isdigit() for ch in alnum)
        if digits / len(alnum) >= 0.55 and len(set(alnum)) <= 18:
            return True
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_=-]*|\d+", text.casefold())
    if len(tokens) < 40:
        return False
    stop = {"a", "an", "and", "as", "at", "be", "by", "for", "from", "if",
            "in", "is", "it", "of", "on", "or", "the", "to", "use", "with"}
    meaningful = [token for token in tokens if token not in stop]
    if not meaningful:
        return False
    counts = Counter(meaningful)
    most_common = counts.most_common(1)[0][1]
    if most_common >= 12 and most_common / len(meaningful) >= 0.25:
        return True
    for width in (2, 3, 4):
        grams = [tuple(meaningful[i : i + width]) for i in range(len(meaningful) - width + 1)]
        if grams:
            repeated = Counter(grams).most_common(1)[0][1]
            if repeated >= 8 and repeated / len(grams) >= 0.12:
                return True
    return False


def memory_update_rejection_reason(
    evolved_experience: Any,
    *,
    sp_answer: Any = "",
    ground_truth_answer: Any = "",
    strict_external: bool = False,
) -> str:
    """Return a non-empty reason when a generated memory must not be written.

    A reusable memory may describe a solving strategy, but it must not contain
    the sample's concrete answer or answer/outcome metadata.  The gate checks
    both explicit answer headings and distinctive answer values, including
    values nested inside JSON solver outputs.
    """
    text = str(evolved_experience or "").strip()
    if not text:
        return "empty_evolved_experience"

    # External bootstrap trajectories are deliberately label-free.  Since no
    # GT is available to compare against, require a non-trivial reusable skill
    # and reject the common short/direct-answer forms before they can enter
    # the bank.  This is an additional write gate, not a change to the RL
    # prompt or to normal MemE evaluation.
    if strict_external:
        # The external bootstrap should write compact, reusable skills rather
        # than an unbounded continuation of the source trajectory.  This also
        # prevents a long decoder loop from consuming the whole memory entry.
        if len(text) > 12000:
            return "external_skill_too_long"
        if _EXTERNAL_OUTCOME_RE.search(text):
            return "external_outcome_metadata"
        if _EXTERNAL_TEMPLATE_RE.search(text):
            return "external_template_artifact"
        concrete_hits = _EXTERNAL_CONCRETE_ID_RE.findall(text)
        if len(concrete_hits) >= 2:
            return "external_concrete_instance_details"
        if "<answer>" in text.casefold() or "</answer>" in text.casefold():
            return "external_structural_artifact"
        if len(text) < 200:
            return "external_skill_too_short_direct_answer_risk"
        if _DIRECT_ANSWER_RE.search(text):
            return "external_direct_answer_form"
        if _degenerate_repetition(text):
            return "external_degenerate_repetition"

    header = _ANSWER_HEADER_RE.search(text)
    if header:
        label = re.sub(r"\s+", "_", header.group(1).casefold())
        return f"explicit_answer_header:{label}"
    if _OUTCOME_METADATA_RE.search(text):
        return "sample_outcome_metadata"

    sources = (
        ("solver_answer", sp_answer, False),
        ("ground_truth_answer", ground_truth_answer, True),
    )
    for source, value, reference in sources:
        for candidate in extract_answer_candidates(value, reference=reference):
            if _distinctive_answer(candidate) and _contains_candidate(text, candidate):
                return f"contains_{source}"
    return ""
