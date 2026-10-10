"""Deterministic answer extraction and comparison for OfflineEvo.

The downstream solver is allowed to return plain text, tagged answers, LaTeX,
or small JSON objects.  Scoring the raw strings makes semantically identical
answers such as ``12`` and ``{"answer": 12}`` disagree, which in turn sends a
false failure signal to MemE.  This module first extracts the answer payload
and then applies conservative text, date, and numeric equivalence rules.
"""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from datetime import datetime
from fractions import Fraction
from typing import Any, Iterable


_ANSWER_KEYS = {
    "answer",
    "final_answer",
    "finalanswer",
    "result",
    "prediction",
    "predicted_answer",
    "response",
    "value",
    "solution",
}
_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer|prediction|result)\s*[:：]\s*(.+?)\s*$"
)
_HASH_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _latex_command_values(text: str, command: str) -> list[str]:
    r"""Extract balanced ``\command{...}`` payloads, including nested braces."""
    marker = "\\" + command + "{"
    values: list[str] = []
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            break
        value_start = pos + len(marker)
        depth = 1
        cursor = value_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(text[value_start : cursor - 1])
            start = cursor
        else:
            break
    return values


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json|python|text)?\s*(.*?)\s*```\s*", text, re.DOTALL | re.I)
    return match.group(1).strip() if match else text.strip()


def _structured_value(text: str) -> Any | None:
    candidate = _strip_code_fence(text)
    if not candidate or candidate[0] not in "[{(\"'":
        return None
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError, TypeError):
            return None


def _scalar_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        preferred = [item for key, item in value.items() if str(key).casefold() in _ANSWER_KEYS]
        remaining = [item for key, item in value.items() if str(key).casefold() not in _ANSWER_KEYS]
        for item in preferred + remaining:
            yield from _scalar_values(item)
        # Preserve coordinate-style answers such as {"x": 2, "y": 3}.
        lowered = {str(key).casefold(): item for key, item in value.items()}
        if "x" in lowered and "y" in lowered:
            yield f"({lowered['x']},{lowered['y']})"
    elif isinstance(value, (list, tuple)):
        if value and all(not isinstance(item, (dict, list, tuple)) for item in value):
            yield "(" + ",".join(str(item) for item in value) + ")"
        for item in value:
            yield from _scalar_values(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, bool):
        yield "yes" if value else "no"


def extract_answer_candidates(text: Any, *, reference: bool = False) -> list[str]:
    """Return likely answer payloads ordered from most to least specific.

    References from MATH/GSM8K often contain a complete derivation.  For those
    formats only the final ``\boxed{...}`` or ``#### ...`` payload should be
    compared, rather than treating every number in the derivation as an answer.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    cleaned = _THINK_RE.sub("", raw).strip()
    extracted: list[str] = []

    hash_matches = _HASH_ANSWER_RE.findall(cleaned)
    boxed = _latex_command_values(cleaned, "boxed") + _latex_command_values(cleaned, "fbox")
    tagged = _ANSWER_TAG_RE.findall(cleaned)
    marked = _FINAL_MARKER_RE.findall(cleaned)

    if hash_matches:
        extracted.append(hash_matches[-1])
    if boxed:
        extracted.append(boxed[-1])
    if tagged:
        extracted.extend(reversed(tagged))
    if marked:
        extracted.extend(reversed(marked))

    structured = _structured_value(cleaned)
    if structured is not None:
        extracted.extend(_scalar_values(structured))

    # A reference derivation must not fall back to its full reasoning text.
    if reference and (hash_matches or boxed):
        return _unique(extracted)
    extracted.append(cleaned)
    return _unique(extracted)


def normalize_answer_text(text: Any) -> str:
    """Normalize harmless typography and formatting differences."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", " ").replace("\\!", "")
    value = value.replace("&", " and ")
    value = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"[\$`*_~]", "", value)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"[^\w.%+\-/]+", " ", value, flags=re.UNICODE)
    value = _ARTICLES_RE.sub(" ", value)
    return " ".join(value.split())


def _numeric_value(text: str) -> float | None:
    value = unicodedata.normalize("NFKC", str(text or "")).strip().casefold()
    value = value.replace("\\,", "").replace(",", "")
    value = value.replace("−", "-").replace("$", "")
    value = re.sub(r"^(?:approximately|approx\.?|about|around|~|≈)\s*", "", value)
    value = re.sub(r"\s*(?:miles?|mi|km|kg|g|lbs?|pounds?|seconds?|minutes?|hours?)\.?$", "", value)
    percent = value.endswith("%")
    if percent:
        value = value[:-1].strip()
    try:
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            number = float(Fraction(value.replace(" ", "")))
        elif re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", value):
            number = float(value)
        else:
            return None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return number / 100.0 if percent else number


def _date_value(text: str) -> tuple[int, int, int] | None:
    value = normalize_answer_text(text)
    value = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", value)
    month_pattern = "|".join(_MONTHS)
    patterns = (
        rf"^(?P<month>{month_pattern}) (?P<day>\d{{1,2}}) (?P<year>\d{{4}})$",
        rf"^(?P<day>\d{{1,2}}) (?P<month>{month_pattern}) (?P<year>\d{{4}})$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            try:
                parsed = datetime(
                    int(match.group("year")),
                    _MONTHS[match.group("month")],
                    int(match.group("day")),
                )
            except ValueError:
                return None
            return parsed.year, parsed.month, parsed.day
    return None


def _yes_no_value(text: str) -> bool | None:
    tokens = normalize_answer_text(text).split()
    if not tokens:
        return None
    if tokens[0] in {"yes", "true"}:
        return True
    if tokens[0] in {"no", "false"}:
        return False
    return None


def candidates_match(predicted: str, reference: str) -> bool:
    pred_norm = normalize_answer_text(predicted)
    ref_norm = normalize_answer_text(reference)
    if not pred_norm or not ref_norm:
        return False
    if pred_norm == ref_norm:
        return True

    pred_number = _numeric_value(predicted)
    ref_number = _numeric_value(reference)
    if pred_number is not None and ref_number is not None:
        return math.isclose(pred_number, ref_number, rel_tol=1e-4, abs_tol=1e-6)

    pred_date = _date_value(predicted)
    ref_date = _date_value(reference)
    if pred_date is not None and ref_date is not None:
        return pred_date == ref_date

    ref_yes_no = _yes_no_value(reference)
    if ref_yes_no is not None:
        pred_yes_no = _yes_no_value(predicted)
        if pred_yes_no is not None:
            return pred_yes_no == ref_yes_no

    # Accept an answer embedded in a short explanatory sentence, but avoid the
    # old one-character substring bug (for example answer "1" matching any
    # derivation containing the digit 1).
    compact_ref = ref_norm.replace(" ", "")
    if len(compact_ref) >= 3 and re.search(
        rf"(?<!\w){re.escape(ref_norm)}(?!\w)", pred_norm
    ):
        return True
    # Short benchmark references commonly include a harmless qualifier, for
    # example "documentary film", "New York City", or "Palm Desert,
    # California".  Preserve that established partial-credit behavior only
    # for short references; never apply it to long derivations, where a stray
    # intermediate value must not be accepted as the final answer.
    compact_pred = pred_norm.replace(" ", "")
    if (
        len(ref_norm.split()) <= 5
        and len(compact_pred) >= 3
        and _numeric_value(predicted) is None
        and re.search(rf"(?<!\w){re.escape(pred_norm)}(?!\w)", ref_norm)
    ):
        return True
    return False


def answers_match(predicted: Any, ground_truth: Any) -> bool:
    """Compare solver output with a reference answer conservatively."""
    predictions = extract_answer_candidates(predicted, reference=False)
    references = extract_answer_candidates(ground_truth, reference=True)
    return any(candidates_match(pred, ref) for pred in predictions for ref in references)
