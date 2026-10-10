import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_STACKPLANNER_TOOL_LINE_RE = re.compile(
    r"^(?P<prefix>\[tool:[^\]\r\n]+\]\s*)(?P<payload>\{.*\})$",
    flags=re.MULTILINE,
)


def _bounded_tool_result(value: str, max_chars: int) -> str:
    """Keep a head/tail excerpt whose total length never exceeds max_chars."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    # Summary extraction above handles conclusions that were placed in a
    # dedicated field.  For an unstructured fallback, delegated search results
    # normally state the answer first, so spend the full budget on the prefix
    # rather than mixing in a usually unhelpful tail.
    if max_chars == 1:
        return "…"
    return value[: max_chars - 1] + "…"


def _tool_result_summary(value: str) -> str:
    """Prefer a delegated research summary over bulky artifact payloads."""
    text = str(value or "").strip()
    try:
        nested = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        nested = None
    if isinstance(nested, dict):
        for key in ("summary", "answer", "result"):
            candidate = nested.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    # Upstream may already have truncated a nested JSON envelope, leaving it
    # invalid as a whole even though the leading quoted summary is complete.
    quoted_summary = re.search(
        r'["\']summary["\']\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted_summary:
        try:
            candidate = json.loads('"' + quoted_summary.group(1) + '"')
        except json.JSONDecodeError:
            candidate = quoted_summary.group(1)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Some StackPlanner workers return a plain envelope such as
    # ``... summary: decisive fact artifact_content: ...`` rather than JSON.
    match = re.search(
        r"(?:^|\s)summary:\s*(.*?)"
        r"(?=\s+(?:artifact_content|artifact_type|completion_status|stop_reason):|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text


def compact_trajectory_tool_results(trajectory: str, max_result_chars: int) -> str:
    """Bound external StackPlanner tool results without touching memory lookup.

    StackPlanner serialises an external tool event on one line as
    ``[tool:name] {json}``.  Only the top-level ``result`` string is compacted;
    action/error/status fields, model reasoning, the final answer, and XML
    ``<tool_response>`` blocks used by MemE's own ``memory_lookup`` remain
    unchanged.
    """
    text = str(trajectory or "")
    if max_result_chars <= 0:
        return text

    def replace(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group("payload"))
        except (json.JSONDecodeError, TypeError):
            return match.group(0)
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), str):
            return match.group(0)
        result = payload["result"]
        if len(result) <= max_result_chars:
            return match.group(0)
        compacted = _bounded_tool_result(
            _tool_result_summary(result), max_result_chars
        )
        if compacted == result:
            return match.group(0)
        payload["result"] = compacted
        return match.group("prefix") + json.dumps(payload, ensure_ascii=False)

    return _STACKPLANNER_TOOL_LINE_RE.sub(replace, text)


def build_meme_trajectory(
    trajectory: str,
    sample: Mapping[str, Any],
    expose_ground_truth: bool,
) -> str:
    """Return the MemE observation, optionally with training-only supervision."""
    trajectory = str(trajectory or "")
    try:
        tool_result_max_chars = int(
            os.getenv("MEME_TRAJECTORY_TOOL_RESULT_CHARS", "0")
        )
    except ValueError as exc:
        raise ValueError(
            "MEME_TRAJECTORY_TOOL_RESULT_CHARS must be an integer"
        ) from exc
    if tool_result_max_chars > 0:
        trajectory = compact_trajectory_tool_results(
            trajectory, tool_result_max_chars
        )
    if not expose_ground_truth:
        return trajectory

    ground_truth = str(sample.get("ground_truth_answer") or "").strip()
    if not ground_truth:
        raise ValueError(
            "--expose-ground-truth-to-meme requires a non-empty "
            "ground_truth_answer for every evolution sample"
        )
    return (
        f"{trajectory.rstrip()}\n\n"
        "<training_only_ground_truth>\n"
        "The following reference answer is privileged supervision available "
        "during memory evolution only; it is not available at test time. "
        "Use it to diagnose the trajectory and distil a reusable lesson, "
        "rather than copying it verbatim.\n"
        f"Reference answer: {ground_truth}\n"
        "</training_only_ground_truth>"
    )


def _normalise_question(sample: Mapping[str, Any]) -> tuple[str, str]:
    dataset = re.sub(r"\s+", " ", str(sample.get("dataset") or "")).strip().casefold()
    question = re.sub(r"\s+", " ", str(sample.get("question") or "")).strip().casefold()
    return dataset, question


def assert_disjoint_samples(
    evolution_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
) -> None:
    """Reject train/test overlap by either explicit ID or normalised question."""
    evolution_ids = {
        str(sample.get("sample_id")).strip()
        for sample in evolution_samples
        if str(sample.get("sample_id") or "").strip()
    }
    validation_ids = {
        str(sample.get("sample_id")).strip()
        for sample in validation_samples
        if str(sample.get("sample_id") or "").strip()
    }
    shared_ids = evolution_ids & validation_ids

    evolution_questions = {_normalise_question(sample) for sample in evolution_samples}
    validation_questions = {_normalise_question(sample) for sample in validation_samples}
    shared_questions = evolution_questions & validation_questions
    if shared_ids or shared_questions:
        raise ValueError(
            "evolution and held-out samples overlap "
            f"(sample_ids={len(shared_ids)}, questions={len(shared_questions)})"
        )


def summarise_evaluation(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise a frozen held-out evaluation without changing model inputs."""
    rows = list(results)
    if not rows:
        raise ValueError("evaluation requires at least one result")

    per_dataset: dict[str, dict[str, Any]] = {}
    datasets = sorted({str(row.get("dataset") or "unknown") for row in rows})
    for dataset in datasets:
        subset = [row for row in rows if str(row.get("dataset") or "unknown") == dataset]
        per_dataset[dataset] = {
            "samples": len(subset),
            "answer_correct": sum(bool(row.get("sp_success")) for row in subset),
            "answer_accuracy": sum(bool(row.get("sp_success")) for row in subset)
            / len(subset),
        }

    request_successes = sum(
        bool(row.get("sp_request_success", True)) for row in rows
    )
    errors = Counter(
        str(row.get("sp_error") or "")
        for row in rows
        if not bool(row.get("sp_request_success", True))
    )
    answer_correct = sum(bool(row.get("sp_success")) for row in rows)
    return {
        "samples": len(rows),
        "answer_correct": answer_correct,
        "answer_accuracy": answer_correct / len(rows),
        "api_request_successes": request_successes,
        "api_request_success_rate": request_successes / len(rows),
        "api_errors": dict(errors),
        "per_dataset": per_dataset,
    }
