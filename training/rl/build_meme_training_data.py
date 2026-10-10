#!/usr/bin/env python3
"""Convert stable failed DeerFlow runs into true MemE RL prompts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


MEME_SYSTEM_PROMPT = """You are MemEvolver, a memory evolution agent. Given a failed execution trajectory, update the memory bank with evolved experience using this exact ReAct order. (1) In a first <think>...</think>, analyse the attempted task, its knowledge domain, what failed, and which SID cluster best captures the transferable lesson. (2) Select exactly one valid four-layer Semantic ID in the form <SID_L1_X><SID_L2_Y><SID_L3_Z><SID_L4_W>, then call memory_lookup exactly once with a sid_list containing only that SID. (3) After reading the tool response, use a second <think>...</think> to compare the failed trajectory with the retrieved experience: identify what is already captured, what is missing or contradicted, and how they should be merged. If no existing experience is found, derive the reusable lesson from the trajectory. (4) Output exactly one non-empty <answer>...</answer> containing the evolved experience. Preserve useful prior guidance, correct the failure, and add actionable decision rules and verification checks. Do not merely answer the original task, copy the old memory verbatim, emit multiple SIDs, or call the tool more than once."""

_REDACTION = "[REDACTED_GROUND_TRUTH]"
_HINT_FIELDS = {
    "privileged_prompt",
    "rlsd_action_prompt",
    "rlsd_answer_prompt",
    "privileged_prompt_tokens",
    "privileged_observation_truncated",
}
_DEFAULT_TOOL_RESULT_TOKEN_CAP = 10


def clean_question(dataset: str, question: str) -> str:
    value = str(question or "").strip()
    if str(dataset).lower() == "alfworld":
        value = value.split("\n\nEnvironment/task excerpt:", 1)[0].strip()
    return value


def candidate_key(dataset: str, question: str) -> tuple[str, str]:
    return str(dataset or "").strip().lower(), clean_question(dataset, question)


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_healthy_failure(record: dict) -> bool:
    return (
        not str(record.get("error") or "").strip()
        and bool(record.get("has_observable_rollout", True))
        and not bool(record.get("success"))
        and _as_float(record.get("score")) <= 0.0
    )


def _quality_key(record: dict) -> tuple[int, int, int, int, str]:
    prediction = str(record.get("prediction") or record.get("raw_response") or "").strip()
    steps = str(record.get("steps_summary") or "").strip()
    reasoning = str(record.get("reasoning") or "").strip()
    raw_response = str(record.get("raw_response") or "").strip()
    return (
        int(bool(prediction)),
        len(steps),
        len(reasoning),
        len(raw_response),
        str(record.get("sample_id") or record.get("run_id") or ""),
    )


def select_failure_run(records: Iterable[dict]) -> dict:
    eligible = [record for record in records if is_healthy_failure(record)]
    if not eligible:
        raise ValueError("no healthy failed trajectory")
    return max(eligible, key=_quality_key)


def _token_ids(tokenizer, text: str) -> list:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _truncate_tokens(tokenizer, text: str, budget: int) -> str:
    ids = _token_ids(tokenizer, text)
    if len(ids) <= budget:
        return text.strip()
    if budget <= 0:
        return ""
    if budget < 8:
        return tokenizer.decode(ids[:budget], skip_special_tokens=False).strip()
    marker = "\n...[middle of execution truncated]...\n"
    marker_ids = _token_ids(tokenizer, marker)
    content_budget = max(1, budget - len(marker_ids))
    head_size = max(1, int(content_budget * 0.65))
    tail_size = max(0, content_budget - head_size)
    kept = ids[:head_size] + marker_ids + (ids[-tail_size:] if tail_size else [])
    return tokenizer.decode(kept[:budget], skip_special_tokens=False).strip()


def _redact_aliases(text: str, aliases: Iterable[object]) -> tuple[str, int]:
    output = text
    count = 0
    meaningful = sorted(
        {str(alias).strip() for alias in aliases if len(str(alias).strip()) >= 4},
        key=len,
        reverse=True,
    )
    for alias in meaningful:
        output, replacements = re.subn(re.escape(alias), _REDACTION, output, flags=re.IGNORECASE)
        count += replacements
    return output, count


def _execution_text(run: dict) -> str:
    for field in ("steps_summary", "reasoning", "raw_response"):
        value = str(run.get(field) or "").strip()
        if value:
            return value
    return "No observable execution text was retained."


def _truncate_head_tokens(tokenizer, text: str, token_cap: int) -> str:
    if token_cap < 0:
        raise ValueError("tool result token cap must be non-negative")
    ids = _token_ids(tokenizer, text)
    if len(ids) <= token_cap:
        return text
    return tokenizer.decode(ids[:token_cap], skip_special_tokens=False).strip()


def _partial_tool_result_suffix(execution: str, content: str) -> tuple[int, str] | None:
    """Find a structured tool result truncated only by the source execution cap."""
    minimum_prefix_chars = 32
    if len(content) < minimum_prefix_chars:
        return None
    probe = content[:minimum_prefix_chars]
    start = execution.rfind(probe)
    if start < 0:
        return None
    retained = execution[start:]
    if not minimum_prefix_chars <= len(retained) < len(content):
        return None
    if not content.startswith(retained):
        return None
    return start, retained


def _outer_whitespace_tool_result_span(
    execution: str, content: str
) -> tuple[int, int, str] | None:
    stripped = content.strip()
    if not stripped or stripped == content:
        return None
    start = execution.find(stripped)
    while start >= 0:
        end = start + len(stripped)
        left_boundary = start == 0 or execution[start - 1] == "\n"
        right_boundary = end == len(execution) or execution[end] == "\n"
        if left_boundary and right_boundary:
            return start, end, stripped
        start = execution.find(stripped, start + 1)
    return None


def compact_structured_tool_results(
    run: dict,
    tokenizer,
    token_cap: int = _DEFAULT_TOOL_RESULT_TOKEN_CAP,
) -> tuple[str, dict]:
    """Cap only tool observations identified by the structured source log."""
    execution = _execution_text(run)
    log = run.get("memory_stack_log")
    stats = {
        "structured_log_available": isinstance(log, list),
        "tool_results": 0,
        "tool_results_aligned": 0,
        "tool_results_truncated": 0,
        "tool_results_partial_prefix": 0,
        "tool_results_whitespace_aligned": 0,
        "tool_results_unaligned": 0,
    }
    if not isinstance(log, list):
        return execution, stats

    for entry in log:
        if not isinstance(entry, dict) or str(entry.get("action") or "").lower() != "tool":
            continue
        content = str(entry.get("content") or "")
        if not content:
            continue
        stats["tool_results"] += 1
        if content in execution:
            compacted = _truncate_head_tokens(tokenizer, content, token_cap)
            execution = execution.replace(content, compacted, 1)
            retained = content
        else:
            whitespace_span = _outer_whitespace_tool_result_span(execution, content)
            if whitespace_span is not None:
                start, end, retained = whitespace_span
                compacted = _truncate_head_tokens(tokenizer, retained, token_cap)
                execution = execution[:start] + compacted + execution[end:]
                stats["tool_results_whitespace_aligned"] += 1
            else:
                partial = _partial_tool_result_suffix(execution, content)
                if partial is None:
                    stats["tool_results_unaligned"] += 1
                    continue
                start, retained = partial
                compacted = _truncate_head_tokens(tokenizer, retained, token_cap)
                execution = execution[:start] + compacted
                stats["tool_results_partial_prefix"] += 1
        stats["tool_results_aligned"] += 1
        if compacted != retained:
            stats["tool_results_truncated"] += 1
    return execution, stats


def render_trajectory(
    question: str,
    run: dict,
    tokenizer,
    execution_token_budget: int,
    aliases: Iterable[object] = (),
    tool_result_token_cap: int = _DEFAULT_TOOL_RESULT_TOKEN_CAP,
) -> str:
    execution, _ = compact_structured_tool_results(
        run,
        tokenizer,
        token_cap=tool_result_token_cap,
    )
    execution, _ = _redact_aliases(execution, aliases)
    prediction = str(run.get("prediction") or run.get("raw_response") or "No answer produced.").strip()
    prediction, _ = _redact_aliases(prediction, aliases)
    truncated = _truncate_tokens(tokenizer, execution, execution_token_budget)
    return (
        f"Original task: {question.strip()}\n\n"
        f"Failed execution:\n{truncated}\n\n"
        f"Failed answer: {prediction}\n"
        "Outcome: historical DeerFlow score=0"
    )


def _dataset_of(row: dict) -> str:
    extra = row.get("extra_info") or {}
    dataset = str(extra.get("dataset") or (row.get("metadata") or {}).get("dataset") or "")
    if not dataset:
        dataset = str(row.get("data_source") or "").removeprefix("mem_retriever_")
    return dataset.lower()


def _question_of(row: dict) -> str:
    question = str((row.get("extra_info") or {}).get("question") or "").strip()
    if not question:
        raise ValueError("row has no original question")
    return clean_question(_dataset_of(row), question)


def _aliases_of(row: dict) -> list[str]:
    ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    aliases = list(ground_truth.get("target_answer_aliases") or [])
    answer = str(ground_truth.get("target_answer") or "").strip()
    if answer and answer not in aliases:
        aliases.append(answer)
    return aliases


def convert_row(
    row: dict,
    run: dict,
    tokenizer,
    max_prompt_tokens: int = 2048,
    tool_result_token_cap: int = _DEFAULT_TOOL_RESULT_TOKEN_CAP,
) -> dict:
    output = copy.deepcopy(row)
    dataset = _dataset_of(row)
    question = _question_of(row)
    aliases = _aliases_of(row)
    fixed_user = "<trajectory>\n{trajectory}\n</trajectory>"
    empty_trajectory = render_trajectory(
        question,
        run,
        tokenizer,
        0,
        aliases,
        tool_result_token_cap=tool_result_token_cap,
    )
    fixed_prompt = MEME_SYSTEM_PROMPT + "\n" + fixed_user.format(trajectory=empty_trajectory)
    fixed_tokens = len(_token_ids(tokenizer, fixed_prompt))
    if fixed_tokens > max_prompt_tokens:
        raise ValueError(f"fixed MemE prompt exceeds token budget: {fixed_tokens}>{max_prompt_tokens}")
    trajectory = render_trajectory(
        question,
        run,
        tokenizer,
        max(0, max_prompt_tokens - fixed_tokens),
        aliases,
        tool_result_token_cap=tool_result_token_cap,
    )
    prompt = [
        {"role": "system", "content": MEME_SYSTEM_PROMPT},
        {"role": "user", "content": fixed_user.format(trajectory=trajectory)},
    ]
    prompt_text = "\n".join(message["content"] for message in prompt)
    while len(_token_ids(tokenizer, prompt_text)) > max_prompt_tokens:
        overflow = len(_token_ids(tokenizer, prompt_text)) - max_prompt_tokens
        current_execution = max(0, max_prompt_tokens - fixed_tokens - overflow - 2)
        trajectory = render_trajectory(
            question,
            run,
            tokenizer,
            current_execution,
            aliases,
            tool_result_token_cap=tool_result_token_cap,
        )
        prompt[1]["content"] = fixed_user.format(trajectory=trajectory)
        prompt_text = "\n".join(message["content"] for message in prompt)
        if current_execution == 0 and len(_token_ids(tokenizer, prompt_text)) > max_prompt_tokens:
            raise ValueError("unable to fit MemE prompt into token budget")

    output["data_source"] = f"mem_evolver_{dataset}"
    output["ability"] = "memory_evolution"
    output["prompt"] = prompt
    ground_truth = copy.deepcopy((output.get("reward_model") or {}).get("ground_truth") or {})
    ground_truth["original_query"] = question
    ground_truth["trajectory"] = trajectory
    ground_truth["old_experience"] = str(ground_truth.get("experience_text") or "")
    output["reward_model"] = {"ground_truth": ground_truth}

    extra = copy.deepcopy(output.get("extra_info") or {})
    for field in _HINT_FIELDS:
        extra.pop(field, None)
    extra.update(
        {
            "dataset": dataset,
            "question": question,
            "trajectory": trajectory,
            "source_run_id": str(run.get("sample_id") or run.get("run_id") or ""),
            "source_history": str(run.get("_history_source") or ""),
            "source_line_number": int(run.get("_line_number") or 0),
            "historical_outcome": 0.0,
            "need_tools_kwargs": True,
            "trajectory_tool_result_token_cap": int(tool_result_token_cap),
        }
    )
    _, compaction_stats = compact_structured_tool_results(
        run,
        tokenizer,
        token_cap=tool_result_token_cap,
    )
    extra["trajectory_tool_compaction"] = compaction_stats
    create_kwargs = {
        "data_source": output["data_source"],
        "question": question,
        "ground_truth": ground_truth,
    }
    extra["tools_kwargs"] = {"memory_lookup": {"create_kwargs": create_kwargs}}
    output["extra_info"] = extra
    return output


def _load_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _scan_best_runs(history_paths: list[Path], keys: set[tuple[str, str]]):
    best: dict[tuple[str, str], dict] = {}
    source_lines = Counter()
    matched_lines = Counter()
    eligible_lines = Counter()
    for path in history_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                source_lines[str(path)] += 1
                record = json.loads(line)
                key = candidate_key(record.get("dataset", ""), record.get("question", ""))
                if key not in keys:
                    continue
                matched_lines[str(path)] += 1
                if not is_healthy_failure(record):
                    continue
                eligible_lines[str(path)] += 1
                record["_history_source"] = str(path)
                record["_line_number"] = line_number
                previous = best.get(key)
                if previous is None or _quality_key(record) > _quality_key(previous):
                    best[key] = record
    return best, dict(source_lines), dict(matched_lines), dict(eligible_lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _auto_tokenizer_class():
    from transformers import AutoTokenizer

    return AutoTokenizer


def load_tokenizer(path: Path):
    """Load checkpoints written with either list- or dict-style special tokens."""
    path = Path(path)
    kwargs: dict[str, object] = {"local_files_only": True}
    config_path = path / "tokenizer_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        extra = config.get("extra_special_tokens")
        if isinstance(extra, list):
            kwargs["extra_special_tokens"] = {str(token): str(token) for token in extra}
    return _auto_tokenizer_class().from_pretrained(path, **kwargs)


def build_dataset(
    train_path: Path,
    val_path: Path,
    history_paths: list[Path],
    output_dir: Path,
    tokenizer,
    expected_train_rows: int = 2702,
    expected_val_rows: int = 307,
    max_prompt_tokens: int = 4096,
    tool_result_token_cap: int = _DEFAULT_TOOL_RESULT_TOKEN_CAP,
) -> dict:
    train_path, val_path = Path(train_path), Path(val_path)
    history_paths = [Path(path) for path in history_paths]
    output_dir = Path(output_dir)
    train_rows, val_rows = _load_rows(train_path), _load_rows(val_path)
    if len(train_rows) != expected_train_rows or len(val_rows) != expected_val_rows:
        raise ValueError(
            f"unexpected hard split sizes: train={len(train_rows)} val={len(val_rows)}; "
            f"expected {expected_train_rows}/{expected_val_rows}"
        )
    indexed: list[tuple[str, int, dict, tuple[str, str]]] = []
    for split, rows in (("train", train_rows), ("hard_val", val_rows)):
        for index, row in enumerate(rows):
            key = candidate_key(_dataset_of(row), _question_of(row))
            indexed.append((split, index, row, key))
    best, source_lines, matched_lines, eligible_lines = _scan_best_runs(history_paths, {item[3] for item in indexed})
    unmatched = [item for item in indexed if item[3] not in best]
    if unmatched:
        preview = [f"{split}[{index}] {key!r}" for split, index, _, key in unmatched[:10]]
        raise ValueError(f"unmatched hard rows: {len(unmatched)}; examples={preview}")

    converted = {"train": [], "hard_val": []}
    prompt_lengths: list[int] = []
    redaction_count = 0
    dataset_counts = {"train": Counter(), "hard_val": Counter()}
    compaction_totals = Counter()
    for split, _, row, key in indexed:
        converted_row = convert_row(
            row,
            best[key],
            tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            tool_result_token_cap=tool_result_token_cap,
        )
        converted[split].append(converted_row)
        dataset_counts[split][_dataset_of(row)] += 1
        prompt_text = "\n".join(message["content"] for message in converted_row["prompt"])
        prompt_lengths.append(len(_token_ids(tokenizer, prompt_text)))
        redaction_count += prompt_text.count(_REDACTION)
        compaction = converted_row["extra_info"]["trajectory_tool_compaction"]
        compaction_totals["rows_with_structured_log"] += int(
            compaction["structured_log_available"]
        )
        for field in (
            "tool_results",
            "tool_results_aligned",
            "tool_results_truncated",
            "tool_results_partial_prefix",
            "tool_results_whitespace_aligned",
            "tool_results_unaligned",
        ):
            compaction_totals[field] += int(compaction[field])

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(converted["train"], output_dir / "train.parquet")
    _atomic_write_parquet(converted["hard_val"], output_dir / "hard_val.parquet")
    readme = """# MemE 历史稳定失败训练数据

本目录由 `scripts/build_meme_training_data.py` 自动生成。每条样本输入一段历史 DeerFlow 失败轨迹，要求 MemE 恰好检索一次四层 SID 对应的旧经验，并在 `<answer>` 中输出融合失败教训后的可复用新经验。原题答案只保存在 reward/tool 元数据中，不作为 actor prompt 的新增信息。

- `train.parquet`：继承 hard train，2,702 条。
- `hard_val.parquet`：继承 hard val，307 条。
- `manifest.json`：关联率、来源分布、prompt token 统计及 SHA256。
"""
    _atomic_write_text(output_dir / "README_中文.md", readme)
    manifest = {
        "schema_version": 1,
        "semantics": "failed trajectory -> one four-layer SID lookup -> one reusable evolved experience",
        "inputs": {
            "train": str(train_path),
            "hard_val": str(val_path),
            "history": [str(path) for path in history_paths],
            "source_lines": source_lines,
            "matched_lines": matched_lines,
            "eligible_failure_lines": eligible_lines,
        },
        "join": {"candidate_rows": len(indexed), "matched_rows": len(indexed), "unmatched_rows": 0},
        "prompt": {
            "max_allowed_tokens": max_prompt_tokens,
            "min_tokens": min(prompt_lengths),
            "max_tokens": max(prompt_lengths),
            "mean_tokens": round(sum(prompt_lengths) / len(prompt_lengths), 2),
            "ground_truth_redactions": redaction_count,
            "tool_result_token_cap": int(tool_result_token_cap),
            "tool_compaction": dict(sorted(compaction_totals.items())),
        },
        "outputs": {
            "train_rows": len(converted["train"]),
            "hard_val_rows": len(converted["hard_val"]),
            "train_by_dataset": dict(sorted(dataset_counts["train"].items())),
            "hard_val_by_dataset": dict(sorted(dataset_counts["hard_val"].items())),
        },
    }
    for name in ("train.parquet", "hard_val.parquet", "README_中文.md"):
        manifest["outputs"][f"{name}_sha256"] = _sha256(output_dir / name)
    _atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest["outputs"]["manifest.json_sha256"] = _sha256(output_dir / "manifest.json")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=2702)
    parser.add_argument("--expected-val-rows", type=int, default=307)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--tool-result-token-cap",
        type=int,
        default=_DEFAULT_TOOL_RESULT_TOKEN_CAP,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer)
    manifest = build_dataset(
        args.train,
        args.val,
        args.history,
        args.output_dir,
        tokenizer,
        args.expected_train_rows,
        args.expected_val_rows,
        args.max_prompt_tokens,
        args.tool_result_token_cap,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
