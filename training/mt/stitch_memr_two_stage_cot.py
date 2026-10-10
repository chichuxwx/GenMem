#!/usr/bin/env python3
"""Recover MemR SIDs and stitch two-stage teacher outputs into ReAct SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from build_cot_generation_inputs import (
    collect_sources,
    flat_record,
    load_merged_bank,
    record_key,
)
from build_meme_react_sft_data import is_valid_sid, write_json_atomic
from build_react_sft_data import SYSTEM_PROMPT, TOOL_SCHEMA_STR


SOURCE_KEYS = ("query", "experience", "trajectory")
STRUCTURAL_TAGS = (
    "<query>",
    "</query>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<answer>",
    "</answer>",
)


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError(f"{path} must contain JSON objects")
    return payload


def canonical_input_hash(record: dict) -> str:
    payload = {key: record[key] for key in SOURCE_KEYS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconstruct_lookup(
    raw_bank: Path,
    merged_bank: Path,
    base_input: list[dict],
    sample_count: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    merged, merged_entries = load_merged_bank(merged_bank)
    representatives, _, _, raw_entries = collect_sources(raw_bank, merged)
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for sid in sorted(set(representatives) & set(merged)):
        source = representatives[sid]
        record = flat_record(
            str(source.get("question") or "").strip(),
            merged[sid]["summary"],
            # collect_sources already proves this is a usable trajectory.
            _trajectory_from_source(source),
        )
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                **record,
                "sid": sid,
                "sample_id": str(source.get("sample_id") or ""),
                "dataset": str(source.get("dataset") or "default"),
            }
        )

    if len(candidates) < sample_count:
        raise ValueError(
            f"requested {sample_count} lookup rows but found only {len(candidates)}"
        )
    rng = random.Random(seed)
    selected = rng.sample(candidates, sample_count)
    rng.shuffle(selected)
    if len(base_input) != sample_count:
        raise ValueError(
            f"base input has {len(base_input)} rows, expected {sample_count}"
        )

    lookup: list[dict] = []
    exact_matches = 0
    for index, (selected_row, base_row) in enumerate(zip(selected, base_input)):
        selected_source = {key: selected_row[key] for key in SOURCE_KEYS}
        base_source = {key: base_row.get(key) for key in SOURCE_KEYS}
        if selected_source != base_source:
            raise ValueError(
                f"deterministic SID replay drift at source_index={index}"
            )
        exact_matches += 1
        lookup.append(
            {
                "source_index": index,
                "sid": selected_row["sid"],
                "dataset": selected_row["dataset"],
                "sample_id": selected_row["sample_id"],
                "input_sha256": canonical_input_hash(base_row),
            }
        )
    stats = {
        "raw_bank_entries": raw_entries,
        "merged_bank_entries": merged_entries,
        "valid_merged_sids": len(merged),
        "eligible_unique_records": len(candidates),
        "base_replay_exact_matches": exact_matches,
    }
    return lookup, stats


def _trajectory_from_source(source: dict) -> str:
    # Local import avoids maintaining a second implementation of trajectory format.
    from build_meme_react_sft_data import trajectory_text

    return trajectory_text(source)


def nonempty_text(row: dict, key: str, source_index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty {key} at source_index={source_index}")
    return value.strip()


def escape_structural_tags(text: str) -> tuple[str, int]:
    escaped = text
    replacements = 0
    for tag in STRUCTURAL_TAGS:
        count = escaped.count(tag)
        if count:
            escaped = escaped.replace(tag, tag.replace("<", "&lt;").replace(">", "&gt;"))
            replacements += count
    return escaped, replacements


def build_sft_records(
    cot_rows: list[dict],
    base_input: list[dict],
    lookup: list[dict],
    expected_cot_rows: int,
    expected_cot_start_index: int,
) -> tuple[list[dict], dict[str, int]]:
    if expected_cot_rows <= 0:
        raise ValueError("expected_cot_rows must be positive")
    if expected_cot_start_index < 0:
        raise ValueError("expected_cot_start_index must be non-negative")
    if len(cot_rows) != expected_cot_rows:
        raise ValueError(
            f"expected {expected_cot_rows} COT rows, got {len(cot_rows)}"
        )
    lookup_by_index = {row["source_index"]: row for row in lookup}
    rows_by_index: dict[int, dict] = {}
    exact_source_matches = 0
    for row_number, row in enumerate(cot_rows):
        source_index = row.get("source_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise ValueError(f"invalid source_index at COT row {row_number}")
        if source_index not in lookup_by_index:
            raise ValueError(f"source_index out of range: {source_index}")
        if source_index in rows_by_index:
            raise ValueError(f"duplicate source_index: {source_index}")
        if row.get("task_type") != "memretriever":
            raise ValueError(f"invalid task_type at source_index={source_index}")
        expected = base_input[source_index]
        actual_source = {key: row.get(key) for key in SOURCE_KEYS}
        expected_source = {key: expected.get(key) for key in SOURCE_KEYS}
        if actual_source != expected_source:
            raise ValueError(f"COT source mismatch at source_index={source_index}")
        exact_source_matches += 1
        nonempty_text(row, "thinking", source_index)
        nonempty_text(row, "rewritten_exp", source_index)
        rows_by_index[source_index] = row

    expected_indices = set(
        range(expected_cot_start_index, expected_cot_start_index + expected_cot_rows)
    )
    actual_indices = set(rows_by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)[:10]
        unexpected = sorted(actual_indices - expected_indices)[:10]
        raise ValueError(
            "COT source_index coverage mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )

    records: list[dict] = []
    trajectory_leaks = 0
    escaped_structural_tags = 0
    for source_index in sorted(rows_by_index):
        row = rows_by_index[source_index]
        meta = lookup_by_index[source_index]
        sid = meta["sid"]
        if not is_valid_sid(sid):
            raise ValueError(f"invalid reconstructed SID at source_index={source_index}")
        thinking, escaped_count = escape_structural_tags(
            nonempty_text(row, "thinking", source_index)
        )
        escaped_structural_tags += escaped_count
        rewritten, escaped_count = escape_structural_tags(
            nonempty_text(row, "rewritten_exp", source_index)
        )
        escaped_structural_tags += escaped_count
        query, escaped_count = escape_structural_tags(str(row["query"]))
        escaped_structural_tags += escaped_count
        experience, escaped_count = escape_structural_tags(str(row["experience"]))
        escaped_structural_tags += escaped_count
        dataset, escaped_count = escape_structural_tags(str(meta["dataset"]))
        escaped_structural_tags += escaped_count
        tool_call = json.dumps(
            {"name": "memory_lookup", "arguments": {"sid_list": [sid]}},
            ensure_ascii=False,
        )
        conversations = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"<query> {query} </query>"},
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{thinking}\n</think>\n"
                    f"<tool_call>\n{tool_call}\n</tool_call>"
                ),
            },
            {
                "from": "observation",
                "value": (
                    f"[SID: {sid}] Dataset: {dataset} | "
                    f"Experience: {experience}"
                ),
            },
            {"from": "gpt", "value": f"<answer>\n{rewritten}\n</answer>"},
        ]
        conversation_text = "\n".join(turn["value"] for turn in conversations)
        trajectory = str(row["trajectory"])
        if trajectory and trajectory in conversation_text:
            trajectory_leaks += 1
        records.append(
            {
                "conversations": conversations,
                "tools": TOOL_SCHEMA_STR,
                "sample_id": meta["sample_id"],
                "dataset": meta["dataset"],
                "sid": sid,
                "source_index": source_index,
            }
        )
    if trajectory_leaks:
        raise ValueError(f"private trajectory leaked into {trajectory_leaks} records")
    return records, {
        "cot_rows": len(cot_rows),
        "output_records": len(records),
        "cot_source_exact_matches": exact_source_matches,
        "duplicate_source_indices": 0,
        "trajectory_leaks": trajectory_leaks,
        "escaped_structural_tags": escaped_structural_tags,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot-input", required=True, type=Path)
    parser.add_argument("--base-input", required=True, type=Path)
    parser.add_argument("--raw-bank", required=True, type=Path)
    parser.add_argument("--merged-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lookup-output", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--base-samples", type=int, default=5_000)
    parser.add_argument("--expected-cot-rows", type=int, default=1_000)
    parser.add_argument("--expected-cot-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_input = load_records(args.base_input)
    cot_rows = load_records(args.cot_input)
    lookup, replay_stats = reconstruct_lookup(
        args.raw_bank,
        args.merged_bank,
        base_input,
        args.base_samples,
        args.seed,
    )
    records, stitch_stats = build_sft_records(
        cot_rows,
        base_input,
        lookup,
        args.expected_cot_rows,
        args.expected_cot_start_index,
    )
    report = {
        "status": "ok",
        **replay_stats,
        **stitch_stats,
        "lookup_rows": len(lookup),
        "unique_output_sids": len({record["sid"] for record in records}),
    }
    write_json_atomic(lookup, args.lookup_output)
    write_json_atomic(records, args.output)
    write_json_atomic(report, args.validation_report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
