#!/usr/bin/env python3
"""Build MemEvolver ReAct SFT records from raw and merged memory banks."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path


SID_RE = re.compile(
    r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
)
SID_LIMITS = (48, 16, 8, 8)

SYSTEM_PROMPT = (
    "You are MemEvolver, a memory evolution agent.\n"
    "Given an execution trajectory, you update the memory bank with evolved experience:\n\n"
    "1. <think>: Analyse the trajectory thoroughly — understand what task was attempted, "
    "what knowledge domain it covers, what succeeded and failed, and hypothesise which SID "
    "cluster captures this domain best. Consider the trajectory's implicit lessons.\n\n"
    "2. <tool_call>: Call memory_lookup with your chosen SID to retrieve the existing "
    "experience.\n\n"
    "3. <think>: Compare the trajectory's insights against the retrieved experience — "
    "identify what is already captured, what is new, what is contradicted, and how to merge "
    "them. If no existing experience is found, derive the experience entirely from the "
    "trajectory.\n\n"
    "4. <answer>: Produce an evolved experience that synthesises the old memory with new "
    "insights from the trajectory. The evolved experience should be reusable: general enough "
    "to transfer to similar future tasks, specific enough to be actionable. Do not copy the "
    "trajectory verbatim — distil the transferable lessons."
)

TOOL_SCHEMA_STR = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "memory_lookup",
                "description": (
                    "Look up experience memories using four-level Semantic IDs "
                    "(SIDs), returning the relevant merged experience summaries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sid_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Four-level SID token strings such as "
                                "<SID_L1_5><SID_L2_3><SID_L3_2><SID_L4_1>."
                            ),
                        }
                    },
                    "required": ["sid_list"],
                },
            },
        }
    ],
    ensure_ascii=False,
)


def is_valid_sid(sid: str) -> bool:
    match = SID_RE.fullmatch(sid)
    return match is not None and all(
        int(value) < limit for value, limit in zip(match.groups(), SID_LIMITS)
    )


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream dictionaries from a top-level JSON array."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError("input ended before the top-level JSON array")
            buffer += chunk
            stripped = buffer.lstrip()
            if stripped:
                if stripped[0] != "[":
                    raise ValueError("input must contain a top-level JSON array")
                buffer = stripped[1:]
                break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError("truncated or malformed JSON array")
                buffer += chunk
                continue
            if not isinstance(value, dict):
                raise TypeError("every bank entry must be a JSON object")
            yield value
            buffer = buffer[end:]


def entry_sid(entry: dict) -> str:
    return str(entry.get("sid", {}).get("tokens", "")).strip()


def entry_experience(entry: dict) -> str:
    return str(
        entry.get("experience_summary") or entry.get("experience_text") or ""
    ).strip()


def truncate_middle(text: str, limit: int) -> str:
    marker = "\n[...truncated...]\n"
    if limit <= len(marker) + 2:
        raise ValueError("max_field_chars is too small")
    if len(text) <= limit:
        return text
    remaining = limit - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _reservoir_add(
    reservoir: list[dict], item: dict, eligible_count: int, capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = rng.randrange(eligible_count)
    if replacement < capacity:
        reservoir[replacement] = item


def sample_targets(
    merged_bank: Path,
    max_samples: int,
    empty_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], int, dict[str, int]]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not 0.0 <= empty_ratio <= 1.0:
        raise ValueError("empty_ratio must be between 0 and 1")

    empty_samples = round(max_samples * empty_ratio)
    update_samples = max_samples - empty_samples
    singleton_reservoir: list[dict] = []
    exact_pair_reservoir: list[dict] = []
    larger_group_reservoir: list[dict] = []
    eligible = {"single": 0, "pair": 0, "larger": 0}

    seen_sids: set[str] = set()
    total_entries = 0
    for entry in iter_json_array(merged_bank):
        total_entries += 1
        sid = entry_sid(entry)
        summary = entry_experience(entry)
        source_count = entry.get("merged_from_count")
        if (
            not isinstance(source_count, int)
            or source_count < 1
            or not is_valid_sid(sid)
            or not summary
        ):
            continue
        if sid in seen_sids:
            raise ValueError(f"duplicate merged SID: {sid}")
        seen_sids.add(sid)
        compact = {
            "sid": sid,
            "dataset": str(entry.get("dataset") or "default"),
            "summary": summary,
            "merged_from_count": source_count,
        }
        if source_count == 1:
            eligible["single"] += 1
            _reservoir_add(
                singleton_reservoir,
                compact,
                eligible["single"],
                empty_samples,
                rng,
            )
        elif source_count == 2:
            eligible["pair"] += 1
            _reservoir_add(
                exact_pair_reservoir,
                compact,
                eligible["pair"],
                update_samples,
                rng,
            )
        else:
            eligible["larger"] += 1
            _reservoir_add(
                larger_group_reservoir,
                compact,
                eligible["larger"],
                update_samples,
                rng,
            )

    if len(singleton_reservoir) != empty_samples:
        raise ValueError(
            f"requested {empty_samples} empty-memory samples but found only "
            f"{eligible['single']} singleton SIDs"
        )
    exact_count = min(len(exact_pair_reservoir), update_samples)
    larger_count = update_samples - exact_count
    if len(larger_group_reservoir) < larger_count:
        raise ValueError(
            f"requested {update_samples} update samples but found only "
            f"{eligible['pair'] + eligible['larger']} multi-source SIDs"
        )

    targets = (
        singleton_reservoir
        + exact_pair_reservoir[:exact_count]
        + larger_group_reservoir[:larger_count]
    )
    rng.shuffle(targets)
    return targets, total_entries, eligible


def collect_raw_sources(
    raw_bank: Path, targets: dict[str, dict]
) -> tuple[dict[str, list[dict]], int]:
    sources: dict[str, list[dict]] = defaultdict(list)
    source_counts = defaultdict(int)
    total_entries = 0
    for entry in iter_json_array(raw_bank):
        total_entries += 1
        sid = entry_sid(entry)
        if sid not in targets:
            continue
        source_counts[sid] += 1
        retained = 1 if targets[sid]["merged_from_count"] == 1 else 2
        if len(sources[sid]) < retained:
            compact = {
                key: entry.get(key)
                for key in (
                    "sample_id",
                    "dataset",
                    "question",
                    "prediction",
                    "success",
                    "steps_summary",
                    "trajectory_summary",
                    "reasoning",
                    "experience_summary",
                    "experience_text",
                )
            }
            compact["sid"] = entry.get("sid")
            sources[sid].append(compact)

    for sid, target in sorted(targets.items()):
        found = sources.get(sid, [])
        expected_total = target["merged_from_count"]
        if source_counts[sid] != expected_total:
            raise ValueError(
                f"expected {expected_total} raw sources for {sid}, "
                f"found {source_counts[sid]}"
            )
        retained = 1 if expected_total == 1 else 2
        if len(found) != retained:
            raise ValueError(f"expected {retained} retained sources for {sid}")
        source_ids = [str(entry.get("sample_id") or "").strip() for entry in found]
        if not all(source_ids) or len(set(source_ids)) != retained:
            raise ValueError(f"distinct non-empty sample IDs are required for {sid}")
        if not all(entry_experience(entry) for entry in found):
            raise ValueError(f"retained raw sources need experience text for {sid}")
    return dict(sources), total_entries


def trajectory_text(entry: dict) -> str:
    question = str(entry.get("question") or "").strip()
    trajectory_summary = entry.get("trajectory_summary")
    nested_summary = ""
    if isinstance(trajectory_summary, dict):
        nested_summary = str(trajectory_summary.get("trajectory_summary") or "").strip()
    execution = str(
        entry.get("steps_summary")
        or nested_summary
        or entry.get("reasoning")
        or entry_experience(entry)
    ).strip()
    if not question or not execution:
        raise ValueError("trajectory source requires non-empty question and execution")

    prediction = str(entry.get("prediction") or "").strip() or "(no answer recorded)"
    if entry.get("success") is True:
        outcome = "correct"
    elif entry.get("success") is False:
        outcome = "incorrect"
    else:
        outcome = "outcome unavailable"
    return (
        f"Question: {question}\n"
        f"Execution:\n{execution}\n"
        f"Answer given: {prediction} ({outcome})"
    )


def build_record(
    target: dict,
    sources: list[dict],
    rng: random.Random,
    max_field_chars: int = 1_000,
) -> dict:
    ordered = list(sources)
    rng.shuffle(ordered)
    trajectory_source = ordered[0]
    existing_memory_source = ordered[1] if len(ordered) == 2 else None
    sid = target["sid"]
    dataset = str(trajectory_source.get("dataset") or target["dataset"])
    tool_call = json.dumps(
        {"name": "memory_lookup", "arguments": {"sid_list": [sid]}},
        ensure_ascii=False,
    )
    source_id = str(trajectory_source["sample_id"])
    merged_summary = truncate_middle(target["summary"], max_field_chars)
    if existing_memory_source is None:
        memory_id = ""
        observation = f"[Memory lookup for {sid}]: No matching experience found."
        update_type = "insert_new"
        second_thought = (
            "No existing experience was found for this SID. I will distil the "
            "trajectory into a reusable memory entry."
        )
    else:
        memory_id = str(existing_memory_source["sample_id"])
        memory_dataset = str(existing_memory_source.get("dataset") or "?")
        existing_experience = truncate_middle(
            entry_experience(existing_memory_source), max_field_chars
        )
        observation = (
            f"[SID: {sid}]\n[1] (Dataset: {memory_dataset}) {existing_experience}"
        )
        update_type = "merge_existing"
        second_thought = (
            "The trajectory adds another grounded example to this memory cluster. "
            "I will preserve valid existing lessons, incorporate supported "
            "information from the trajectory, and remove redundancy."
        )

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": (
                    "<trajectory> "
                    f"{truncate_middle(trajectory_text(trajectory_source), max_field_chars)}"
                    " </trajectory>"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "<think>\n"
                    f"This trajectory concerns {dataset} problem solving. I will "
                    "retrieve the existing experience from its relevant memory cluster.\n"
                    "</think>\n"
                    f"<tool_call>\n{tool_call}\n</tool_call>"
                ),
            },
            {
                "from": "observation",
                "value": observation,
            },
            {
                "from": "gpt",
                "value": (
                    "<think>\n"
                    f"{second_thought}\n"
                    "</think>\n"
                    f"<answer>\n{merged_summary}\n</answer>"
                ),
            },
        ],
        "tools": TOOL_SCHEMA_STR,
        "sid": sid,
        "dataset": dataset,
        "trajectory_source_id": source_id,
        "existing_memory_source_id": memory_id,
        "merged_from_count": target["merged_from_count"],
        "update_type": update_type,
    }


def write_json_atomic(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--merged-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=2_000)
    parser.add_argument("--empty-ratio", type=float, default=0.2)
    parser.add_argument("--max-field-chars", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    targets, merged_entries, eligible = sample_targets(
        args.merged_bank, args.max_samples, args.empty_ratio, rng
    )
    targets_by_sid = {target["sid"]: target for target in targets}
    raw_sources, raw_entries = collect_raw_sources(args.input, targets_by_sid)
    records = [
        build_record(
            target,
            raw_sources[target["sid"]],
            rng,
            max_field_chars=args.max_field_chars,
        )
        for target in targets
    ]
    write_json_atomic(records, args.output)
    print(
        f"raw_entries={raw_entries} merged_entries={merged_entries} "
        f"eligible_single_sids={eligible['single']} "
        f"eligible_pair_sids={eligible['pair']} "
        f"eligible_larger_sids={eligible['larger']} "
        f"insert_records={sum(r['update_type'] == 'insert_new' for r in records)} "
        f"update_records={sum(r['update_type'] == 'merge_existing' for r in records)} "
        f"output_records={len(records)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
