#!/usr/bin/env python3
"""Stitch MemEvolver teacher thinking into grounded five-turn ReAct SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_cot_generation_inputs import (
    collect_sources,
    is_clean_meme_record,
    load_merged_bank,
    meme_record,
    pair_record,
    record_key,
)
from build_meme_react_sft_data import (
    SYSTEM_PROMPT,
    TOOL_SCHEMA_STR,
    entry_experience,
    is_valid_sid,
    trajectory_text,
    write_json_atomic,
)


SOURCE_KEYS = ("query", "experience", "trajectory", "evo_exp")
STRUCTURAL_TAGS = (
    "<query>",
    "</query>",
    "<trajectory>",
    "</trajectory>",
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


def parse_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    indices: set[int] = set()
    for item in value.split(","):
        index = int(item.strip())
        if index < 0:
            raise ValueError("indices must be non-negative")
        indices.add(index)
    return indices


def parse_source_mismatches(value: str) -> dict[int, set[str]]:
    if not value.strip():
        return {}
    mismatches: dict[int, set[str]] = defaultdict(set)
    for item in value.split(","):
        index_text, separator, field = item.strip().partition(":")
        if not separator or field not in SOURCE_KEYS:
            raise ValueError(
                "source mismatches must use INDEX:FIELD with FIELD in "
                f"{SOURCE_KEYS}"
            )
        index = int(index_text)
        if index < 0:
            raise ValueError("source mismatch indices must be non-negative")
        mismatches[index].add(field)
    return dict(mismatches)


def source_hash(row: dict) -> str:
    payload = {key: row[key] for key in SOURCE_KEYS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_with_meta(
    record: dict,
    sid: str,
    trajectory_source: dict,
    existing_source: dict | None,
    merged_from_count: int,
) -> dict:
    return {
        **record,
        "sid": sid,
        "dataset": str(trajectory_source.get("dataset") or "default"),
        "trajectory_source_id": str(trajectory_source.get("sample_id") or ""),
        "existing_memory_source_id": (
            str(existing_source.get("sample_id") or "") if existing_source else ""
        ),
        "merged_from_count": merged_from_count,
        "update_type": "merge_existing" if existing_source else "insert_new",
    }


def replay_meme_records(
    raw_bank: Path,
    merged_bank: Path,
    merge_count: int,
    insert_count: int,
    base_seed: int,
) -> tuple[list[dict], dict[str, int]]:
    merged, merged_entries = load_merged_bank(merged_bank)
    _, sources, raw_counts, raw_entries = collect_sources(raw_bank, merged)
    rng = random.Random(base_seed + 1)

    pair_sids = sorted(
        sid
        for sid, target in merged.items()
        if target["merged_from_count"] == 2
        and raw_counts.get(sid) == 2
        and len(sources.get(sid, [])) == 2
        and len(
            {
                str(source.get("sample_id") or "").strip()
                for source in sources[sid]
            }
        )
        == 2
    )
    singleton_sids = sorted(
        sid
        for sid, target in merged.items()
        if target["merged_from_count"] == 1
        and raw_counts.get(sid) == 1
        and len(sources.get(sid, [])) == 1
    )

    rng.shuffle(pair_sids)
    primary: list[dict] = []
    secondary: list[dict] = []
    pending_secondary: list[dict] = []
    seen_pair: set[tuple[str, str, str]] = set()
    for sid in pair_sids:
        orientation = rng.randrange(2)
        sid_sources = sources[sid]
        evo_exp = merged[sid]["summary"]
        oriented: list[dict] = []
        for trajectory_index in (orientation, 1 - orientation):
            record = pair_record(sid_sources, trajectory_index, evo_exp)
            if not is_clean_meme_record(record):
                continue
            oriented.append(
                candidate_with_meta(
                    record,
                    sid,
                    sid_sources[trajectory_index],
                    sid_sources[1 - trajectory_index],
                    2,
                )
            )
        chosen_index = None
        for index, candidate in enumerate(oriented):
            key = record_key(candidate)
            if key not in seen_pair:
                primary.append(candidate)
                seen_pair.add(key)
                chosen_index = index
                break
        if chosen_index is not None:
            pending_secondary.extend(
                candidate
                for index, candidate in enumerate(oriented)
                if index != chosen_index
            )

    for candidate in pending_secondary:
        key = record_key(candidate)
        if key not in seen_pair:
            secondary.append(candidate)
            seen_pair.add(key)
    all_pair_orientations = primary + secondary
    if merge_count > len(all_pair_orientations):
        raise ValueError(
            f"requested {merge_count} merge rows but found "
            f"{len(all_pair_orientations)}"
        )
    primary_count = min(merge_count, len(primary))
    selected = primary[:primary_count]
    selected.extend(secondary[: merge_count - primary_count])

    rng.shuffle(singleton_sids)
    inserts: list[dict] = []
    seen_insert: set[tuple[str, str, str]] = set()
    for sid in singleton_sids:
        source = sources[sid][0]
        record = meme_record(
            str(source.get("question") or "").strip(),
            "",
            trajectory_text(source),
            merged[sid]["summary"],
        )
        if not is_clean_meme_record(record):
            continue
        key = record_key(record)
        if key in seen_insert:
            continue
        inserts.append(candidate_with_meta(record, sid, source, None, 1))
        seen_insert.add(key)
    if insert_count > len(inserts):
        raise ValueError(
            f"requested {insert_count} insert rows but found {len(inserts)}"
        )
    selected.extend(inserts[:insert_count])
    rng.shuffle(selected)
    return selected, {
        "raw_bank_entries": raw_entries,
        "merged_bank_entries": merged_entries,
        "valid_merged_sids": len(merged),
        "eligible_pair_sids": len(pair_sids),
        "eligible_singleton_sids": len(singleton_sids),
    }


def validate_base(base: list[dict], expected_rows: int) -> tuple[int, int]:
    if len(base) != expected_rows:
        raise ValueError(f"expected {expected_rows} base rows, got {len(base)}")
    merge_count = 0
    insert_count = 0
    for index, row in enumerate(base):
        for key in SOURCE_KEYS:
            if not isinstance(row.get(key), str):
                raise ValueError(f"invalid base {key} at source_index={index}")
        if not row["query"].strip() or not row["trajectory"].strip():
            raise ValueError(f"empty base query/trajectory at source_index={index}")
        if not row["evo_exp"].strip():
            raise ValueError(f"empty base evo_exp at source_index={index}")
        if row["experience"].strip():
            merge_count += 1
        else:
            insert_count += 1
    return merge_count, insert_count


def validate_teacher_rows(
    rows: list[dict],
    base: list[dict],
    expected_rows: int,
    expected_missing: set[int],
    expected_source_mismatches: dict[int, set[str]],
) -> tuple[dict[int, dict], dict]:
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} teacher rows, got {len(rows)}")
    by_index: dict[int, dict] = {}
    mismatch_fields: Counter[str] = Counter()
    mismatch_fields_by_index: dict[int, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows):
        source_index = row.get("source_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise ValueError(f"invalid source_index at teacher row {row_number}")
        if not 0 <= source_index < len(base):
            raise ValueError(f"source_index out of range: {source_index}")
        if source_index in by_index:
            raise ValueError(f"duplicate source_index: {source_index}")
        if row.get("task_type") != "memevolver":
            raise ValueError(f"invalid task_type at source_index={source_index}")
        thinking = row.get("thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            raise ValueError(f"empty thinking at source_index={source_index}")
        for key in SOURCE_KEYS:
            if row.get(key) != base[source_index][key]:
                mismatch_fields[key] += 1
                mismatch_fields_by_index[source_index].add(key)
        by_index[source_index] = row

    missing = set(range(len(base))) - set(by_index)
    if missing != expected_missing:
        raise ValueError(
            "missing source_index mismatch: "
            f"expected={sorted(expected_missing)} actual={sorted(missing)}"
        )
    actual_mismatch_shape = {
        index: set(fields) for index, fields in mismatch_fields_by_index.items()
    }
    if actual_mismatch_shape != expected_source_mismatches:
        raise ValueError(
            "source mismatch shape mismatch: "
            f"expected={sorted((i, sorted(v)) for i, v in expected_source_mismatches.items())} "
            f"actual={sorted((i, sorted(v)) for i, v in actual_mismatch_shape.items())}"
        )
    return by_index, {
        "missing_source_indices": sorted(missing),
        "repaired_source_mismatch_indices": sorted(actual_mismatch_shape),
        "repaired_source_mismatches_by_field": dict(sorted(mismatch_fields.items())),
        "repaired_source_mismatches_by_index": {
            str(index): sorted(fields)
            for index, fields in sorted(actual_mismatch_shape.items())
        },
        "unique_thinking": len({row["thinking"].strip() for row in rows}),
    }


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
    teacher_by_index: dict[int, dict], base: list[dict], replayed: list[dict]
) -> tuple[list[dict], dict]:
    if len(replayed) != len(base):
        raise ValueError("replayed/base row count mismatch")
    lookup: list[dict] = []
    exact_matches = 0
    for source_index, (base_row, replayed_row) in enumerate(zip(base, replayed)):
        actual = {key: replayed_row[key] for key in SOURCE_KEYS}
        expected = {key: base_row[key] for key in SOURCE_KEYS}
        if actual != expected:
            raise ValueError(f"deterministic MemE replay drift at source_index={source_index}")
        exact_matches += 1
        lookup.append(
            {
                "source_index": source_index,
                "sid": replayed_row["sid"],
                "dataset": replayed_row["dataset"],
                "trajectory_source_id": replayed_row["trajectory_source_id"],
                "existing_memory_source_id": replayed_row["existing_memory_source_id"],
                "merged_from_count": replayed_row["merged_from_count"],
                "update_type": replayed_row["update_type"],
                "input_sha256": source_hash(base_row),
            }
        )

    records: list[dict] = []
    escaped_count = 0
    for source_index in sorted(teacher_by_index):
        teacher = teacher_by_index[source_index]
        source = base[source_index]
        meta = replayed[source_index]
        sid = meta["sid"]
        if not is_valid_sid(sid):
            raise ValueError(f"invalid SID at source_index={source_index}")
        query, count = escape_structural_tags(source["query"])
        escaped_count += count
        trajectory, count = escape_structural_tags(source["trajectory"])
        escaped_count += count
        thinking, count = escape_structural_tags(teacher["thinking"].strip())
        escaped_count += count
        dataset, count = escape_structural_tags(meta["dataset"])
        escaped_count += count
        experience, count = escape_structural_tags(source["experience"])
        escaped_count += count
        evo_exp, count = escape_structural_tags(source["evo_exp"])
        escaped_count += count

        tool_call = json.dumps(
            {"name": "memory_lookup", "arguments": {"sid_list": [sid]}},
            ensure_ascii=False,
        )
        if source["experience"].strip():
            observation = (
                f"[SID: {sid}] Dataset: {dataset} | Experience: {experience}"
            )
        else:
            observation = f"[Memory lookup for {sid}]: No matching experience found."
        records.append(
            {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {
                        "from": "human",
                        "value": (
                            f"<query> {query} </query>\n"
                            f"<trajectory>\n{trajectory}\n</trajectory>"
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": (
                            f"<think>\n{thinking}\n</think>\n"
                            f"<tool_call>\n{tool_call}\n</tool_call>"
                        ),
                    },
                    {"from": "observation", "value": observation},
                    {"from": "gpt", "value": f"<answer>\n{evo_exp}\n</answer>"},
                ],
                "tools": TOOL_SCHEMA_STR,
                "source_index": source_index,
                "sid": sid,
                "dataset": meta["dataset"],
                "trajectory_source_id": meta["trajectory_source_id"],
                "existing_memory_source_id": meta["existing_memory_source_id"],
                "merged_from_count": meta["merged_from_count"],
                "update_type": meta["update_type"],
            }
        )
    return records, {
        "base_replay_exact_matches": exact_matches,
        "lookup_rows": len(lookup),
        "output_records": len(records),
        "unique_output_sids": len({row["sid"] for row in records}),
        "empty_initial_experience_rows": sum(
            not base[index]["experience"].strip() for index in teacher_by_index
        ),
        "nonempty_initial_experience_rows": sum(
            bool(base[index]["experience"].strip()) for index in teacher_by_index
        ),
        "escaped_structural_tags": escaped_count,
        "lookup": lookup,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-input", required=True, type=Path)
    parser.add_argument("--base-input", required=True, type=Path)
    parser.add_argument("--raw-bank", required=True, type=Path)
    parser.add_argument("--merged-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lookup-output", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--expected-base-rows", type=int, default=5_000)
    parser.add_argument("--expected-teacher-rows", type=int, default=4_999)
    parser.add_argument("--expected-missing-indices", default="1607")
    parser.add_argument("--expected-source-mismatches", default="10:query")
    parser.add_argument("--base-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_records(args.base_input)
    teacher_rows = load_records(args.teacher_input)
    merge_count, insert_count = validate_base(base, args.expected_base_rows)
    expected_missing = parse_indices(args.expected_missing_indices)
    expected_mismatches = parse_source_mismatches(args.expected_source_mismatches)
    teacher_by_index, teacher_stats = validate_teacher_rows(
        teacher_rows,
        base,
        args.expected_teacher_rows,
        expected_missing,
        expected_mismatches,
    )
    replayed, replay_stats = replay_meme_records(
        args.raw_bank,
        args.merged_bank,
        merge_count,
        insert_count,
        args.base_seed,
    )
    records, output_stats = build_sft_records(teacher_by_index, base, replayed)
    lookup = output_stats.pop("lookup")
    report = {
        "status": "ok",
        "base_rows": len(base),
        "teacher_rows": len(teacher_rows),
        "base_merge_rows": merge_count,
        "base_insert_rows": insert_count,
        **teacher_stats,
        **replay_stats,
        **output_stats,
    }
    write_json_atomic(lookup, args.lookup_output)
    write_json_atomic(records, args.output)
    write_json_atomic(report, args.validation_report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
