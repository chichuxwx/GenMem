#!/usr/bin/env python3
"""Build flat MemR and MemE inputs for external CoT generation.

MemR records contain exactly ``query``, ``experience``, and ``trajectory``.
MemE records additionally contain ``evo_exp`` as the evolution target.  MemR
uses a merged memory as the relevant experience.  MemE uses only exact
two-source clusters for merge examples and singleton clusters for empty-memory
insert examples.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from build_meme_react_sft_data import (
    entry_experience,
    entry_sid,
    is_valid_sid,
    iter_json_array,
    trajectory_text,
    write_json_atomic,
)


SOURCE_FIELDS = (
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


def load_merged_bank(path: Path) -> tuple[dict[str, dict], int]:
    merged: dict[str, dict] = {}
    total_entries = 0
    for entry in iter_json_array(path):
        total_entries += 1
        sid = entry_sid(entry)
        summary = entry_experience(entry)
        source_count = entry.get("merged_from_count")
        if (
            not is_valid_sid(sid)
            or not summary
            or not isinstance(source_count, int)
            or source_count < 1
        ):
            continue
        if sid in merged:
            raise ValueError(f"duplicate merged SID: {sid}")
        merged[sid] = {
            "summary": summary,
            "merged_from_count": source_count,
        }
    return merged, total_entries


def compact_source(entry: dict) -> dict:
    compact = {key: entry.get(key) for key in SOURCE_FIELDS}
    compact["sid"] = entry.get("sid")
    return compact


def has_real_execution(entry: dict) -> bool:
    if str(entry.get("steps_summary") or "").strip():
        return True
    trajectory_summary = entry.get("trajectory_summary")
    if isinstance(trajectory_summary, dict) and str(
        trajectory_summary.get("trajectory_summary") or ""
    ).strip():
        return True
    return bool(str(entry.get("reasoning") or "").strip())


def collect_sources(
    raw_bank: Path, merged: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, int], int]:
    """Collect one MemR representative and grounded MemE source pairs."""
    representatives: dict[str, dict] = {}
    meme_sources: dict[str, list[dict]] = defaultdict(list)
    raw_counts: dict[str, int] = defaultdict(int)
    total_entries = 0

    for entry in iter_json_array(raw_bank):
        total_entries += 1
        sid = entry_sid(entry)
        target = merged.get(sid)
        if target is None:
            continue
        raw_counts[sid] += 1
        source = compact_source(entry)
        try:
            trajectory_text(source)
        except ValueError:
            continue

        representatives.setdefault(sid, source)
        retained = 2 if target["merged_from_count"] == 2 else 1
        if (
            target["merged_from_count"] in (1, 2)
            and has_real_execution(source)
            and len(meme_sources[sid]) < retained
        ):
            if target["merged_from_count"] == 1 or entry_experience(source):
                meme_sources[sid].append(source)

    return representatives, dict(meme_sources), dict(raw_counts), total_entries


def flat_record(query: str, experience: str, trajectory: str) -> dict[str, str]:
    return {
        "query": query,
        "experience": experience,
        "trajectory": trajectory,
    }


def meme_record(
    query: str, experience: str, trajectory: str, evo_exp: str
) -> dict[str, str]:
    return {
        "query": query,
        "experience": experience,
        "trajectory": trajectory,
        "evo_exp": evo_exp,
    }


def record_key(record: dict[str, str]) -> tuple[str, str, str]:
    return record["query"], record["experience"], record["trajectory"]


def normalise_text(text: str) -> str:
    return " ".join(text.split())


def is_clean_meme_record(record: dict[str, str]) -> bool:
    experience = normalise_text(record["experience"])
    trajectory = normalise_text(record["trajectory"])
    evo_exp = normalise_text(record["evo_exp"])
    if not trajectory or not evo_exp:
        return False
    if trajectory == evo_exp or (experience and trajectory == experience):
        return False
    if experience and experience == evo_exp:
        return False
    if evo_exp in trajectory or (experience and experience in trajectory):
        return False
    return True


def build_memr_records(
    representatives: dict[str, dict],
    merged: dict[str, dict],
    sample_count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if sample_count <= 0:
        raise ValueError("memr_samples must be positive")
    eligible_sids = sorted(set(representatives) & set(merged))
    candidates: list[dict[str, str]] = []
    seen_records: set[tuple[str, str, str]] = set()
    for sid in eligible_sids:
        source = representatives[sid]
        record = flat_record(
            str(source.get("question") or "").strip(),
            merged[sid]["summary"],
            trajectory_text(source),
        )
        key = record_key(record)
        if key not in seen_records:
            candidates.append(record)
            seen_records.add(key)

    if len(candidates) < sample_count:
        raise ValueError(
            f"requested {sample_count} MemR samples but found only "
            f"{len(candidates)} unique eligible records"
        )
    records = rng.sample(candidates, sample_count)
    rng.shuffle(records)
    return records


def pair_record(
    sources: list[dict], trajectory_index: int, evo_exp: str
) -> dict[str, str]:
    trajectory_source = sources[trajectory_index]
    existing_source = sources[1 - trajectory_index]
    return meme_record(
        str(trajectory_source.get("question") or "").strip(),
        entry_experience(existing_source),
        trajectory_text(trajectory_source),
        evo_exp,
    )


def build_meme_records(
    sources: dict[str, list[dict]],
    raw_counts: dict[str, int],
    merged: dict[str, dict],
    merge_count: int,
    insert_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], int, int]:
    if merge_count < 0 or insert_count < 0 or merge_count + insert_count <= 0:
        raise ValueError("MemE sample counts must be non-negative with a positive total")

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
    primary_pair_records: list[dict[str, str]] = []
    secondary_pair_records: list[dict[str, str]] = []
    pending_secondary: list[dict[str, str]] = []
    seen_pair_records: set[tuple[str, str, str]] = set()
    for sid in pair_sids:
        orientation = rng.randrange(2)
        evo_exp = merged[sid]["summary"]
        candidates = [
            pair_record(sources[sid], orientation, evo_exp),
            pair_record(sources[sid], 1 - orientation, evo_exp),
        ]
        candidates = [record for record in candidates if is_clean_meme_record(record)]
        chosen_index = None
        for index, record in enumerate(candidates):
            key = record_key(record)
            if key not in seen_pair_records:
                primary_pair_records.append(record)
                seen_pair_records.add(key)
                chosen_index = index
                break
        if chosen_index is not None:
            pending_secondary.extend(
                record
                for index, record in enumerate(candidates)
                if index != chosen_index
            )

    for record in pending_secondary:
        key = record_key(record)
        if key not in seen_pair_records:
            secondary_pair_records.append(record)
            seen_pair_records.add(key)

    unique_pair_orientations = primary_pair_records + secondary_pair_records
    if merge_count > len(unique_pair_orientations):
        raise ValueError(
            f"requested {merge_count} MemE merge samples but only "
            f"{len(unique_pair_orientations)} unique grounded pair orientations "
            "are available"
        )
    primary_count = min(merge_count, len(primary_pair_records))
    merge_records = primary_pair_records[:primary_count]
    merge_records.extend(secondary_pair_records[: merge_count - primary_count])

    rng.shuffle(singleton_sids)
    insert_candidates: list[dict[str, str]] = []
    seen_insert_records: set[tuple[str, str, str]] = set()
    for sid in singleton_sids:
        record = meme_record(
            str(sources[sid][0].get("question") or "").strip(),
            "",
            trajectory_text(sources[sid][0]),
            merged[sid]["summary"],
        )
        if not is_clean_meme_record(record):
            continue
        key = record_key(record)
        if key not in seen_insert_records:
            insert_candidates.append(record)
            seen_insert_records.add(key)
    if insert_count > len(insert_candidates):
        raise ValueError(
            f"requested {insert_count} MemE insert samples but only "
            f"{len(insert_candidates)} unique grounded singleton records are available"
        )
    insert_records = insert_candidates[:insert_count]
    records = merge_records + insert_records
    rng.shuffle(records)
    return records, len(pair_sids), len(singleton_sids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-bank", required=True, type=Path)
    parser.add_argument("--merged-bank", required=True, type=Path)
    parser.add_argument("--memr-output", required=True, type=Path)
    parser.add_argument("--meme-output", required=True, type=Path)
    parser.add_argument("--memr-samples", type=int, default=5_000)
    parser.add_argument("--meme-merge-samples", type=int, default=4_000)
    parser.add_argument("--meme-insert-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    merged, merged_entries = load_merged_bank(args.merged_bank)
    representatives, sources, raw_counts, raw_entries = collect_sources(
        args.raw_bank, merged
    )

    memr_rng = random.Random(args.seed)
    meme_rng = random.Random(args.seed + 1)
    memr_records = build_memr_records(
        representatives, merged, args.memr_samples, memr_rng
    )
    meme_records, eligible_pairs, eligible_singletons = build_meme_records(
        sources,
        raw_counts,
        merged,
        args.meme_merge_samples,
        args.meme_insert_samples,
        meme_rng,
    )
    write_json_atomic(memr_records, args.memr_output)
    write_json_atomic(meme_records, args.meme_output)
    print(
        f"raw_entries={raw_entries} merged_entries={merged_entries} "
        f"valid_merged_sids={len(merged)} memr_records={len(memr_records)} "
        f"eligible_pair_sids={eligible_pairs} "
        f"eligible_singleton_sids={eligible_singletons} "
        f"meme_merge_records={args.meme_merge_samples} "
        f"meme_insert_records={args.meme_insert_samples} "
        f"memr_output={args.memr_output} meme_output={args.meme_output}"
    )


if __name__ == "__main__":
    main()
