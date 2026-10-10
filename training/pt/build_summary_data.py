#!/usr/bin/env python3
"""Build four-level merged-summary <-> SID SFT data for Stage C."""

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


SID_RE = re.compile(
    r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
)
SID_LIMITS = (48, 16, 8, 8)

SUMMARY_TO_SID = (
    "Given the following merged memory-bank experience, identify its "
    "four-level semantic ID. Output exactly four SID tokens in the format "
    "<SID_L1_X><SID_L2_Y><SID_L3_Z><SID_L4_W>, with no other text."
)
SID_TO_SUMMARY = (
    "Generate the representative merged memory-bank experience associated "
    "with the following four-level semantic ID."
)


@dataclass(frozen=True)
class ConversionStats:
    input_leaves: int
    valid_leaves: int
    excluded_leaves: int
    skipped_leaves: int
    output_records: int


def is_valid_sid(sid: str) -> bool:
    match = SID_RE.fullmatch(sid)
    if match is None:
        return False
    return all(int(value) < limit for value, limit in zip(match.groups(), SID_LIMITS))


def normalize_entries(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [
            {"sid": {"tokens": sid}, "experience_summary": summary}
            for sid, summary in payload.items()
        ]
    raise TypeError("merged memory bank must be a JSON list or object")


def entry_sid(entry: dict) -> str:
    return str(entry.get("sid", {}).get("tokens", "")).strip()


def convert_entries(
    entries: list[dict], exclude_sids: set[str] | None = None
) -> tuple[list[dict], ConversionStats]:
    records = []
    seen = set()
    excluded = 0
    skipped = 0
    exclude_sids = exclude_sids or set()

    for entry in entries:
        sid = entry_sid(entry)
        summary = str(
            entry.get("experience_summary") or entry.get("experience_text") or ""
        ).strip()
        if not is_valid_sid(sid) or not summary:
            skipped += 1
            continue
        if sid in exclude_sids:
            excluded += 1
            continue
        if sid in seen:
            raise ValueError(f"duplicate SID: {sid}")
        seen.add(sid)
        records.append(
            {
                "instruction": SUMMARY_TO_SID,
                "input": summary,
                "output": sid,
                "task_type": "summary_to_sid",
            }
        )
        records.append(
            {
                "instruction": SID_TO_SUMMARY,
                "input": sid,
                "output": summary,
                "task_type": "sid_to_summary",
            }
        )

    return records, ConversionStats(
        input_leaves=len(entries),
        valid_leaves=len(seen),
        excluded_leaves=excluded,
        skipped_leaves=skipped,
        output_records=len(records),
    )


def write_jsonl_atomic(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-leaves", type=int)
    parser.add_argument("--exclude-sids-from", action="append", type=Path, default=[])
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        entries = normalize_entries(json.load(handle))

    exclude_sids = set()
    for exclude_path in args.exclude_sids_from:
        with exclude_path.open(encoding="utf-8") as handle:
            exclude_entries = normalize_entries(json.load(handle))
        exclude_sids.update(entry_sid(entry) for entry in exclude_entries)

    records, stats = convert_entries(entries, exclude_sids=exclude_sids)
    if args.expected_leaves is not None and stats.input_leaves != args.expected_leaves:
        raise ValueError(
            f"expected {args.expected_leaves} merged leaves, got {stats.input_leaves}"
        )
    expected_valid = stats.input_leaves - stats.excluded_leaves
    if stats.valid_leaves != expected_valid or stats.skipped_leaves:
        raise ValueError(
            f"valid={stats.valid_leaves} excluded={stats.excluded_leaves} "
            f"skipped={stats.skipped_leaves}; refusing partial data"
        )

    write_jsonl_atomic(records, args.output)
    print(
        f"input_leaves={stats.input_leaves} valid_leaves={stats.valid_leaves} "
        f"excluded_leaves={stats.excluded_leaves} "
        f"output_records={stats.output_records} output={args.output}"
    )


if __name__ == "__main__":
    main()
