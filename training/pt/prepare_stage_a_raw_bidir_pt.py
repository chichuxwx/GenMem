#!/usr/bin/env python3
"""Build Stage-A four-level SID bidirectional PT data plus general PT text.

The three content corpora produced for the earlier 48x24x24 run are reused as
the canonical, already-truncated text source.  Their obsolete three-level SID
targets are replaced by the new mapping via sample_id.  For every source row we
emit both content -> SID and SID -> content, without deduplication.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_SPECS = (
    (
        "raw_experience",
        "sid_raw_experience_to_sid.jsonl",
        "Raw experience",
        "Generate one representative raw experience associated with this four-level semantic ID (SID).",
    ),
    (
        "query",
        "sid_query_to_sid.jsonl",
        "User query",
        "Generate one representative user query associated with this four-level semantic ID (SID).",
    ),
    (
        "full_trajectory",
        "sid_full_trajectory_to_sid.jsonl",
        "Complete agent trajectory",
        "Generate one representative complete agent trajectory associated with this four-level semantic ID (SID).",
    ),
)

SID_RE = re.compile(r"(?:<SID_L\d+_\d+>)+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--nemotron-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-levels", type=int, default=4)
    return parser.parse_args()


def valid_sid(sid: str, num_levels: int) -> bool:
    tokens = re.findall(r"<SID_L(\d+)_(\d+)>", sid)
    return len(tokens) == num_levels and [int(level) for level, _ in tokens] == list(
        range(1, num_levels + 1)
    )


def load_mapping(path: Path, num_levels: int) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    mapping: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        sid = str(row.get("sid_tokens", ""))
        if not sample_id or not valid_sid(sid, num_levels):
            raise ValueError(f"Invalid mapping row: {row}")
        previous = mapping.get(sample_id)
        if previous is not None and previous != sid:
            raise ValueError(f"Conflicting SID for sample_id={sample_id}")
        mapping[sample_id] = sid
    return mapping


def extract_content(text: str, label: str) -> str:
    target_marker = "\n\nCorresponding SID:"
    if target_marker not in text:
        raise ValueError("Missing Corresponding SID marker")
    prompt = text.rsplit(target_marker, 1)[0]
    content_marker = f"\n\n{label}:\n"
    if content_marker not in prompt:
        raise ValueError(f"Missing content marker {content_marker!r}")
    return prompt.split(content_marker, 1)[1].strip()


def forward_text(kind: str, label: str, content: str, sid: str) -> str:
    names = {
        "raw_experience": "raw experience",
        "query": "user query",
        "full_trajectory": "complete agent trajectory",
    }
    return (
        f"Predict the four-level semantic ID (SID) for this {names[kind]}.\n\n"
        f"{label}:\n{content}\n\nCorresponding SID: {sid}"
    )


def reverse_text(instruction: str, label: str, content: str, sid: str) -> str:
    return f"{instruction}\n\nSID: {sid}\n\n{label}:\n{content}"


def general_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    query = str(row.get("input", "")).strip()
    response = str(row.get("output", "")).strip()
    parts = [f"Instruction:\n{instruction}"]
    if query:
        parts.append(f"Input:\n{query}")
    parts.append(f"Response:\n{response}")
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(Path(args.sid_mapping), args.num_levels)

    sid_output = output_dir / "sid_stage_a_raw_bidir_pt_48_16_8_8_w0p5.jsonl"
    general_output = output_dir / "nemotron_general_pt_48_16_8_8_w0p5.jsonl"
    stats: Counter[str] = Counter()
    missing: Counter[str] = Counter()

    with sid_output.open("w", encoding="utf-8") as out:
        for kind, filename, label, reverse_instruction in SOURCE_SPECS:
            source_path = Path(args.source_dir) / filename
            with source_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id", ""))
                    sid = mapping.get(sample_id)
                    if sid is None:
                        missing[kind] += 1
                        continue
                    content = extract_content(str(row.get("text", "")), label)
                    common = {
                        "sample_id": sample_id,
                        "dataset": row.get("dataset", "unknown"),
                        "source_type": kind,
                    }
                    out.write(
                        json.dumps(
                            {
                                "text": forward_text(kind, label, content, sid),
                                "task_type": f"{kind}_to_sid",
                                **common,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    out.write(
                        json.dumps(
                            {
                                "text": reverse_text(reverse_instruction, label, content, sid),
                                "task_type": f"sid_to_{kind}",
                                **common,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats[f"{kind}_to_sid"] += 1
                    stats[f"sid_to_{kind}"] += 1

    with Path(args.nemotron_json).open(encoding="utf-8") as handle:
        general_rows = json.load(handle)
    with general_output.open("w", encoding="utf-8") as out:
        for row in general_rows:
            text = general_text(row)
            if not text.strip():
                continue
            out.write(
                json.dumps(
                    {"text": text, "task_type": "nemotron_general_pt"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["nemotron_general_pt"] += 1

    dataset_info = {
        "sid_stage_a_raw_bidir_pt_48_16_8_8_w0p5": {
            "file_name": sid_output.name,
            "columns": {"prompt": "text"},
        },
        "nemotron_general_pt_48_16_8_8_w0p5": {
            "file_name": general_output.name,
            "columns": {"prompt": "text"},
        },
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)
    report = {
        "sid_mapping_rows": len(mapping),
        "counts": dict(stats),
        "missing_mapping": dict(missing),
        "specialized_total": sum(v for k, v in stats.items() if k != "nemotron_general_pt"),
        "outputs": {
            "sid": str(sid_output),
            "general": str(general_output),
        },
    }
    with (output_dir / "generation_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
