#!/usr/bin/env python3
"""Retarget existing MemR COT SFT rows to the deployed RL prompt contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path


RL_SYSTEM_PROMPT = (
    "You are MemRetriever, a memory retrieval agent. For every query, choose exactly "
    "one four-layer Semantic ID (SID) in the exact form "
    "<SID_L1_X><SID_L2_Y><SID_L3_Z><SID_L4_W>, then make exactly one memory_lookup "
    "call whose sid_list contains only that SID. You may reason briefly in "
    "<think>...</think>. After the tool response, output exactly one non-empty "
    "<answer>...</answer> containing a concise, actionable summary of the retrieved "
    "experience. Do not call the tool again, output multiple SIDs, or copy the entire "
    "memory. If no relevant memory is returned, output "
    "<answer>No relevant experience found.</answer>."
)

RL_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "memory_lookup",
            "description": (
                "Look up reusable experience memories using one four-layer "
                "Semantic ID (SID)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sid_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 1,
                        "description": (
                            "Exactly one SID such as "
                            "<SID_L1_5><SID_L2_3><SID_L3_2><SID_L4_1>. "
                            "Natural-language descriptions are invalid."
                        ),
                    }
                },
                "required": ["sid_list"],
            },
        },
    }
]

SID_RE = re.compile(
    r"^<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>$"
)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
SID_LIMITS = (48, 16, 8, 8)


def _valid_sid(value: str) -> bool:
    match = SID_RE.fullmatch(value)
    return bool(match) and all(
        0 <= int(code) < limit for code, limit in zip(match.groups(), SID_LIMITS)
    )


def _validate_record(record: dict, index: int) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 5:
        raise ValueError(f"record {index}: invalid conversations")
    if conversations[0].get("from") != "system":
        raise ValueError(f"record {index}: first turn is not system")

    assistant_text = "\n".join(
        str(turn.get("value") or "")
        for turn in conversations
        if turn.get("from") == "gpt"
    )
    calls = TOOL_CALL_RE.findall(assistant_text)
    if len(calls) != 1:
        raise ValueError(
            f"record {index}: expected exactly one memory_lookup tool call, got {len(calls)}"
        )
    try:
        call = json.loads(calls[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"record {index}: invalid tool-call JSON") from exc
    sid_list = call.get("arguments", {}).get("sid_list")
    if call.get("name") != "memory_lookup" or not isinstance(sid_list, list) or len(sid_list) != 1:
        raise ValueError(f"record {index}: tool call must contain exactly one SID")
    sid = sid_list[0]
    if not isinstance(sid, str) or not _valid_sid(sid):
        raise ValueError(f"record {index}: invalid four-layer SID")
    if record.get("sid") != sid:
        raise ValueError(f"record {index}: metadata SID differs from tool-call SID")

    answers = ANSWER_RE.findall(assistant_text)
    if len(answers) != 1 or not answers[0].strip():
        raise ValueError(f"record {index}: expected exactly one non-empty answer")
    observations = [
        str(turn.get("value") or "")
        for turn in conversations
        if turn.get("from") == "observation"
    ]
    if len(observations) != 1 or sid not in observations[0]:
        raise ValueError(f"record {index}: observation does not match SID")
    return sid


def retarget(records: list[dict]) -> list[dict]:
    transformed = []
    tool_schema = json.dumps(RL_TOOL_SCHEMA, ensure_ascii=False)
    for index, record in enumerate(records):
        _validate_record(record, index)
        updated = copy.deepcopy(record)
        updated["conversations"][0]["value"] = RL_SYSTEM_PROMPT
        updated["tools"] = tool_schema
        transformed.append(updated)
    return transformed


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != args.expected_records:
        actual = len(records) if isinstance(records, list) else "not-a-list"
        raise ValueError(f"expected {args.expected_records} records, got {actual}")
    transformed = retarget(records)
    write_json_atomic(args.output, transformed)
    print(
        json.dumps(
            {
                "records": len(transformed),
                "system_prompt": RL_SYSTEM_PROMPT,
                "tool_schema": RL_TOOL_SCHEMA,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
