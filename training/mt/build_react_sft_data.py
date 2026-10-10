#!/usr/bin/env python3
"""Build four-level MemRetriever ReAct SFT data from summarized leaves."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path


SID_RE = re.compile(
    r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
)
SID_LIMITS = (48, 16, 8, 8)

SYSTEM_PROMPT = (
    "You are MemRetriever, a memory retrieval agent. For each query, reason in "
    "<think>...</think>, retrieve relevant experience using memory_lookup, then "
    "summarise in <answer>...</answer>."
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

THINK1_TEMPLATES = (
    "This question involves {domain}. I will look up relevant memories using SID tokens.",
    "The query is about {domain}. Let me retrieve related experience from the memory bank.",
    "I need memories related to {domain}. I'll call memory_lookup with the appropriate SID.",
    "This is a {domain} question. I'll search the memory bank for relevant experience.",
)
THINK2_TEMPLATES = (
    "I found relevant experience. Let me organise and summarise it.",
    "The retrieved experience is relevant to the query. I'll summarise the key points.",
    "Good, I have the relevant experience. Let me provide a clean summary.",
    "The memory lookup returned useful information. I'll synthesise it into a clear answer.",
)
DATASET_DOMAIN = {
    "math": "mathematics and arithmetic",
    "popqa": "factual knowledge lookup",
    "triviaqa": "trivia and general knowledge",
    "hotpotqa": "multi-hop reasoning",
    "nq": "general knowledge",
    "musique": "multi-step reasoning",
    "2wiki": "multi-hop knowledge",
    "alfworld": "embodied household planning",
}


def is_valid_sid(sid: str) -> bool:
    match = SID_RE.fullmatch(sid)
    return match is not None and all(
        int(value) < limit for value, limit in zip(match.groups(), SID_LIMITS)
    )


def get_domain(dataset: str) -> str:
    lowered = dataset.lower()
    for key, domain in DATASET_DOMAIN.items():
        if key in lowered:
            return domain
    return "general problem solving"


def load_response_bank(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = (
            (
                str(entry.get("sid", {}).get("tokens", "")).strip(),
                entry.get("experience_summary") or entry.get("experience_text") or "",
            )
            for entry in payload
        )
    else:
        raise TypeError("response bank must be a JSON object or list")

    responses: dict[str, str] = {}
    for sid, text in items:
        sid = str(sid).strip()
        text = str(text).strip()
        if not is_valid_sid(sid) or not text:
            raise ValueError(f"invalid summarized response for SID: {sid!r}")
        if sid in responses:
            raise ValueError(f"duplicate summarized SID: {sid}")
        responses[sid] = text
    return responses


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream objects from a top-level JSON array without loading its embeddings."""
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
                raise TypeError("every raw memory-bank entry must be a JSON object")
            yield value
            buffer = buffer[end:]


def build_sample(entry: dict, summary: str, template_seed: int) -> dict:
    sid = str(entry["sid"]["tokens"]).strip()
    question = str(entry["question"]).strip()
    dataset = str(entry.get("dataset") or "default")
    rng = random.Random(template_seed)
    think1 = rng.choice(THINK1_TEMPLATES).format(domain=get_domain(dataset))
    think2 = rng.choice(THINK2_TEMPLATES)
    tool_call = json.dumps(
        {"name": "memory_lookup", "arguments": {"sid_list": [sid]}},
        ensure_ascii=False,
    )
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"<query> {question} </query>"},
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{think1}\n</think>\n"
                    f"<tool_call>\n{tool_call}\n</tool_call>"
                ),
            },
            {
                "from": "observation",
                "value": f"[SID: {sid}] Dataset: {dataset} | Experience: {summary}",
            },
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{think2}\n</think>\n"
                    f"<answer>\n{summary}\n</answer>"
                ),
            },
        ],
        "tools": TOOL_SCHEMA_STR,
        "sample_id": str(entry.get("sample_id") or ""),
        "dataset": dataset,
        "sid": sid,
    }


def reservoir_samples(
    raw_bank: Path,
    responses: dict[str, str],
    max_samples: int,
    seed: int,
) -> tuple[list[dict], int, int]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = random.Random(seed)
    reservoir: list[tuple[dict, str]] = []
    total_entries = 0
    eligible_entries = 0
    for entry in iter_json_array(raw_bank):
        total_entries += 1
        sid = str(entry.get("sid", {}).get("tokens", "")).strip()
        question = str(entry.get("question") or "").strip()
        summary = responses.get(sid, "")
        if not is_valid_sid(sid) or not question or not summary:
            continue
        eligible_entries += 1
        compact_entry = {
            "sample_id": entry.get("sample_id") or "",
            "dataset": entry.get("dataset") or "default",
            "question": question,
            "sid": {"tokens": sid},
        }
        item = (compact_entry, summary)
        if len(reservoir) < max_samples:
            reservoir.append(item)
        else:
            replacement = rng.randrange(eligible_entries)
            if replacement < max_samples:
                reservoir[replacement] = item

    if len(reservoir) != max_samples:
        raise ValueError(
            f"requested {max_samples} samples but found only {eligible_entries} eligible entries"
        )
    rng.shuffle(reservoir)
    samples = [
        build_sample(entry, summary, template_seed=seed + index)
        for index, (entry, summary) in enumerate(reservoir)
    ]
    return samples, total_entries, eligible_entries


def write_json_atomic(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--response-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-response-sids", type=int)
    args = parser.parse_args()

    responses = load_response_bank(args.response_bank)
    if (
        args.expected_response_sids is not None
        and len(responses) != args.expected_response_sids
    ):
        raise ValueError(
            f"expected {args.expected_response_sids} summarized SIDs, got {len(responses)}"
        )
    records, total_entries, eligible_entries = reservoir_samples(
        args.input, responses, args.max_samples, args.seed
    )
    write_json_atomic(records, args.output)
    print(
        f"raw_entries={total_entries} summarized_sids={len(responses)} "
        f"eligible_entries={eligible_entries} output_records={len(records)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
