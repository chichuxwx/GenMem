"""Build task-balanced four-layer MemRetriever RL parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
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
_SID_RE = re.compile(
    r"^<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>$"
)
_BOUNDS = (48, 16, 8, 8)
_UNSUPPORTED_OUTCOME = {"healthbench", "skills_crawl_clustered"}


def _clean_question(dataset: str, question: str) -> str:
    value = str(question or "").strip()
    if dataset.lower() == "alfworld":
        value = value.split("\n\nEnvironment/task excerpt:", 1)[0].strip()
    return value


def _valid_sid(record: dict) -> tuple[list[int], str]:
    sid = record.get("sid") if isinstance(record.get("sid"), dict) else {}
    codes = sid.get("codes")
    tokens = str(sid.get("tokens") or "")
    match = _SID_RE.fullmatch(tokens)
    if (
        not isinstance(codes, list)
        or len(codes) != 4
        or not match
        or [int(match.group(i)) for i in range(1, 5)] != [int(x) for x in codes]
        or any(int(code) < 0 or int(code) >= bound for code, bound in zip(codes, _BOUNDS))
    ):
        raise ValueError(f"record {record.get('sample_id')!r} has no valid four-layer SID")
    return [int(x) for x in codes], tokens


def _has_outcome(record: dict) -> bool:
    dataset = str(record.get("dataset") or "").lower()
    if dataset in _UNSUPPORTED_OUTCOME:
        return False
    if dataset == "alfworld":
        return bool(_clean_question(dataset, record.get("question", "")))
    return bool(str(record.get("ground_truth") or "").strip())


def _representative_rank(record: dict) -> tuple[int, int, float, str]:
    evaluation = str(record.get("evaluation") or "").lower()
    return (
        int(record.get("merged_from_count") or 0),
        int(bool(record.get("success")) or evaluation == "correct"),
        float(record.get("score") or 0.0),
        str(record.get("sample_id") or ""),
    )


def select_representatives(records: list[dict], max_per_dataset: int = 1600):
    """Keep one stable representative per dataset/question, then cap large tasks."""
    stats = Counter()
    by_question: dict[tuple[str, str], dict] = {}
    for record in records:
        if not _has_outcome(record):
            stats["dropped_unsupported_outcome"] += 1
            continue
        dataset = str(record.get("dataset") or "").lower()
        question = _clean_question(dataset, record.get("question", ""))
        key = (dataset, question)
        current = by_question.get(key)
        if current is None:
            by_question[key] = record
        else:
            stats["dropped_duplicate_question"] += 1
            if _representative_rank(record) > _representative_rank(current):
                by_question[key] = record

    by_dataset: dict[str, list[dict]] = {}
    for record in by_question.values():
        by_dataset.setdefault(str(record["dataset"]).lower(), []).append(record)

    selected: list[dict] = []
    for dataset in sorted(by_dataset):
        items = sorted(by_dataset[dataset], key=lambda item: str(item.get("sample_id") or ""))
        if len(items) > max_per_dataset:
            rng = random.Random(f"v33:{dataset}:42")
            rng.shuffle(items)
            stats["dropped_dataset_cap"] += len(items) - max_per_dataset
            items = sorted(items[:max_per_dataset], key=lambda item: str(item.get("sample_id") or ""))
        selected.extend(items)
        stats[f"selected_{dataset}"] = len(items)
    return selected, dict(stats)


def _ground_truth(record: dict) -> dict:
    codes, tokens = _valid_sid(record)
    dataset = str(record.get("dataset") or "").lower()
    experience = str(
        record.get("experience_text") or record.get("experience_summary") or ""
    ).strip()
    if not experience:
        raise ValueError(f"record {record.get('sample_id')!r} has empty experience text")
    aliases = record.get("ground_truth_aliases")
    if not isinstance(aliases, list):
        aliases = []
    return {
        "experience_text": f"[SID: {tokens}] Dataset: {dataset} | Experience: {experience}",
        "sample_id": str(record.get("sample_id") or ""),
        "sid_codes": codes,
        "sid_tokens": tokens,
        "target_answer": str(record.get("ground_truth") or ""),
        "target_answer_aliases": [str(item) for item in aliases],
    }


def build_row(record: dict, index: int = 0, split: str = "train") -> dict:
    dataset = str(record.get("dataset") or "").lower()
    question = _clean_question(dataset, record.get("question", ""))
    if not question:
        raise ValueError(f"record {record.get('sample_id')!r} has empty question")
    ground_truth = _ground_truth(record)
    user_content = f"<query> {question} </query>"
    create_kwargs = {
        "data_source": f"mem_retriever_{dataset}",
        "ground_truth": ground_truth,
        "question": question,
    }
    rlsd_action_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_content}\n\n[Training-only SID hint]\n"
                f"Correct SID: {ground_truth['sid_tokens']}"
            ),
        },
    ]
    rlsd_answer_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_content}\n\n[Training-only correct lookup context]\n"
                f"Correct SID: {ground_truth['sid_tokens']}\n"
                "<tool_response>\n"
                f"{ground_truth['experience_text']}\n"
                "</tool_response>"
            ),
        },
    ]
    return {
        "data_source": f"mem_retriever_{dataset}",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "ability": "memory_retrieval",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "answer_source": "environment" if dataset == "alfworld" else "ground_truth",
            "dataset": dataset,
            "index": int(index),
            "need_tools_kwargs": True,
            "privileged_observation_truncated": False,
            # Keep the legacy field as the action context for existing
            # RLSD callers.  The actor uses the explicit contexts below to
            # avoid leaking a memory observation into SID selection.
            "privileged_prompt": rlsd_action_prompt,
            "rlsd_action_prompt": rlsd_action_prompt,
            "rlsd_answer_prompt": rlsd_answer_prompt,
            "privileged_prompt_tokens": 0,
            "question": question,
            "sid_tokens": ground_truth["sid_tokens"],
            "split": split,
            "tools_kwargs": {"memory_lookup": {"create_kwargs": create_kwargs}},
        },
        "metadata": {
            "dataset": dataset,
            "sample_id": ground_truth["sample_id"],
        },
    }


def _is_test_record(record: dict, test_fraction: float) -> bool:
    dataset = str(record.get("dataset") or "").lower()
    question = _clean_question(dataset, record.get("question", ""))
    digest = hashlib.sha256(f"{dataset}\0{question}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < test_fraction


def build_dataset(bank_path: Path, output_dir: Path, max_per_dataset: int, test_fraction: float):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with bank_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("memory bank root must be a JSON array")
    selected, stats = select_representatives(records, max_per_dataset=max_per_dataset)
    train_records = [item for item in selected if not _is_test_record(item, test_fraction)]
    test_records = [item for item in selected if _is_test_record(item, test_fraction)]
    train_rows = [build_row(item, index=index, split="train") for index, item in enumerate(train_records)]
    test_rows = [build_row(item, index=index, split="test") for index, item in enumerate(test_records)]
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train_rows), output_dir / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(test_rows), output_dir / "test.parquet", compression="zstd")
    manifest = {
        "source_memory_bank": str(bank_path.resolve()),
        "source_records": len(records),
        "selected_records": len(selected),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "max_per_dataset": max_per_dataset,
        "test_fraction": test_fraction,
        "selection_stats": stats,
        "train_by_dataset": dict(Counter(row["metadata"]["dataset"] for row in train_rows)),
        "test_by_dataset": dict(Counter(row["metadata"]["dataset"] for row in test_rows)),
        "sid_shape": [48, 16, 8, 8],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-dataset", type=int, default=1600)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    manifest = build_dataset(
        args.bank, args.output_dir, args.max_per_dataset, args.test_fraction
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
