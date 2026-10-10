#!/usr/bin/env python3
"""Generate indexed CoT fields through an OpenAI-compatible teacher endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from retarget_memr_rl_prompt import write_json_atomic


def parse_field(text: str, field: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    obj = json.loads(text)
    value = obj.get(field) if isinstance(obj, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing teacher field: {field}")
    if any(tag in value for tag in ("<think>", "</think>", "<answer>", "</answer>", "<tool_call>")):
        raise ValueError("Teacher fields must be plain text")
    return value.strip()


def request_field(args, instruction: str, payload: dict, field: str) -> str:
    import requests

    key = os.environ.get(args.api_key_env, "EMPTY")
    for attempt in range(args.retries):
        try:
            response = requests.post(
                args.api_base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Teacher response did not finish normally")
            return parse_field(choice["message"].get("content") or "", field)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            if attempt + 1 == args.retries:
                raise RuntimeError(f"Teacher {field} failed after {args.retries} attempts") from None
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("retries must be positive")


def generate_record(index, row, args, prompts):
    output = dict(row, source_index=index, task_type=args.agent)
    if args.agent == "memretriever":
        output["thinking"] = request_field(
            args, prompts["memr_thinking"], {"query": row["query"]}, "thinking"
        )
        output["rewritten_exp"] = request_field(
            args, prompts["memr_rewrite"],
            {"query": row["query"], "experience": row["experience"], "thinking": output["thinking"]},
            "rewritten_exp",
        )
    else:
        output["thinking"] = request_field(
            args, prompts["meme_thinking"],
            {"query": row["query"], "trajectory": row["trajectory"]}, "thinking",
        )
    return output


def load_completed(path, rows, agent):
    completed = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        index = record.get("source_index")
        if type(index) is not int or not 0 <= index < len(rows) or index in completed:
            raise ValueError(f"Invalid or duplicate source_index at line {line_number}")
        if record.get("task_type") != agent or any(record.get(k) != v for k, v in rows[index].items()):
            raise ValueError(f"Source fields changed at line {line_number}")
        for field in ("thinking", "rewritten_exp") if agent == "memretriever" else ("thinking",):
            parse_field(json.dumps({field: record.get(field)}), field)
        completed[index] = record
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=("memretriever", "memevolver"), required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("teacher_prompts.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if min(args.workers, args.retries, args.max_tokens, args.timeout) <= 0:
        parser.error("workers, retries, max-tokens and timeout must be positive")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    fields = {"query", "experience", "trajectory"}
    if args.agent == "memevolver":
        fields.add("evo_exp")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty teacher-input array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != fields or any(not isinstance(row[k], str) for k in fields):
            raise ValueError(f"Invalid input schema at source_index={index}")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    metadata = args.output.with_suffix(args.output.suffix + ".meta.json")
    signature = hashlib.sha256(json.dumps({
        "rows": rows, "agent": args.agent, "model": args.model,
        "prompts": prompts, "max_tokens": args.max_tokens, "temperature": args.temperature,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if args.input.resolve() in (args.output.resolve(), metadata.resolve()):
        raise ValueError("Input and output paths must differ")
    if metadata.exists():
        if json.loads(metadata.read_text())["signature"] != signature:
            raise ValueError("Input/settings changed; choose a new output path")
    elif args.output.exists():
        raise ValueError("Existing output lacks generation metadata; choose a new output path")
    else:
        write_json_atomic(metadata, {"signature": signature, "model": args.model, "agent": args.agent})
    completed = load_completed(args.output, rows, args.agent)
    failed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a", encoding="utf-8") as stream:
        jobs = {
            pool.submit(generate_record, i, row, args, prompts): i
            for i, row in enumerate(rows) if i not in completed
        }
        for future in as_completed(jobs):
            index = jobs[future]
            try:
                record = future.result()
            except Exception as exc:
                failed += 1
                print(f"source_index={index}: failed ({type(exc).__name__}); rerun to retry", flush=True)
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed[index] = record
            print(f"completed={len(completed)}/{len(rows)}", flush=True)
    if failed:
        raise SystemExit(f"{failed} records failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
