#!/usr/bin/env python3
"""Merge experiences by SID with an OpenAI-compatible summarization service."""
import argparse
import json
import hashlib
import tempfile
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


MERGE_PROMPT = """You are an expert at synthesizing problem-solving experiences. Below are {n} experience summaries that all belong to the same semantic cluster (SID: {sid}). They describe similar types of problems and strategies.

Your task: Merge them into ONE comprehensive experience summary that:
1. Captures the common problem type and core strategy
2. Preserves the most useful specific techniques and decision rules
3. Includes both success patterns and failure pitfalls if mentioned
4. Is concise but complete — no redundancy

Output ONLY the merged experience summary, no preamble.

/no_think

---
{experiences}
---"""


def load_memory_bank(path: str) -> list[dict]:
    with open(path) as f:
        bank = json.load(f)
    if not isinstance(bank, list) or not bank:
        raise ValueError("Expected a non-empty JSON array of raw memory entries")
    # Embeddings are stale after summarization and account for most of the
    # in-memory footprint. They are removed from the final output anyway.
    for entry in bank:
        if not isinstance(entry, dict):
            raise ValueError("Every memory entry must be an object")
        entry.pop("embedding", None)
    from build_summary_data import is_valid_sid
    for i, entry in enumerate(bank):
        if not is_valid_sid(str(entry.get("sid", {}).get("tokens", ""))):
            raise ValueError(f"Invalid four-layer SID at row {i}")
        text = entry.get("experience_summary") or entry.get("experience_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing experience at row {i}")
    return bank


def group_by_sid(bank: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for entry in bank:
        sid = entry.get("sid", {})
        key = sid.get("tokens", "")
        if key:
            groups[key].append(entry)
    return dict(groups)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 2:
        return text[:limit]
    remaining = limit - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def build_merge_prompt(
    sid: str, texts: list[str], max_prompt_chars: int
) -> str:
    """Render a prompt that strictly respects the total character budget."""
    empty_experiences = "".join(
        f"\n### Experience {index + 1}\n\n" for index in range(len(texts))
    )
    fixed_size = len(
        MERGE_PROMPT.format(
            n=len(texts), sid=sid, experiences=empty_experiences
        )
    )
    content_budget = max_prompt_chars - fixed_size
    if content_budget <= 0:
        raise ValueError(
            f"--max-prompt-chars={max_prompt_chars} is too small for prompt headers"
        )

    per_text = max(1, content_budget // len(texts))
    trimmed = [_truncate_middle(text, per_text) for text in texts]
    exp_text = "".join(
        f"\n### Experience {index + 1}\n{text}\n"
        for index, text in enumerate(trimmed)
    )
    prompt = MERGE_PROMPT.format(
        n=len(trimmed), sid=sid, experiences=exp_text
    )
    # Integer division and variable index widths can leave a tiny overflow.
    while len(prompt) > max_prompt_chars and per_text > 1:
        per_text = max(1, per_text - (len(prompt) - max_prompt_chars + len(texts) - 1) // len(texts))
        trimmed = [_truncate_middle(text, per_text) for text in texts]
        exp_text = "".join(
            f"\n### Experience {index + 1}\n{text}\n"
            for index, text in enumerate(trimmed)
        )
        prompt = MERGE_PROMPT.format(
            n=len(trimmed), sid=sid, experiences=exp_text
        )
    if len(prompt) > max_prompt_chars:
        raise ValueError(
            f"Could not fit merge prompt into {max_prompt_chars} characters"
        )
    return prompt


def merge_experiences(client, model: str, sid: str,
                      entries: list[dict], max_entries: int = 30,
                      max_prompt_chars: int = 25000,
                      max_tokens: int = 2048,
                      max_retries: int = 3,
                      retry_backoff: float = 5.0) -> str:
    """Use LLM to merge multiple experiences into one."""
    if len(entries) == 1:
        return entries[0].get("experience_summary") or entries[0].get("experience_text", "")

    texts = []
    for e in entries:
        text = e.get("experience_summary") or e.get("experience_text", "")
        if text.strip():
            texts.append(text.strip())

    if not texts:
        return ""

    # Deduplicate exact matches
    texts = list(dict.fromkeys(texts))

    if len(texts) == 1:
        return texts[0]

    # If too many, sample
    if len(texts) > max_entries:
        import random
        texts = random.Random(42).sample(texts, max_entries)

    prompt = build_merge_prompt(sid, texts, max_prompt_chars)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            message = resp.choices[0].message
            content = message.content or ""
            if resp.choices[0].finish_reason == "length":
                raise ValueError("Summary hit the output limit; increase --max-tokens")
            if not content.strip() or "<think>" in content or "</think>" in content:
                raise ValueError("Expected final summary text without reasoning tags")
            return content.strip()
        except Exception as exc:
            last_error = exc
            print(
                f"  [RETRY {attempt}/{max_retries}] {sid}: {exc}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_backoff * attempt)
    raise RuntimeError(f"LLM merge failed after {max_retries} attempts: {last_error}")


def single_experience_text(entries: list[dict]) -> str:
    return entries[0].get("experience_summary") or entries[0].get("experience_text", "")


def save_checkpoint(path: str, merged: object) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def merge_sid_job(api_base: str, model: str, sid: str, entries: list[dict],
                  request_timeout: float, max_entries: int,
                  max_prompt_chars: int, max_tokens: int,
                  max_retries: int, retry_backoff: float) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=request_timeout)
    return sid, merge_experiences(
        client,
        model,
        sid,
        entries,
        max_entries=max_entries,
        max_prompt_chars=max_prompt_chars,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent SID merge requests.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save checkpoint after this many completed SIDs.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-entries", type=int, default=30, help="Maximum sampled experiences per SID sent to the LLM.")
    parser.add_argument("--max-prompt-chars", type=int, default=25000, help="Maximum prompt length before trimming sampled experiences.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens per merge request.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient LLM/API failures.")
    parser.add_argument("--retry-backoff", type=float, default=5.0, help="Linear retry backoff in seconds.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    for name in ("max_entries", "max_prompt_chars", "max_tokens", "max_retries"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.checkpoint is None:
        args.checkpoint = args.output + ".ckpt.json"

    paths = [os.path.realpath(p) for p in (args.input, args.output, args.checkpoint)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output and checkpoint must be different files")
    print(f"Loading memory bank from {args.input} ...")
    bank = load_memory_bank(args.input)
    print(f"  Total entries: {len(bank)}")

    groups = group_by_sid(bank)
    print(f"  Unique SIDs: {len(groups)}")

    # Load checkpoint if exists
    signature = hashlib.sha256(json.dumps({
        "bank": bank, "model": args.model, "prompt": MERGE_PROMPT,
        "max_entries": args.max_entries, "max_prompt_chars": args.max_prompt_chars,
        "max_tokens": args.max_tokens,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    merged = {}
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            saved = json.load(f)
        if saved.get("signature") != signature:
            raise ValueError("Checkpoint input/settings changed; use a new checkpoint path")
        merged = saved["merged"]
        print(f"  Resumed from checkpoint: {len(merged)} SIDs already done", flush=True)

    if set(merged) - set(groups) or any(not isinstance(v, str) or not v.strip() for v in merged.values()):
        raise ValueError("Checkpoint has unknown SIDs or empty summaries")

    sorted_sids = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    total = len(sorted_sids)
    done = len(merged)
    t0 = time.time()

    failures = {}
    pending = []
    for sid, entries in sorted_sids:
        if sid in merged:
            continue
        if len(entries) == 1:
            merged[sid] = single_experience_text(entries)
            done += 1
            if done % args.checkpoint_every == 0:
                save_checkpoint(args.checkpoint, {"signature": signature, "merged": merged})
        else:
            pending.append((sid, entries))

    if pending:
        print(f"  Merging {len(pending)} SIDs with {args.workers} worker(s) ...", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sid = {}
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                sid, entries = next(pending_iter)
            except StopIteration:
                return False
            n = len(entries)
            print(f"  Submit {sid} ({n} entries)", flush=True)
            future = executor.submit(
                merge_sid_job,
                args.api_base,
                args.model,
                sid,
                entries,
                args.request_timeout,
                args.max_entries,
                args.max_prompt_chars,
                args.max_tokens,
                args.max_retries,
                args.retry_backoff,
            )
            future_to_sid[future] = (sid, n)
            return True

        for _ in range(args.workers):
            submit_next()

        while future_to_sid:
            completed, _ = wait(future_to_sid, return_when=FIRST_COMPLETED)
            for future in completed:
                sid, n = future_to_sid.pop(future)
                try:
                    sid, text = future.result()
                except Exception as e:
                    print(f"  [ERROR] {sid}: {e}", flush=True)
                    failures[sid] = str(e)
                    submit_next()
                    continue

                merged[sid] = text
                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{done}/{total}] Merged {sid} ({n} entries, {len(text)} chars, {rate:.0f}/min)", flush=True)

                if done % args.checkpoint_every == 0:
                    save_checkpoint(args.checkpoint, {"signature": signature, "merged": merged})

                submit_next()

    save_checkpoint(args.checkpoint, {"signature": signature, "merged": merged})

    if failures:
        raise RuntimeError(f"{len(failures)} SIDs failed; successful work was checkpointed. Rerun to retry.")

    # Build final merged memory bank
    print(f"\nBuilding merged memory bank ...")
    merged_bank = []
    for sid, entries in groups.items():
        template = entries[0].copy()
        template["experience_summary"] = merged.get(sid, "")
        template["experience_text"] = merged.get(sid, "")
        template.pop("embedding", None)
        template["merged_from_count"] = len(entries)
        merged_bank.append(template)

    save_checkpoint(args.output, merged_bank)

    print(f"  Saved {len(merged_bank)} merged entries to {args.output}")
    print(f"  Original: {len(bank)} entries across {len(groups)} SIDs")
    print(f"  Merged:   {len(merged_bank)} entries (1 per SID)")
    print(f"  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
