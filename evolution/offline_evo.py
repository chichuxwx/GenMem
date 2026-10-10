"""
offline_evo.py  —  OfflineEvo: Self-bootstrapping memory evolution pipeline.

One iteration:
  For every sample in the dataset:
    1. MemR(question, memory_iter_N)  →  curated_experience + predicted_SID
    2. SP(question + curated_experience)  →  trajectory + sp_answer + sp_success
    3. MemE(trajectory, memory_iter_N)   →  evolved_experience + target_SID
  Collect all (target_SID, evolved_experience) updates.
  Apply updates to memory_iter_N  →  save as memory_iter_{N+1}.

Dataset format (JSON list):
  [
    {"question": "...", "ground_truth_answer": "...", "dataset": "triviaqa"},
    ...
  ]
  Also accepts parquet with 'extra_info' column (mem_retriever_processed format).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memory_manager as mm
from answer_scoring import answers_match
from agent_parallel import transformers_agent_worker
from data_parallel import parse_gpu_ids, run_data_parallel
from gt_evolution_helpers import build_meme_trajectory
from mem_agent_infer import MemAgentInfer
from memory_update_gate import memory_update_rejection_reason
from sp_runner import run_sp_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("offline_evo")

SID_RESULT_KEYS = ("sid_l1", "sid_l2", "sid_l3", "sid_l4")


def result_sid(result: dict) -> tuple:
    return tuple(result.get(key) for key in SID_RESULT_KEYS)


def format_sid(values: tuple) -> str:
    return "".join(
        f"<SID_L{level}_{value}>" for level, value in enumerate(values, start=1)
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """
    Load dataset from JSON or parquet.

    Returns list of dicts with at least:
        question            : str
        ground_truth_answer : str (may be empty)
        dataset             : str
    """
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd
        import numpy as np

        df = pd.read_parquet(path)
        samples = []
        for _, row in df.iterrows():
            ei = row.get("extra_info", {})
            if isinstance(ei, (np.ndarray, list)):
                ei = ei[0] if len(ei) > 0 else {}
            if not isinstance(ei, dict):
                try:
                    ei = dict(ei)
                except Exception:
                    ei = {}

            rm = row.get("reward_model", {})
            if isinstance(rm, (np.ndarray, list)):
                rm = rm[0] if len(rm) > 0 else {}
            if not isinstance(rm, dict):
                try:
                    rm = dict(rm)
                except Exception:
                    rm = {}

            gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
            if not isinstance(gt, dict):
                gt = {}

            samples.append({
                "sample_id": str(ei.get("sample_id", row.get("id", "")) or ""),
                "question": str(ei.get("question", "")),
                "ground_truth_answer": str(gt.get("target_answer", "") or ""),
                "dataset": str(ei.get("dataset", row.get("data_source", "unknown"))),
            })
        logger.info(f"Loaded {len(samples)} samples from parquet: {path}")
        return samples

    elif p.suffix == ".json" or p.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            if p.suffix == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        # Normalize keys
        samples = []
        for d in data:
            samples.append({
                "sample_id": str(d.get("sample_id", d.get("id", d.get("_id", ""))) or ""),
                "question": str(d.get("question", d.get("query", ""))),
                "ground_truth_answer": str(d.get("ground_truth_answer",
                                                   d.get("ground_truth",
                                                         d.get("answer", d.get("target_answer", ""))))),
                "dataset": str(d.get("dataset", "unknown")),
            })
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}")


# ---------------------------------------------------------------------------
# Progress / checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str) -> dict:
    """Load per-sample checkpoint for a given iteration."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ckpt_path)


def save_iter_results(results_path: str, results: list[dict]) -> None:
    """Save per-sample results for an iteration (for analysis)."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _compact(value, limit: int = 1000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def save_iteration_observations(
    iter_dir: str,
    iter_idx: int,
    results: list[dict],
    sp_workers: int,
    update_sid_source: str,
    tracked_cases: int,
) -> dict:
    """Persist compact metrics and a fixed small set of inspectable cases."""
    rows = [row for row in results if row is not None]
    total = len(rows)
    request_rows = [row for row in rows if row.get("sp_request_success") is not None]
    request_successes = sum(bool(row.get("sp_request_success")) for row in request_rows)
    error_counts = Counter(
        str(row.get("sp_error") or "")
        for row in request_rows
        if not row.get("sp_request_success")
    )
    metrics = {
        "iteration": iter_idx,
        "samples": total,
        "sp_workers": sp_workers,
        "update_sid_source": update_sid_source,
        "answer_scorer": "structured_normalized_v2",
        "answer_correct": sum(bool(row.get("sp_success")) for row in rows),
        "answer_accuracy": (
            sum(bool(row.get("sp_success")) for row in rows) / total if total else 0.0
        ),
        "answer_empty": sum(not str(row.get("sp_answer") or "").strip() for row in rows),
        "api_requests_observed": len(request_rows),
        "api_request_successes": request_successes,
        "api_request_success_rate": (
            request_successes / len(request_rows) if request_rows else None
        ),
        "api_failure_rate": (
            1.0 - request_successes / len(request_rows) if request_rows else None
        ),
        "api_retry_count": sum(int(row.get("sp_retry_count") or 0) for row in rows),
        "api_errors": dict(error_counts),
        "valid_memr_sid": sum(
            isinstance(row.get("memr_sid"), (list, tuple))
            and len(row["memr_sid"]) == 4
            and all(value is not None for value in row["memr_sid"])
            for row in rows
        ),
        "valid_meme_sid": sum(
            isinstance(row.get("meme_sid"), (list, tuple))
            and len(row["meme_sid"]) == 4
            and all(value is not None for value in row["meme_sid"])
            for row in rows
        ),
        "meme_sources": dict(Counter(str(row.get("meme_source")) for row in rows)),
        "memory_update_gate_passed": sum(
            bool(row.get("memory_update_gate_passed")) for row in rows
        ),
        "memory_update_gate_rejections": dict(Counter(
            str(row.get("memory_update_gate_reason"))
            for row in rows
            if not row.get("memory_update_gate_passed")
        )),
    }
    # Keep benchmark-level metrics in every iteration.  These are the
    # quantities used by the online test-time-evolution report; GT is only
    # consumed by the scorer above, never by the model inputs.
    per_dataset = {}
    for dataset in sorted({str(row.get("dataset") or "unknown") for row in rows}):
        subset = [row for row in rows if str(row.get("dataset") or "unknown") == dataset]
        correct = sum(bool(row.get("sp_success")) for row in subset)
        per_dataset[dataset] = {
            "samples": len(subset),
            "answer_correct": correct,
            "answer_accuracy": correct / len(subset) if subset else 0.0,
            "api_request_success_rate": (
                sum(bool(row.get("sp_request_success", True)) for row in subset) / len(subset)
                if subset else None
            ),
        }
    metrics["per_dataset"] = per_dataset
    with open(os.path.join(iter_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    cases = []
    for row in rows[: max(0, tracked_cases)]:
        cases.append({
            "global_idx": row.get("global_idx"),
            "question": row.get("question"),
            "ground_truth_answer": row.get("ground_truth_answer"),
            "memr_sid": row.get("memr_sid"),
            "memr_experience": _compact(row.get("memr_answer")),
            "retrieved_context": _compact(row.get("memr_retrieved_context")),
            "sp_answer": row.get("sp_answer"),
            "sp_success": row.get("sp_success"),
            "sp_request_success": row.get("sp_request_success"),
            "meme_sid": row.get("meme_sid"),
            "meme_source": row.get("meme_source"),
            "memory_update_gate_passed": row.get("memory_update_gate_passed"),
            "memory_update_gate_reason": row.get("memory_update_gate_reason"),
            "evolved_experience": _compact(row.get("evolved_experience")),
        })
    with open(os.path.join(iter_dir, "cases.json"), "w", encoding="utf-8") as handle:
        json.dump(cases, handle, ensure_ascii=False, indent=2)
    return metrics


def save_metrics_history(output_dir: str) -> None:
    metrics = []
    for path in Path(output_dir).glob("iter_*/metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            results_path = path.parent / "results.json"
            if results_path.is_file():
                results = json.loads(results_path.read_text(encoding="utf-8"))
                rescored = [
                    answers_match(
                        result.get("sp_answer", ""),
                        result.get("ground_truth_answer", ""),
                    )
                    for result in results
                ]
                gate_reasons = Counter(
                    reason
                    for result in results
                    if (reason := memory_update_rejection_reason(
                        result.get("evolved_experience", ""),
                        sp_answer=result.get("sp_answer", ""),
                        ground_truth_answer=result.get("ground_truth_answer", ""),
                    ))
                )
                row.update({
                    "answer_correct": sum(rescored),
                    "answer_accuracy": sum(rescored) / len(rescored) if rescored else 0.0,
                    "answer_scorer": "structured_normalized_v2",
                    "memory_update_gate_passed": len(results) - sum(gate_reasons.values()),
                    "memory_update_gate_rejections": dict(gate_reasons),
                })
                path.write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            metrics.append(row)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read iteration metrics: %s", path)
    metrics.sort(key=lambda row: int(row.get("iteration", 0)))
    Path(output_dir, "metrics_history.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# OfflineEvo metrics",
        "",
        "| iter | workers | answer acc | API success | empty | valid MemR SID | valid MemE SID | gate pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        api_rate = row.get("api_request_success_rate")
        api_text = "n/a" if api_rate is None else f"{100 * api_rate:.1f}%"
        lines.append(
            f"| {row['iteration']} | {row['sp_workers']} | "
            f"{100 * row['answer_accuracy']:.1f}% | {api_text} | "
            f"{row['answer_empty']} | {row['valid_memr_sid']} | {row['valid_meme_sid']} | "
            f"{row.get('memory_update_gate_passed', 'n/a')} |"
        )
    Path(output_dir, "metrics_history.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_best_by_benchmark(output_dir: str, metrics: dict) -> None:
    """Update the running best score for each benchmark.

    This is deliberately a reporting-only artifact: benchmark ground truth is
    used by the scorer/write gate, but no reference answer is placed in model
    inputs or in the memory bank.  Keeping the best per benchmark (rather than
    only the best joint iteration) is useful for online test-time evolution,
    where different datasets can peak at different iterations.
    """
    path = Path(output_dir, "best_by_benchmark.json")
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    for dataset, row in (metrics.get("per_dataset") or {}).items():
        if not isinstance(row, dict):
            continue
        candidate = dict(row)
        candidate["iteration"] = metrics.get("iteration")
        previous = current.get(dataset)
        if (
            not isinstance(previous, dict)
            or float(candidate.get("answer_accuracy", 0.0))
            > float(previous.get("answer_accuracy", -1.0))
            or (
                float(candidate.get("answer_accuracy", 0.0))
                == float(previous.get("answer_accuracy", -1.0))
                and int(candidate.get("iteration", 0)) < int(previous.get("iteration", 0))
            )
        ):
            current[dataset] = candidate
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Best online-evolution score by benchmark",
        "",
        "GT is used only by the external scorer/write gate; it is not sent to MemR, the server model, or MemE and is never written to the memory bank.",
        "",
        "| benchmark | best iteration | correct | samples | accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted(current):
        row = current[dataset]
        lines.append(
            f"| {dataset} | {row.get('iteration', '')} | "
            f"{row.get('answer_correct', '')} | {row.get('samples', '')} | "
            f"{100 * float(row.get('answer_accuracy', 0.0)):.2f}% |"
        )
    Path(output_dir, "best_by_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core iteration
# ---------------------------------------------------------------------------

def run_one_iter(
    iter_idx: int,
    samples: list[dict],
    bank: list[dict],
    index: dict,
    memr_model_path: str,
    meme_model_path: str,
    output_dir: str,
    batch_size: int,
    sp_workers: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    inference_backend: str,
    max_turns: int,
    max_new_tokens: int,
    meme_max_new_tokens: int,
    memr_sid_beams: int,
    memr_beam_fusion: str,
    memr_reranker_model: str,
    memr_rerank_top_k: int,
    memr_reranker_batch_size: int,
    memr_reranker_max_length: int,
    update_sid_source: str,
    tracked_cases: int,
    max_api_failure_rate: float,
    expose_ground_truth_to_meme: bool,
    gpu_ids: list[int] | None = None,
    lazy_update_every: int = 0,
) -> tuple[list[dict], dict, dict]:
    """
    Run one full evolution iteration over all samples.

    Returns (new_bank, new_index) after applying evolved experience updates.
    """
    iter_dir = os.path.join(output_dir, f"iter_{iter_idx}")
    os.makedirs(iter_dir, exist_ok=True)
    ckpt_path = os.path.join(iter_dir, "checkpoint.json")
    checkpoint = load_checkpoint(ckpt_path)

    all_results: list[dict] = checkpoint.get("results", [])
    done_indices: set = set(checkpoint.get("done_indices", []))
    lazy = None
    if checkpoint.get("lazy_memory") and not lazy_update_every:
        raise ValueError("Checkpoint contains lazy updates; resume with --lazy-update-every")
    if lazy_update_every:
        from lazy_memory import LazyMemory
        lazy = LazyMemory(bank, index, checkpoint.get("lazy_memory"))
        # Existing pre-lazy checkpoints are migrated explicitly at startup.
        migrated = lazy.flush(all_results, update_sid_source, lazy_update_every)
        bank, index = lazy.bank, lazy.index
        if migrated:
            save_checkpoint(ckpt_path, {"results": all_results,
                            "done_indices": sorted(done_indices), "lazy_memory": lazy.state()})
        logger.info("Lazy memory ready: %s commits, %s consumed samples; interval=%s",
                    len(lazy.journal), len(lazy.applied), lazy_update_every)

    logger.info(f"=== Iteration {iter_idx}: {len(samples)} samples "
                f"({len(done_indices)} already done) ===")

    # --- Process in batches ---
    pending = [(i, s) for i, s in enumerate(samples) if i not in done_indices]
    logger.info(f"Pending samples: {len(pending)}")

    # Extend all_results list to correct length if needed
    while len(all_results) < len(samples):
        all_results.append(None)

    # Process one batch at a time (load model → process → unload → next model)
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        batch_indices = [idx for idx, _ in batch]
        batch_samples = [s for _, s in batch]

        logger.info(f"  Batch {batch_start//batch_size + 1}: "
                    f"samples {batch_indices[0]}–{batch_indices[-1]}")

        # ---- Step 1: MemR inference ----
        t0 = time.time()
        questions = [s["question"] for s in batch_samples]
        logger.info("  Loading MemR model...")
        remote_memr_urls = [
            item.strip()
            for item in os.getenv("REMOTE_MEMR_URLS", "").split(",")
            if item.strip()
        ]
        if remote_memr_urls and gpu_ids:
            from remote_memr_client import run_hybrid_memr_batch

            logger.info(
                "  MemR hybrid pool: local GPUs %s + remote %s",
                gpu_ids,
                remote_memr_urls,
            )
            memr_results = run_hybrid_memr_batch(
                questions,
                local_gpu_ids=gpu_ids,
                remote_urls=remote_memr_urls,
                local_worker_fn=transformers_agent_worker,
                local_worker_kwargs={
                    "role": "memr",
                    "model_path": memr_model_path,
                    "bank": bank,
                    "index": index,
                    "max_turns": max_turns,
                    "max_new_tokens": max_new_tokens,
                    "memr_sid_beams": memr_sid_beams,
                    "memr_beam_fusion": memr_beam_fusion,
                    "memr_reranker_path": memr_reranker_model or None,
                    "memr_rerank_top_k": memr_rerank_top_k,
                    "memr_reranker_batch_size": memr_reranker_batch_size,
                    "memr_reranker_max_length": memr_reranker_max_length,
                },
            )
        elif remote_memr_urls:
            from remote_memr_client import run_remote_memr_batch

            logger.info("  MemR remote pool on %s", remote_memr_urls)
            memr_results = run_remote_memr_batch(questions, remote_memr_urls)
        elif gpu_ids:
            logger.info("  MemR data parallel on physical GPUs %s", gpu_ids)
            memr_results = run_data_parallel(
                questions,
                gpu_ids=gpu_ids,
                worker_fn=transformers_agent_worker,
                worker_kwargs={
                    "role": "memr",
                    "model_path": memr_model_path,
                    "bank": bank,
                    "index": index,
                    "max_turns": max_turns,
                    "max_new_tokens": max_new_tokens,
                    "memr_sid_beams": memr_sid_beams,
                    "memr_beam_fusion": memr_beam_fusion,
                    "memr_reranker_path": memr_reranker_model or None,
                    "memr_rerank_top_k": memr_rerank_top_k,
                    "memr_reranker_batch_size": memr_reranker_batch_size,
                    "memr_reranker_max_length": memr_reranker_max_length,
                },
            )
        else:
            infer_cls = MemAgentInfer
            if inference_backend == "transformers":
                from mem_agent_transformers import MemAgentTransformersInfer

                infer_cls = MemAgentTransformersInfer
            memr_extra_kwargs = {}
            if inference_backend == "transformers":
                memr_extra_kwargs = {
                    "memr_reranker_path": memr_reranker_model or None,
                    "memr_rerank_top_k": memr_rerank_top_k,
                    "memr_reranker_batch_size": memr_reranker_batch_size,
                    "memr_reranker_max_length": memr_reranker_max_length,
                }
            memr_infer = infer_cls(
                model_path=memr_model_path,
                bank=bank,
                index=index,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                memr_sid_beams=memr_sid_beams,
                memr_beam_fusion=memr_beam_fusion,
                **memr_extra_kwargs,
            )
            memr_results = memr_infer.run_memr_batch(questions)
            memr_infer.unload()
        logger.info(f"  MemR done in {time.time()-t0:.1f}s")

        # ---- Step 2: SP call ----
        t0 = time.time()
        sp_inputs = []
        for s, mr in zip(batch_samples, memr_results):
            sp_inputs.append({
                "question": s["question"],
                "curated_experience": mr["answer"],
                "ground_truth_answer": s.get("ground_truth_answer", ""),
            })
        sp_results = run_sp_batch(sp_inputs, max_workers=sp_workers)
        request_successes = sum(
            bool(result.get("sp_request_success", True)) for result in sp_results
        )
        request_success_rate = request_successes / len(sp_results)
        logger.info(f"  SP done in {time.time()-t0:.1f}s | "
                    f"answer_accuracy={sum(r['sp_success'] for r in sp_results)/len(sp_results):.2%} | "
                    f"request_success_rate={request_success_rate:.2%}")

        # A provider outage must not turn into a completed, empty batch. Abort
        # before MemE inference and before checkpointing so a later restart can
        # retry the same indices from the previous valid checkpoint.
        batch_failure_rate = 1.0 - request_success_rate
        if batch_failure_rate > max_api_failure_rate:
            raise RuntimeError(
                f"Batch API failure rate {batch_failure_rate:.2%} exceeds "
                f"the configured limit {max_api_failure_rate:.2%}; "
                "batch was not checkpointed"
            )

        # ---- Step 3: MemE inference ----
        t0 = time.time()
        logger.info("  Loading MemE model...")
        trajectory_texts = [
            build_meme_trajectory(
                result["trajectory_text"],
                sample,
                expose_ground_truth=expose_ground_truth_to_meme,
            )
            for result, sample in zip(sp_results, batch_samples)
        ]
        if gpu_ids:
            logger.info("  MemE data parallel on physical GPUs %s", gpu_ids)
            from remote_meme_client import run_hybrid_meme_batch
            meme_results = run_hybrid_meme_batch(
                trajectory_texts,
                gpu_ids=gpu_ids,
                remote_urls=[u.strip() for u in os.getenv("REMOTE_MEME_URLS", "").split(",") if u.strip()],
                worker_fn=transformers_agent_worker,
                worker_kwargs={
                    "role": "meme",
                    "model_path": meme_model_path,
                    "bank": bank,
                    "index": index,
                    "max_turns": max_turns,
                    "max_new_tokens": meme_max_new_tokens,
                    "memr_sid_beams": memr_sid_beams,
                    "memr_beam_fusion": memr_beam_fusion,
                },
            )
        else:
            meme_infer = infer_cls(
                model_path=meme_model_path,
                bank=bank,
                index=index,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_turns=max_turns,
                max_new_tokens=meme_max_new_tokens,
                memr_sid_beams=memr_sid_beams,
                memr_beam_fusion=memr_beam_fusion,
            )
            meme_results = meme_infer.run_meme_batch(trajectory_texts)
            meme_infer.unload()
        logger.info(f"  MemE done in {time.time()-t0:.1f}s")

        # The exported MemE checkpoint occasionally emits the worked arithmetic
        # example or stops before producing a SID.  For long evolution studies,
        # optionally repair only those invalid outputs with a direct API editor.
        # The editor repairs text only; it must not invent a MemE routing SID.
        if os.getenv("MEME_FALLBACK_DIRECT", "0") == "1":
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from sp_runner import run_direct_memory_evolution

            repair_jobs = []
            for i, (mr, spr, mer) in enumerate(zip(memr_results, sp_results, meme_results)):
                invalid = (
                    any(mer.get(k) is None for k in SID_RESULT_KEYS)
                    or "For multi-step arithmetic problems" in str(mer.get("answer", ""))
                )
                if not invalid or any(mr.get(k) is None for k in SID_RESULT_KEYS):
                    continue
                memr_sid = format_sid(result_sid(mr))
                existing = mm.lookup_sid(memr_sid, bank, index)
                repair_jobs.append((i, spr, mer, existing))

            def _repair_one(job):
                i, spr, mer, existing = job
                evolved = run_direct_memory_evolution(
                    trajectory_texts[i],
                    existing,
                    timeout=int(os.getenv("DIRECT_MEMORY_TIMEOUT", "15")),
                )
                return i, mer, evolved

            repaired = 0
            logger.info(
                "  Repairing %s invalid MemE outputs with direct API (workers=%s)",
                len(repair_jobs),
                sp_workers,
            )
            with ThreadPoolExecutor(max_workers=sp_workers) as executor:
                futures = [executor.submit(_repair_one, job) for job in repair_jobs]
                for future in as_completed(futures):
                    _, mer, evolved = future.result()
                    if evolved:
                        mer["answer"] = evolved
                        # The fallback editor repairs text but does not make a
                        # MemE routing decision. MemE-target runs skip it;
                        # MemR-target runs may write it to the retrieved SID.
                        for key in SID_RESULT_KEYS:
                            mer[key] = None
                        mer["fallback_source"] = "direct_api"
                        repaired += 1
                    else:
                        # Do not let the original invalid MemE response pass
                        # through when the fallback service fails (e.g. 502).
                        mer["answer"] = ""
                        for key in SID_RESULT_KEYS:
                            mer[key] = None
                        mer["fallback_source"] = "fallback_failed"
            logger.info("  Repaired invalid MemE outputs with direct API: %s", repaired)

        # ---- Collect batch results ----
        for local_i, (global_idx, s, mr, spr, mer) in enumerate(
            zip(batch_indices, batch_samples, memr_results, sp_results, meme_results)
        ):
            memr_sid = None
            if all(mr.get(key) is not None for key in SID_RESULT_KEYS):
                memr_sid = format_sid(result_sid(mr))
            rec = {
                "global_idx": global_idx,
                "iteration": iter_idx,
                "question": s["question"],
                "dataset": s.get("dataset", ""),
                "ground_truth_answer": s.get("ground_truth_answer", ""),
                # MemR outputs
                "memr_answer": mr["answer"],
                "memr_sid": result_sid(mr),
                "memr_candidate_sids": mr.get("candidate_sids", []),
                "memr_retrieved_sids": mr.get("retrieved_sids", []),
                "memr_rerank_scores": mr.get("rerank_scores", []),
                "memr_branch_summaries": mr.get("branch_summaries", []),
                "memr_fusion_strategy": mr.get("fusion_strategy", memr_beam_fusion),
                "memr_retrieved_context": mm.lookup_sid(memr_sid, bank, index) if memr_sid else "",
                # SP outputs
                "sp_answer": spr["sp_answer"],
                "sp_success": spr["sp_success"],
                "sp_request_success": spr.get("sp_request_success"),
                "sp_error": spr.get("sp_error", ""),
                "sp_retry_count": spr.get("sp_retry_count", 0),
                "trajectory_text": spr["trajectory_text"],
                # MemE outputs
                "evolved_experience": mer["answer"],
                "meme_sid": result_sid(mer),
                "meme_source": mer.get("fallback_source", "meme_model"),
                "meme_ground_truth_exposed": expose_ground_truth_to_meme,
            }
            gate_reason = memory_update_rejection_reason(
                rec["evolved_experience"],
                sp_answer=rec["sp_answer"],
                ground_truth_answer=rec["ground_truth_answer"],
            )
            rec["memory_update_gate_passed"] = not gate_reason
            rec["memory_update_gate_reason"] = gate_reason
            all_results[global_idx] = rec
            done_indices.add(global_idx)

        if lazy is not None:
            if lazy.flush(all_results, update_sid_source, lazy_update_every):
                logger.info("Lazy memory commit %s: %s consumed samples",
                            len(lazy.journal), len(lazy.applied))
            bank, index = lazy.bank, lazy.index

        # Save checkpoint after each batch, atomically with the memory journal.
        save_checkpoint(ckpt_path, {
            "results": all_results,
            "done_indices": list(done_indices),
            **({"lazy_memory": lazy.state()} if lazy is not None else {}),
        })
        logger.info(f"  Checkpoint saved ({len(done_indices)}/{len(samples)} total done)")

    # Re-score checkpointed rows as well as newly generated rows.  This is
    # important when resuming an iteration created by an older scorer.
    for rec in all_results:
        if rec is None:
            continue
        rec["sp_success"] = answers_match(
            rec.get("sp_answer", ""), rec.get("ground_truth_answer", "")
        )
        gate_reason = memory_update_rejection_reason(
            rec.get("evolved_experience", ""),
            sp_answer=rec.get("sp_answer", ""),
            ground_truth_answer=rec.get("ground_truth_answer", ""),
        )
        rec["memory_update_gate_passed"] = not gate_reason
        rec["memory_update_gate_reason"] = gate_reason

    # ---- Save iteration results ----
    results_path = os.path.join(iter_dir, "results.json")
    save_iter_results(results_path, [r for r in all_results if r is not None])
    logger.info(f"Iteration {iter_idx} results saved to {results_path}")

    metrics = save_iteration_observations(
        iter_dir=iter_dir,
        iter_idx=iter_idx,
        results=all_results,
        sp_workers=sp_workers,
        update_sid_source=update_sid_source,
        tracked_cases=tracked_cases,
    )
    save_metrics_history(output_dir)
    logger.info(
        "Iteration %s metrics: answer_accuracy=%.2f%%, API success=%s",
        iter_idx,
        100 * metrics["answer_accuracy"],
        "n/a" if metrics["api_request_success_rate"] is None
        else f"{100 * metrics['api_request_success_rate']:.2f}%",
    )

    # ---- Build updates list for memory bank ----
    # The write target is explicit because experiments need to compare whether
    # evolved content is more useful at MemR's reachable SID or MemE's SID.
    updates = []
    for rec in all_results:
        if rec is None:
            continue
        target_sid = rec.get(f"{update_sid_source}_sid") or (None,) * 4
        sid_l1, sid_l2, sid_l3, sid_l4 = target_sid
        evolved = str(rec.get("evolved_experience") or "").strip()
        meme_source = rec.get("meme_source", "meme_model")
        invalid_text = (
            "<think>" in evolved.lower()
            or "For multi-step arithmetic problems" in evolved
        )
        gate_reason = memory_update_rejection_reason(
            evolved,
            sp_answer=rec.get("sp_answer", ""),
            ground_truth_answer=rec.get("ground_truth_answer", ""),
        )
        rec["memory_update_gate_passed"] = not gate_reason
        rec["memory_update_gate_reason"] = gate_reason
        # MemE must provide a valid card. The selected routing component must
        # provide a valid target SID; fallback-repaired cards are allowed when
        # MemR owns the target SID.
        if (
            any(x is None for x in (sid_l1, sid_l2, sid_l3, sid_l4))
            or not evolved
            or meme_source == "fallback_failed"
            or invalid_text
            or gate_reason
        ):
            continue
        updates.append({
            "sid_l1": sid_l1,
            "sid_l2": sid_l2,
            "sid_l3": sid_l3,
            "sid_l4": sid_l4,
            "evolved_experience": evolved,
            "sp_score": 1.0 if rec["sp_success"] else 0.5,
            "question": rec["question"],
            "dataset": rec.get("dataset", ""),
        })

    logger.info(
        "Collected %s memory updates from iteration %s (%s SID targets)",
        len(updates), iter_idx, update_sid_source.upper(),
    )
    gate_rejections = Counter(
        str(rec.get("memory_update_gate_reason"))
        for rec in all_results
        if rec is not None and rec.get("memory_update_gate_reason")
    )
    if gate_rejections:
        logger.info("Rejected memory updates by answer-leakage gate: %s", dict(gate_rejections))

    # ---- Apply updates ----
    if lazy is not None:
        lazy.flush(all_results, update_sid_source, lazy_update_every, force=True)
        save_checkpoint(ckpt_path, {"results": all_results,
                        "done_indices": sorted(done_indices), "lazy_memory": lazy.state()})
        return lazy.bank, lazy.index, metrics
    new_bank, new_index = mm.update_memory_bank(bank, updates, copy_bank=False)

    return new_bank, new_index, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    samples = load_dataset(args.dataset)
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
        logger.info(f"Capped to {len(samples)} samples (--max_samples)")

    memory_dir = os.path.join(args.output_dir, "memory")
    resume_from = -1
    if args.resume_evo:
        snapshots = sorted(Path(memory_dir).glob("memory_iter_*/memory_bank.json"))
        if snapshots:
            resume_from = max(int(p.parent.name.rsplit("_", 1)[-1]) for p in snapshots)

    if resume_from >= 0:
        resume_path = os.path.join(memory_dir, f"memory_iter_{resume_from}", "memory_bank.json")
        logger.info(f"Resuming from memory_iter_{resume_from}: {resume_path}")
        bank, index = mm.load_memory_bank(resume_path)
    else:
        logger.info(f"Loading base memory bank from {args.memory_bank}")
        bank, index = mm.load_memory_bank(args.memory_bank)
        mm.backup_memory(bank, memory_dir, iter_idx=0)
        logger.info(f"Original memory backed up as memory_iter_0")

    # Run iterations. Direct-API concurrency can step down between iterations
    # when throttling remains high after retries.
    current_sp_workers = args.sp_workers
    for iter_idx in range(max(1, resume_from + 1), args.num_iters + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting iteration {iter_idx} / {args.num_iters}")
        logger.info(f"{'='*60}")

        bank, index, metrics = run_one_iter(
            iter_idx=iter_idx,
            samples=samples,
            bank=bank,
            index=index,
            memr_model_path=args.memr_model,
            meme_model_path=args.meme_model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sp_workers=current_sp_workers,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            inference_backend=args.inference_backend,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            meme_max_new_tokens=(
                args.meme_max_new_tokens
                if args.meme_max_new_tokens > 0
                else args.max_new_tokens
            ),
            memr_sid_beams=args.memr_sid_beams,
            memr_beam_fusion=args.memr_beam_fusion,
            memr_reranker_model=args.memr_reranker_model,
            memr_rerank_top_k=args.memr_rerank_top_k,
            memr_reranker_batch_size=args.memr_reranker_batch_size,
            memr_reranker_max_length=args.memr_reranker_max_length,
            update_sid_source=args.update_sid_source,
            tracked_cases=args.tracked_cases,
            max_api_failure_rate=args.max_api_failure_rate,
            expose_ground_truth_to_meme=args.expose_ground_truth_to_meme,
            gpu_ids=parse_gpu_ids(args.gpu_ids) if args.gpu_ids else None,
            lazy_update_every=args.lazy_update_every,
        )
        update_best_by_benchmark(args.output_dir, metrics)

        failure_rate = metrics.get("api_failure_rate")
        if (
            args.adaptive_sp_workers
            and failure_rate is not None
            and failure_rate > args.max_api_failure_rate
            and current_sp_workers > args.min_sp_workers
        ):
            previous = current_sp_workers
            current_sp_workers = max(args.min_sp_workers, current_sp_workers // 2)
            logger.warning(
                "API failure rate %.2f%% exceeds %.2f%%; reducing SP workers %s -> %s",
                100 * failure_rate,
                100 * args.max_api_failure_rate,
                previous,
                current_sp_workers,
            )

        # Keep per-iteration results/checkpoints, but snapshot the large bank
        # only at the requested cadence (and always at the final iteration).
        if iter_idx % args.snapshot_every == 0 or iter_idx == args.num_iters:
            mm.backup_memory(bank, memory_dir, iter_idx=iter_idx)
            logger.info(f"memory_iter_{iter_idx} saved ({len(bank)} entries)")
        else:
            next_snapshot = iter_idx + (args.snapshot_every - iter_idx % args.snapshot_every)
            logger.info(
                f"memory_iter_{iter_idx} not snapshotted; next snapshot at iter_{next_snapshot}"
            )

    logger.info("\nOfflineEvo complete.")
    logger.info(f"Memory snapshots: {args.output_dir}/memory/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OfflineEvo: self-bootstrapping memory evolution")

    # Data
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset (.json/.jsonl/.parquet)")
    parser.add_argument("--memory_bank",
                        required=True,
                        help="Path to base memory_bank.json")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit dataset size (0 = no limit)")

    # Models
    parser.add_argument("--memr_model",
                        required=True,
                        help="MemRetriever model path")
    parser.add_argument("--meme_model",
                        required=True,
                        help="MemEvolver model path")

    # Output
    parser.add_argument("--output_dir",
                        required=True,
                        help="Output directory for results and memory snapshots")

    # Iteration
    parser.add_argument("--num_iters", type=int, default=3,
                        help="Number of evolution iterations")
    parser.add_argument("--resume-evo", action=argparse.BooleanOptionalAction, default=False,
                        help="Resume from the latest saved memory_iter_N snapshot")
    parser.add_argument("--snapshot-every", type=int, default=5,
                        help="Save memory bank snapshots every N iterations (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of samples per inference batch")
    parser.add_argument("--lazy-update-every", type=int, default=0,
                        help="Flush memory after this many processed samples at a batch boundary; 0 disables")
    parser.add_argument("--update-sid-source", choices=["meme", "memr"], default="meme",
                        help="Choose whether evolved content is written to MemE or MemR SID")
    parser.add_argument(
        "--expose-ground-truth-to-meme",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the reference answer to MemE's evolution trajectory only. "
            "MemR and the solver never receive this privileged supervision."
        ),
    )
    parser.add_argument("--tracked-cases", type=int, default=5,
                        help="Save this many fixed cases per iteration")

    # SP
    parser.add_argument("--sp_workers", type=int, default=4,
                        help="Parallel SP subprocess workers")
    parser.add_argument("--adaptive-sp-workers", action=argparse.BooleanOptionalAction,
                        default=False, help="Reduce SP concurrency after API throttling")
    parser.add_argument("--min-sp-workers", type=int, default=2,
                        help="Minimum SP workers used by adaptive concurrency")
    parser.add_argument("--max-api-failure-rate", type=float, default=0.10,
                        help="Reduce SP workers when API failure rate exceeds this value")

    # vLLM
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallel size (GPUs per model)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--inference_backend", choices=["vllm", "transformers"], default="vllm",
                        help="Model inference backend")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum MemR/MemE tool-loop turns")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum generated tokens per tool-loop turn")
    parser.add_argument(
        "--meme-max-new-tokens",
        type=int,
        default=0,
        help=(
            "MemE-only generated-token limit per tool-loop turn; "
            "0 reuses --max_new_tokens."
        ),
    )
    parser.add_argument(
        "--memr-sid-beams", type=int, default=1,
        help="Number of deterministic MemR SID beams in the Transformers backend.",
    )
    parser.add_argument(
        "--memr-beam-fusion",
        choices=["merge_then_summarize", "summarize_then_merge"],
        default="merge_then_summarize",
        help="Fuse Beam-SID memories before summarization or summarize each branch first.",
    )
    parser.add_argument(
        "--memr-reranker-model",
        default="",
        help="Optional BGE cross-encoder used to rerank MemR SID candidates.",
    )
    parser.add_argument(
        "--memr-rerank-top-k",
        type=int,
        default=0,
        help="Keep this many reranked SID memories for fusion (0 disables reranking).",
    )
    parser.add_argument(
        "--memr-reranker-batch-size", type=int, default=32,
        help="Pair-scoring batch size for the MemR cross-encoder.",
    )
    parser.add_argument(
        "--memr-reranker-max-length", type=int, default=512,
        help="Maximum cross-encoder input length for query/SID-memory pairs.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Physical GPU IDs for forked Transformers data parallelism, e.g. 0,1,2,3.",
    )

    args = parser.parse_args()
    if args.snapshot_every < 1:
        parser.error("--snapshot-every must be >= 1")
    if args.memr_sid_beams < 1:
        parser.error("--memr-sid-beams must be >= 1")
    if bool(args.memr_reranker_model) != (args.memr_rerank_top_k > 0):
        parser.error(
            "--memr-reranker-model and a positive --memr-rerank-top-k must be set together"
        )
    if args.memr_rerank_top_k > args.memr_sid_beams:
        parser.error("--memr-rerank-top-k cannot exceed --memr-sid-beams")
    if args.memr_reranker_batch_size < 1 or args.memr_reranker_max_length < 1:
        parser.error("reranker batch size and max length must be positive")
    if args.memr_reranker_model and args.inference_backend != "transformers":
        parser.error("MemR reranking currently requires --inference_backend transformers")
    if args.gpu_ids:
        try:
            parse_gpu_ids(args.gpu_ids)
        except ValueError as exc:
            parser.error(str(exc))
        if args.inference_backend != "transformers":
            parser.error("--gpu-ids requires --inference_backend transformers")
    if args.min_sp_workers < 1 or args.min_sp_workers > args.sp_workers:
        parser.error("--min-sp-workers must be between 1 and --sp_workers")
    if not 0.0 <= args.max_api_failure_rate <= 1.0:
        parser.error("--max-api-failure-rate must be between 0 and 1")
    main(args)
