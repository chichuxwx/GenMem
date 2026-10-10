"""Small fork-based data-parallel runner for single-GPU inference workers."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
from collections.abc import Callable
from typing import Any


def parse_gpu_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("GPU IDs must be a comma-separated list of integers")
    gpu_ids: list[int] = []
    for part in parts:
        try:
            gpu_id = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid GPU ID: {part!r}") from exc
        if gpu_id < 0:
            raise ValueError(f"GPU ID must be non-negative: {gpu_id}")
        # Repeated physical IDs are intentional: two independent inference
        # replicas may share one large-memory GPU when utilization is low.
        gpu_ids.append(gpu_id)
    return gpu_ids


def shard_indexed_inputs(items: list[Any], worker_count: int) -> list[list[tuple[int, Any]]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    shards: list[list[tuple[int, Any]]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(items):
        shards[index % worker_count].append((index, item))
    return shards


def _worker_entry(
    connection: Any,
    gpu_id: int,
    indexed_items: list[tuple[int, Any]],
    worker_fn: Callable[..., list[Any]],
    worker_kwargs: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    indices = [index for index, _ in indexed_items]
    items = [item for _, item in indexed_items]
    try:
        outputs = list(worker_fn(items, **worker_kwargs))
        if len(outputs) != len(items):
            raise RuntimeError(
                f"GPU {gpu_id} returned {len(outputs)} results for {len(items)} inputs"
            )
        # MemE results can be much larger than a multiprocessing pipe buffer.
        # Spilling the payload avoids a child blocking in send() while the
        # parent waits for a complete framed message from another worker.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"xwx_dp_{os.getpid()}_{gpu_id}_",
            suffix=".pkl",
            delete=False,
        ) as result_file:
            pickle.dump(list(zip(indices, outputs)), result_file, protocol=pickle.HIGHEST_PROTOCOL)
            result_path = result_file.name
        connection.send({"status": "ok", "result_path": result_path})
    except BaseException:
        connection.send({"status": "error", "error": traceback.format_exc()})
    finally:
        connection.close()


def run_data_parallel(
    items: list[Any],
    gpu_ids: list[int],
    worker_fn: Callable[..., list[Any]],
    worker_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Run one model replica per GPU and restore input ordering at the barrier."""
    if not items:
        return []
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must contain non-negative integers")
    # Duplicate IDs are allowed for controlled oversubscription.  The caller
    # should only use them after checking per-GPU memory headroom.

    shards = shard_indexed_inputs(items, min(len(gpu_ids), len(items)))
    context = mp.get_context("fork")
    jobs: list[tuple[Any, Any, int]] = []
    kwargs = dict(worker_kwargs or {})
    for gpu_id, shard in zip(gpu_ids, shards):
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_entry,
            args=(child_connection, gpu_id, shard, worker_fn, kwargs),
        )
        process.start()
        child_connection.close()
        jobs.append((process, parent_connection, gpu_id))

    merged: dict[int, Any] = {}
    errors: list[str] = []
    # Drain every control pipe concurrently. Reading jobs strictly in launch
    # order can deadlock when an earlier worker is serializing a large result
    # while a later worker is already ready to report.
    pending = list(jobs)
    started_at = {
        id(process): __import__("time").monotonic()
        for process, _, _ in pending
    }
    while pending:
        progressed = False
        next_pending: list[tuple[Any, Any, int]] = []
        for process, connection, gpu_id in pending:
            message = None
            if connection.poll(0.05):
                try:
                    message = connection.recv()
                except (EOFError, OSError):
                    message = None
                progressed = True
            elif not process.is_alive():
                progressed = True
            elif __import__("time").monotonic() - started_at[id(process)] > 3600.0:
                try:
                    process.terminate()
                except OSError:
                    pass
                progressed = True

            if message is None and process.is_alive():
                next_pending.append((process, connection, gpu_id))
                continue
            if message is None:
                message = {
                    "status": "error",
                    "error": (
                        f"GPU {gpu_id} worker exited or timed out without returning results "
                        f"(exit={process.exitcode})"
                    ),
                }
            connection.close()
            process.join(timeout=10)
            if message.get("status") != "ok" or process.exitcode != 0:
                errors.append(str(message.get("error") or f"GPU {gpu_id} exit={process.exitcode}"))
                continue
            result_path = message.get("result_path")
            try:
                with open(result_path, "rb") as result_file:
                    rows = pickle.load(result_file)
            finally:
                try:
                    os.unlink(result_path)
                except OSError:
                    pass
            for index, output in rows:
                if index in merged:
                    errors.append(f"duplicate result index: {index}")
                merged[index] = output
        pending = next_pending
        if pending and not progressed:
            __import__("time").sleep(0.05)

    if errors:
        raise RuntimeError("data-parallel worker failure:\n" + "\n".join(errors))
    expected = set(range(len(items)))
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        extra = sorted(set(merged) - expected)
        raise RuntimeError(f"incomplete data-parallel merge: missing={missing}, extra={extra}")
    return [merged[index] for index in range(len(items))]
