"""Client helpers for a remote MemR replica pool."""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from data_parallel import run_data_parallel


def _post(url: str, questions: list[str], timeout: int = 3600) -> list[dict]:
    request = urllib.request.Request(
        url.rstrip("/") + "/infer",
        data=json.dumps({"questions": questions}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(questions):
        raise RuntimeError(f"remote MemR returned an invalid result count from {url}")
    return results


def run_remote_memr_batch(questions: list[str], urls: list[str]) -> list[dict]:
    """Shard a batch across remote workers and restore the original order."""
    if not questions:
        return []
    if not urls:
        raise ValueError("at least one remote MemR URL is required")
    shards: list[list[tuple[int, str]]] = [[] for _ in urls]
    for index, question in enumerate(questions):
        shards[index % len(urls)].append((index, question))
    jobs = []
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        for url, shard in zip(urls, shards):
            if shard:
                jobs.append((url, shard, executor.submit(_post, url, [q for _, q in shard])))
        merged: dict[int, dict] = {}
        for url, shard, future in jobs:
            rows = future.result()
            for (index, _), row in zip(shard, rows):
                merged[index] = row
    if len(merged) != len(questions):
        raise RuntimeError("remote MemR batch lost results")
    return [merged[index] for index in range(len(questions))]


def run_hybrid_memr_batch(
    questions: list[str],
    *,
    local_gpu_ids: list[int],
    remote_urls: list[str],
    local_worker_fn,
    local_worker_kwargs: dict,
) -> list[dict]:
    """Run local GPU replicas and remote replicas concurrently."""
    if not questions:
        return []
    if not local_gpu_ids and not remote_urls:
        raise ValueError("hybrid MemR requires a local GPU or remote URL")
    worker_count = len(local_gpu_ids) + len(remote_urls)
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(worker_count)]
    for index, question in enumerate(questions):
        buckets[index % worker_count].append((index, question))
    local_buckets = buckets[: len(local_gpu_ids)]
    remote_buckets = buckets[len(local_gpu_ids) :]

    def local_job() -> dict[int, dict]:
        rows: dict[int, dict] = {}
        nonempty = [(gpu, shard) for gpu, shard in zip(local_gpu_ids, local_buckets) if shard]
        if not nonempty:
            return rows
        # run_data_parallel expects one shard per GPU and restores that
        # subset's order; empty tail GPUs are omitted for small batches.
        gpu_ids = [gpu for gpu, _ in nonempty]
        local_items = [item for _, shard in nonempty for _, item in shard]
        # Reconstruct the same round-robin ordering expected by the helper.
        local_rows = run_data_parallel(
            local_items,
            gpu_ids=gpu_ids,
            worker_fn=local_worker_fn,
            worker_kwargs=local_worker_kwargs,
        )
        cursor = 0
        for _, shard in nonempty:
            for index, _ in shard:
                rows[index] = local_rows[cursor]
                cursor += 1
        return rows

    def remote_job() -> dict[int, dict]:
        rows: dict[int, dict] = {}
        jobs = []
        with ThreadPoolExecutor(max_workers=max(1, len(remote_urls))) as executor:
            for url, shard in zip(remote_urls, remote_buckets):
                if shard:
                    jobs.append((shard, executor.submit(_post, url, [q for _, q in shard])))
            for shard, future in jobs:
                for (index, _), row in zip(shard, future.result()):
                    rows[index] = row
        return rows

    merged: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(local_job) if local_gpu_ids else None
        remote_future = executor.submit(remote_job) if remote_urls else None
        if local_future:
            merged.update(local_future.result())
        if remote_future:
            merged.update(remote_future.result())
    if len(merged) != len(questions):
        raise RuntimeError("hybrid MemR batch lost results")
    return [merged[index] for index in range(len(questions))]
