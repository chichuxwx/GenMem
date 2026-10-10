"""Model-replica entry points used by OfflineEvo data parallelism."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_agent_replica(infer: Any, role: str, items: list[str]) -> list[dict]:
    try:
        if role == "memr":
            return infer.run_memr_batch(items)
        if role == "meme":
            return infer.run_meme_batch(items)
        raise ValueError(f"unsupported agent role: {role}")
    finally:
        infer.unload()


def transformers_agent_worker(
    items: list[str],
    *,
    role: str,
    model_path: str,
    bank: list[dict],
    index: dict,
    max_turns: int,
    max_new_tokens: int,
    memr_sid_beams: int,
    memr_beam_fusion: str,
    memr_reranker_path: str | None = None,
    memr_rerank_top_k: int = 0,
    memr_reranker_batch_size: int = 32,
    memr_reranker_max_length: int = 512,
    infer_factory: Callable[..., Any] | None = None,
) -> list[dict]:
    if infer_factory is None:
        from mem_agent_transformers import MemAgentTransformersInfer

        infer_factory = MemAgentTransformersInfer
    infer = infer_factory(
        model_path=model_path,
        bank=bank,
        index=index,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.60,
        max_turns=max_turns,
        max_new_tokens=max_new_tokens,
        memr_sid_beams=memr_sid_beams,
        memr_beam_fusion=memr_beam_fusion,
        memr_reranker_path=memr_reranker_path,
        memr_rerank_top_k=memr_rerank_top_k,
        memr_reranker_batch_size=memr_reranker_batch_size,
        memr_reranker_max_length=memr_reranker_max_length,
    )
    return run_agent_replica(infer, role, items)
