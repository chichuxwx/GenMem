"""BGE cross-encoder reranking for SID candidates and current memories."""

from __future__ import annotations

from collections.abc import Sequence


def build_candidate_passage(sid: str, memory: str) -> str:
    """Build the same semantic passage used by the standalone MT evaluator."""
    return f"SID: {sid}\nMemory: {memory}"


def rank_scored_candidates(
    candidates: Sequence[tuple[str, str]],
    scores: Sequence[float],
    *,
    top_k: int,
) -> list[dict]:
    """Return a stable descending ranking, limited to ``top_k`` rows."""
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    return [
        {
            "sid": candidates[index][0],
            "memory": candidates[index][1],
            "score": float(scores[index]),
        }
        for index in order[:top_k]
    ]


class BgeSidMemoryReranker:
    """Load one BGE reranker per inference worker and rank current-bank memories."""

    def __init__(
        self,
        model_path: str,
        device,
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

    def rank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int,
    ) -> list[dict]:
        import torch

        passages = [build_candidate_passage(sid, memory) for sid, memory in candidates]
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits.float()
            batch_scores = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
            scores.extend(batch_scores.cpu().tolist())
        return rank_scored_candidates(candidates, scores, top_k=top_k)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
