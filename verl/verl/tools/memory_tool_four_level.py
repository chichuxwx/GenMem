# Modified for GenMem; see verl/README.md for provenance.
"""Exact four-layer Semantic ID lookup for MemRetriever v33.

Each complete ``(L1, L2, L3, L4)`` code identifies one memory.  This tool
deliberately performs no prefix fallback, candidate expansion, or reranking.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SID_PATTERN = re.compile(
    r"^<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>$"
)


def _parse_sid_tokens(
    sid: str, bounds: tuple[int, int, int, int]
) -> Optional[tuple[int, int, int, int]]:
    match = _SID_PATTERN.fullmatch(str(sid or "").strip())
    if match is None:
        return None
    codes = tuple(int(match.group(index)) for index in range(1, 5))
    if any(code < 0 or code >= bound for code, bound in zip(codes, bounds)):
        return None
    return codes


def _canonical_sid(codes: tuple[int, int, int, int]) -> str:
    return "".join(
        f"<SID_L{level}_{code}>" for level, code in enumerate(codes, start=1)
    )


def _load_exact_index(
    memory_bank_path: Path, bounds: tuple[int, int, int, int]
) -> dict[tuple[int, int, int, int], dict[str, str]]:
    with memory_bank_path.open(encoding="utf-8") as handle:
        bank = json.load(handle)
    if not isinstance(bank, list):
        raise ValueError("memory bank root must be a JSON array")

    index: dict[tuple[int, int, int, int], dict[str, str]] = {}
    for row_number, entry in enumerate(bank):
        sid_info = entry.get("sid") or {}
        sid_tokens = sid_info.get("tokens", "") if isinstance(sid_info, dict) else ""
        codes = _parse_sid_tokens(sid_tokens, bounds)
        if codes is None and isinstance(sid_info, dict):
            raw_codes = sid_info.get("codes")
            if isinstance(raw_codes, list) and len(raw_codes) == 4:
                candidate = tuple(int(code) for code in raw_codes)
                if all(0 <= code < bound for code, bound in zip(candidate, bounds)):
                    codes = candidate
        if codes is None:
            raise ValueError(f"invalid four-layer SID at memory-bank row {row_number}")
        if codes in index:
            raise ValueError(
                f"duplicate four-layer SID {_canonical_sid(codes)} in memory bank"
            )
        experience = str(
            entry.get("experience_summary") or entry.get("experience_text") or ""
        ).strip()
        if not experience:
            raise ValueError(f"empty experience at memory-bank row {row_number}")
        index[codes] = {
            "dataset": str(entry.get("dataset") or "unknown"),
            "experience": experience,
            "sample_id": str(entry.get("sample_id") or ""),
        }
    return index


class MemoryToolFourLevel(BaseTool):
    """Map exactly one complete four-layer SID to exactly one memory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        if int(config.get("top_k", 1)) != 1:
            raise ValueError("MemoryToolFourLevel requires top_k=1")

        sid_config_path = Path(config["sid_config_path"])
        with sid_config_path.open(encoding="utf-8") as handle:
            sid_config = json.load(handle)
        self._bounds = tuple(int(sid_config[f"L{level}"]) for level in range(1, 5))
        if len(self._bounds) != 4:
            raise ValueError("four SID bounds are required")

        memory_bank_path = Path(config["memory_bank_path"])
        self._sid_index = _load_exact_index(memory_bank_path, self._bounds)
        self._instance_dict: dict[str, dict[str, list[str]]] = {}
        logger.info("Loaded %d unique four-layer memories", len(self._sid_index))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {
            "retrieved_sids": [],
            "retrieved_texts": [],
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        sid_list = parameters.get("sid_list")
        if not isinstance(sid_list, list) or len(sid_list) != 1:
            return ToolResponse(text="Invalid four-layer SID."), 0.0, {"hit_count": 0}

        raw_sid = sid_list[0]
        if not isinstance(raw_sid, str):
            return ToolResponse(text="Invalid four-layer SID."), 0.0, {"hit_count": 0}
        codes = _parse_sid_tokens(raw_sid, self._bounds)
        if codes is None:
            return ToolResponse(text="Invalid four-layer SID."), 0.0, {"hit_count": 0}

        entry = self._sid_index.get(codes)
        if entry is None:
            return ToolResponse(text="No matching experience found."), 0.0, {"hit_count": 0}

        sid = _canonical_sid(codes)
        response_text = (
            f"[SID: {sid}] Dataset: {entry['dataset']} | "
            f"Experience: {entry['experience']}"
        )
        state = self._instance_dict.get(instance_id)
        if state is not None:
            state["retrieved_sids"].append(sid)
            state["retrieved_texts"].append(response_text)
        return (
            ToolResponse(text=response_text),
            0.0,
            {"hit_count": 1, "sample_id": entry["sample_id"]},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> dict:
        return self._instance_dict.get(
            instance_id, {"retrieved_sids": [], "retrieved_texts": []}
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
