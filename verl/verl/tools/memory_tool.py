# Modified for GenMem; see verl/README.md for provenance.
"""
MemoryTool: Memory bank lookup tool for MemRetriever / MemEvolver training.

Accepts a list of SID tokens (e.g. ["<SID_L1_8><SID_L2_9><SID_L3_10>"]) and
returns the corresponding experience text from the memory bank.

Usage in tool_config yaml:
    tools:
      - class_name: verl.tools.memory_tool.MemoryTool
        config:
          memory_bank_path: ./private_data/memory_bank.json
          sid_config_path: ./private_data/sid_config.json
          top_k: 3
        tool_schema: ...
"""#增删改查

import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# SID token pattern: <SID_L1_X><SID_L2_Y><SID_L3_Z>
_SID_PATTERN = re.compile(r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)>")


def _parse_sid_tokens(sid_str: str) -> Optional[tuple[int, int, int]]:
    """Parse a SID token string into (l1, l2, l3) integer codes."""
    m = _SID_PATTERN.search(sid_str)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _build_sid_index(memory_bank: list[dict]) -> dict[tuple, list[dict]]:
    """Build a (l1, l2, l3) -> [entries] lookup index from memory bank."""
    index: dict[tuple, list[dict]] = {}
    for entry in memory_bank:
        sid_info = entry.get("sid")
        if not sid_info:
            continue
        key = (sid_info.get("s1"), sid_info.get("s2"), sid_info.get("s3"))
        if None in key:
            continue
        index.setdefault(key, []).append(entry)
    return index


def _load_memory_bank(path: str) -> tuple[list[dict], dict[tuple, list[dict]]]:
    """Load memory bank JSON and build lookup index."""
    with open(path, "r", encoding="utf-8") as f:
        bank = json.load(f)
    index = _build_sid_index(bank)
    return bank, index


class MemoryTool(BaseTool):
    """Memory bank retrieval tool for MemRetriever / MemEvolver.

    Given a list of SID strings, returns the corresponding experience texts
    from the pre-built memory bank.  Falls back to partial matching (L1 only,
    then L1+L2) when no exact match is found.

    Methods:
        get_openai_tool_schema  – OpenAI function schema
        create                  – per-trajectory instance creation
        execute                 – SID → experience lookup
        calc_reward             – unused (reward is computed externally)
        release                 – clean up instance state
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.top_k = config.get("top_k", 3)

        memory_bank_path = config.get(
            "memory_bank_path",
            "./private_data/memory_bank.json",
        )
        sid_config_path = config.get(
            "sid_config_path",
            "./private_data/sid_config.json",
        )

        self._memory_bank, self._sid_index = _load_memory_bank(memory_bank_path)

        with open(sid_config_path, "r", encoding="utf-8") as f:
            sid_cfg = json.load(f)
        self._sid_bounds = {
            "L1": sid_cfg.get("L1", 64),
            "L2": sid_cfg.get("L2", 32),
            "L3": sid_cfg.get("L3", 16),
        }

        # Build partial index: (l1,) -> entries, (l1, l2) -> entries
        self._l1_index: dict[int, list[dict]] = {}
        self._l1l2_index: dict[tuple, list[dict]] = {}
        for entry in self._memory_bank:
            sid_info = entry.get("sid")
            if not sid_info:
                continue
            s1, s2 = sid_info.get("s1"), sid_info.get("s2")
            if s1 is None:
                continue
            self._l1_index.setdefault(s1, []).append(entry)
            if s2 is not None:
                self._l1l2_index.setdefault((s1, s2), []).append(entry)

        self._instance_dict: dict[str, dict] = {}
        logger.info(
            f"MemoryTool loaded {len(self._memory_bank)} entries, "
            f"{len(self._sid_index)} unique (L1,L2,L3) keys."
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "retrieved_sids": [],
            "retrieved_texts": [],
        }
        return instance_id, ToolResponse()

    def _lookup_sid(self, sid_str: str) -> list[dict]:
        """Return up to top_k entries matching the given SID string."""
        codes = _parse_sid_tokens(sid_str)
        if codes is None:
            return []

        l1, l2, l3 = codes
        # exact match
        exact = self._sid_index.get((l1, l2, l3), [])
        if exact:
            return exact[: self.top_k]

        # L1+L2 fallback
        l1l2 = self._l1l2_index.get((l1, l2), [])
        if l1l2:
            return l1l2[: self.top_k]

        # L1-only fallback
        return self._l1_index.get(l1, [])[: self.top_k]

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Look up experience memories by SID.

        Parameters:
            sid_list: list of SID strings, e.g.
                ["<SID_L1_8><SID_L2_9><SID_L3_10>", "<SID_L1_3><SID_L2_1><SID_L3_5>"]
        """
        sid_list = parameters.get("sid_list", [])
        if isinstance(sid_list, str):
            # model sometimes passes a bare string
            sid_list = [sid_list]

        results = []
        for sid_str in sid_list:
            entries = self._lookup_sid(sid_str)
            if not entries:
                results.append(
                    f"[Memory lookup for {sid_str!r}]: No matching experience found."
                )
                continue
            for e in entries:
                text = e.get("experience_summary") or e.get("experience_text", "")
                results.append(
                    f"[SID: {sid_str}] Dataset: {e.get('dataset', '?')} | "
                    f"Experience: {text}"
                )

        response_text = "\n\n".join(results) if results else "No experience found."

        # Store for reward computation
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["retrieved_sids"].extend(sid_list)
            self._instance_dict[instance_id]["retrieved_texts"].extend(results)

        metrics = {
            "sid_count": len(sid_list),
            "hit_count": sum(
                1 for s in sid_list if self._lookup_sid(s)
            ),
        }
        return ToolResponse(text=response_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> dict:
        """Return raw retrieved data so the reward function can use it."""
        if instance_id not in self._instance_dict:
            return {"retrieved_sids": [], "retrieved_texts": []}
        return self._instance_dict[instance_id]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
