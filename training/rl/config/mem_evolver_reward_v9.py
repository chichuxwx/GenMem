"""MemE v9 clean direct-outcome reward.

Compared with v7, direct-answer hacks and exact old-memory copies are
hard-masked to neutral zero; merely similar memories are allowed.
Any non-hacked evolved experience, whether the bank is empty or non-empty, can
be submitted to DeerFlow for the direct 2-or-0 outcome reward. Retrieval
relevance remains diagnostic only and does not gate the outcome call.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "mem_evolver_reward_v7_for_v9", _HERE / "mem_evolver_reward_v7.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load MemE v7 reward")
_v7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v7)

_REWARD_VERSION = "meme_v9_clean_direct_outcome_neutral_hack"
_v7._REWARD_VERSION = _REWARD_VERSION
_v7._HACK_PENALTY = 0.0
_v7._REQUIRE_POSITIVE_RELEVANCE_FOR_BANK_HIT = False


def compute_score(*args, **kwargs):
    return _v7.compute_score(*args, **kwargs)
