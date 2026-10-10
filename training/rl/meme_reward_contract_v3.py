"""Pure, replayable MemE v3 reward contract with substantive-delta guards."""

from __future__ import annotations

import math
import re
from typing import Mapping


REWARD_VERSION = "meme_v3_orthogonal_sid_substantive_delta_guarded"
MAX_PROCESS_REWARD = 0.21
MAX_TOTAL_REWARD = 2.21
COPY_HACK_THRESHOLD = 0.80

_SID_RE = re.compile(r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>")
_SID_BOUNDS = (48, 16, 8, 8)
_SID_LEVEL_REWARD = 0.0225
_WORD_RE = re.compile(r"[a-z0-9]+")


def _sid_codes(value: object) -> tuple[int, int, int, int] | None:
    match = _SID_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    codes = tuple(int(match.group(index)) for index in range(1, 5))
    if any(code < 0 or code >= bound for code, bound in zip(codes, _SID_BOUNDS)):
        return None
    return codes


def longest_correct_sid_prefix(predicted: object, target: object) -> int:
    predicted_codes, target_codes = _sid_codes(predicted), _sid_codes(target)
    if predicted_codes is None or target_codes is None:
        return 0
    length = 0
    for predicted_code, target_code in zip(predicted_codes, target_codes):
        if predicted_code != target_code:
            break
        length += 1
    return length


def independent_sid_matches(predicted: object, target: object) -> tuple[bool, bool, bool, bool]:
    """Return one match bit per SID level without imposing prefix continuity."""
    predicted_codes, target_codes = _sid_codes(predicted), _sid_codes(target)
    if predicted_codes is None or target_codes is None:
        return False, False, False, False
    return tuple(
        predicted_code == target_code
        for predicted_code, target_code in zip(predicted_codes, target_codes)
    )


def ngram_copy_ratio(old_experience: object, evolved_experience: object, n: int = 5) -> float:
    """Return the fraction of evolved n-grams copied verbatim from old memory."""
    old = _WORD_RE.findall(str(old_experience or "").casefold())
    evolved = _WORD_RE.findall(str(evolved_experience or "").casefold())
    if not evolved:
        return 0.0
    if len(evolved) < n:
        return 1.0 if evolved == old else 0.0
    old_ngrams = {tuple(old[index : index + n]) for index in range(max(0, len(old) - n + 1))}
    evolved_ngrams = [
        tuple(evolved[index : index + n])
        for index in range(len(evolved) - n + 1)
    ]
    if not evolved_ngrams:
        return 0.0
    return round(sum(item in old_ngrams for item in evolved_ngrams) / len(evolved_ngrams), 6)


def _unit_interval(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def score_evidence(evidence: Mapping[str, object]) -> dict[str, object]:
    """Score observed evidence; all semantic and anti-hack gates fail closed."""
    single_action = bool(evidence.get("single_sid_action"))
    protocol = bool(evidence.get("protocol_valid"))
    retrieval_hit = bool(evidence.get("retrieval_hit"))
    reusable = evidence.get("evolution_reusable") is True
    substantive = evidence.get("substantive_improvement") is True
    grounded = evidence.get("trajectory_grounded") is True
    direct_hack = bool(evidence.get("direct_answer_hack"))
    copy_hack = bool(evidence.get("copy_hack"))
    any_hack = direct_hack or copy_hack
    prefix_len = longest_correct_sid_prefix(
        evidence.get("predicted_sid"), evidence.get("target_sid")
    )
    sid_level_matches = independent_sid_matches(
        evidence.get("predicted_sid"), evidence.get("target_sid")
    )
    sid_matched_levels = sum(sid_level_matches)
    retrieval_raw = _unit_interval(evidence.get("retrieval_score_raw"))
    evolution_raw = _unit_interval(evidence.get("evolution_score_raw"))

    sid_reward = 0.005 if single_action else 0.0
    if protocol:
        sid_reward += 0.005 + _SID_LEVEL_REWARD * sid_matched_levels

    retrieval_reward = 0.0
    if protocol and retrieval_hit:
        retrieval_reward = 0.01 + (0.05 * retrieval_raw if retrieval_raw is not None else 0.0)

    evolution_gate = bool(
        protocol
        and retrieval_hit
        and retrieval_raw is not None
        and evolution_raw is not None
        and reusable
        and substantive
        and grounded
        and not any_hack
    )
    evolution_reward = 0.05 * evolution_raw if evolution_gate else 0.0

    # Outcome is an independent counterfactual signal.  It must not depend on
    # the evolution judge: an unavailable/strict evolution assessment should
    # not prevent us from checking whether the evolved memory improves the
    # downstream answer.  Safety still requires a valid protocol, a retrieval
    # hit, and no direct-answer/copy hack before making the external call.
    outcome_gate = bool(protocol and retrieval_hit)
    outcome_reward_gate = bool(outcome_gate and not any_hack)
    outcome_reward = 2.0 if (
        outcome_reward_gate
        and bool(evidence.get("outcome_available"))
        and float(evidence.get("with_memory_em_raw") or 0.0) > 0.0
    ) else 0.0

    if any_hack:
        # SID is an orthogonal action-learning signal. Preserve its bounded
        # credit; hacks invalidate downstream retrieval/evolution/outcome only.
        retrieval_reward = evolution_reward = outcome_reward = 0.0
        evolution_gate = False

    # Small bounded negative signal discourages copying/direct-answer hacks
    # without erasing the orthogonal SID action credit.
    hack_penalty = -0.02 if any_hack else 0.0

    sid_reward = round(sid_reward, 6)
    retrieval_reward = round(retrieval_reward, 6)
    evolution_reward = round(evolution_reward, 6)
    process_total = round(sid_reward + retrieval_reward + evolution_reward + hack_penalty, 6)
    total = round(process_total + outcome_reward, 6)
    if process_total > MAX_PROCESS_REWARD + 1e-9 or total > MAX_TOTAL_REWARD + 1e-9:
        raise AssertionError(
            f"MemE v3 reward cap violated: process={process_total}, total={total}"
        )
    return {
        "reward_version": REWARD_VERSION,
        "sid_prefix_len": prefix_len,
        "sid_level_matches": list(sid_level_matches),
        "sid_matched_levels": sid_matched_levels,
        "sid_reward": sid_reward,
        "retrieval_relevance": retrieval_reward,
        "evolution_gate": evolution_gate,
        "evolution_quality": evolution_reward,
        "outcome_gate": outcome_gate,
        "outcome_reward_gate": outcome_reward_gate,
        "outcome_score": outcome_reward,
        "hack_penalty": round(hack_penalty, 6),
        "process_reward_total": process_total,
        "total_scalar_reward": total,
    }
