"""MemE v7 direct 2/0 outcome reward with no baseline and pre-call hard hack gates."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from pathlib import Path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_v1 = _load("mem_evolver_reward_v1_for_v7", _HERE / "mem_evolver_reward_v1.py")
_contract = _load("meme_reward_contract_v3_for_v7", _ROOT / "meme_reward_contract_v3.py")

_REWARD_VERSION = "meme_v7_direct_outcome_no_baseline_hard_hack"
_HACK_PENALTY = -0.02
_REQUIRE_POSITIVE_RELEVANCE_FOR_BANK_HIT = False
_DRAFT_SHAPING = 0.01
_MIN_DRAFT_CHARS = 32
_MAX_PROCESS_REWARD = 0.22
_MAX_TOTAL_REWARD = 2.22
_ANSWER_OPEN_RE = re.compile(r"<answer>")
_ANSWER_CLOSE_RE = re.compile(r"</answer>")


def _extract_json_object(value: str) -> dict | None:
    text = str(value or "").strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_evolution_assessment(value: str) -> dict | None:
    payload = _extract_json_object(value)
    if payload is None:
        return None
    score = payload.get("score")
    bool_fields = (
        "reusable",
        "substantive_improvement",
        "trajectory_grounded",
        "direct_answer_hack",
        "copy_hack",
    )
    reason = payload.get("reason")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    if any(not isinstance(payload.get(field), bool) for field in bool_fields):
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    return {
        "score": score,
        **{field: payload[field] for field in bool_fields},
        "reason": reason.strip(),
    }


def _get_evolution_assessment(
    query: str,
    trajectory: str,
    retrieved_experience: str,
    evolved_experience: str,
) -> tuple[dict | None, str]:
    prompt = (
        "You are a strict pairwise judge for memory evolution and reward-hacking prevention. "
        "Judge the DELTA from OLD EXPERIENCE to EVOLVED EXPERIENCE. If OLD EXPERIENCE is empty "
        "because retrieval found no match, judge whether EVOLVED EXPERIENCE is a useful new memory "
        "derived from the trajectory; do not penalize the absence of old memory. reusable=true only "
        "if the result can guide related future tasks. substantive_improvement=true only when the "
        "evolved text adds or corrects a concrete, actionable decision rule, recovery step, or "
        "verification gate supported by FAILED TRAJECTORY. copy_hack=true only when the evolved experience "
        "is exactly identical to the old experience after surrounding whitespace is stripped; do not flag "
        "merely similar guidance. direct_answer_hack=true when the text "
        "mainly gives the answer to ORIGINAL TASK rather than reusable guidance. Return only JSON with "
        "every field: {\"score\":0.0,\"reusable\":false,\"substantive_improvement\":false,"
        "\"trajectory_grounded\":false,\"direct_answer_hack\":false,\"copy_hack\":false,"
        "\"reason\":\"brief pairwise reason\"}.\n\n"
        f"ORIGINAL TASK:\n{query[:4000]}\n\nFAILED TRAJECTORY:\n{trajectory[:12000]}\n\n"
        f"OLD EXPERIENCE:\n{retrieved_experience[:16000]}\n\nEVOLVED EXPERIENCE:\n{evolved_experience[:16000]}"
    )
    raw = _v1._judge_raw(prompt, "v7 evolution and anti-hack", max_tokens=256)
    return _parse_evolution_assessment(raw or ""), raw or ""


def _decode_solution(solution_str, tokenizer, input_ids) -> str:
    solution = solution_str or ""
    if tokenizer is not None and input_ids is not None:
        try:
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            solution = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            pass
    return solution


def _parse_protocol(solution: str) -> dict:
    """Use the strict parser, recovering only a final unclosed answer tail."""
    parsed = dict(_v1._parse_protocol(solution))
    open_tags = list(_ANSWER_OPEN_RE.finditer(solution or ""))
    close_tags = list(_ANSWER_CLOSE_RE.finditer(solution or ""))
    parsed.update({
        "answer_recovered_from_truncation": False,
        "raw_num_answer_open_tags": len(open_tags),
        "raw_num_answer_close_tags": len(close_tags),
    })
    final_answer_is_open = bool(
        open_tags and (not close_tags or open_tags[-1].start() > close_tags[-1].start())
    )
    if not final_answer_is_open:
        return parsed

    recovered = (solution or "")[open_tags[-1].end() :].strip()
    parsed["evolved_experience"] = recovered
    parsed["num_answers"] = 1
    parsed["answer_recovered_from_truncation"] = bool(recovered)
    parsed["protocol_valid"] = bool(
        parsed.get("single_sid_action")
        and parsed.get("num_tool_responses") == 1
        and parsed.get("retrieved_experience")
        and recovered
    )
    return parsed


def _is_no_matching_experience(value: str) -> bool:
    return "no matching experience found" in str(value or "").casefold()


def _token_count(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
    except Exception:
        return None


def _unit_interval(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _draft_shaping(protocol_valid: bool, eligible: bool, hacked: bool, text: str) -> float:
    if protocol_valid and eligible and not hacked and len(str(text or "").strip()) >= _MIN_DRAFT_CHARS:
        return _DRAFT_SHAPING
    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
    tokenizer=None,
    input_ids=None,
    **kwargs,
) -> dict:
    extra_info = extra_info or {}
    ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
    solution = _decode_solution(solution_str, tokenizer, input_ids)
    parsed = _parse_protocol(solution)
    query = str(extra_info.get("question") or ground_truth.get("original_query") or "")
    trajectory = str(extra_info.get("trajectory") or ground_truth.get("trajectory") or "")
    target_sid = str(ground_truth.get("sid_tokens") or "")

    no_match = bool(
        parsed["protocol_valid"]
        and _is_no_matching_experience(parsed["retrieved_experience"])
    )
    retrieval_hit = bool(
        parsed["protocol_valid"]
        and not no_match
        and _v1._retrieval_hit(parsed["retrieved_experience"])
    )
    evolution_candidate = bool(parsed["protocol_valid"] and (retrieval_hit or no_match))
    old_experience = parsed["retrieved_experience"] if retrieval_hit else ""

    retrieval_score_raw = None
    retrieval_judge_raw = ""
    if retrieval_hit:
        retrieval_score_raw, retrieval_judge_raw = _v1._get_retrieval_assessment(
            trajectory, old_experience
        )

    evolution_assessment = None
    evolution_judge_raw = ""
    if evolution_candidate:
        evolution_assessment, evolution_judge_raw = _get_evolution_assessment(
            query, trajectory, old_experience, parsed["evolved_experience"]
        )
    assessed = evolution_assessment or {}

    local_direct_hack = _v1._local_direct_answer_hack(
        parsed["evolved_experience"], ground_truth
    )
    copy_ratio = (
        _contract.ngram_copy_ratio(old_experience, parsed["evolved_experience"])
        if retrieval_hit
        else 0.0
    )
    # Similar memories can still be valid when a trajectory supplies little new
    # information. Only exact reuse is a local copy hack; the judge's broader
    # near-copy label is intentionally ignored for this contract.
    local_copy_hack = bool(
        retrieval_hit
        and old_experience.strip()
        and parsed["evolved_experience"].strip() == old_experience.strip()
    )
    direct_answer_hack = bool(assessed.get("direct_answer_hack") or local_direct_hack)
    copy_hack = local_copy_hack
    hacked = direct_answer_hack or copy_hack

    sid_level_matches = _contract.independent_sid_matches(
        parsed["predicted_sid"], target_sid
    )
    sid_matched_levels = sum(sid_level_matches)
    sid_reward = 0.005 if parsed["single_sid_action"] else 0.0
    if parsed["protocol_valid"]:
        sid_reward += 0.005 + 0.0225 * sid_matched_levels

    retrieval_raw = _unit_interval(retrieval_score_raw)
    retrieval_semantically_relevant = bool(
        no_match
        or (retrieval_hit and retrieval_raw is not None and retrieval_raw > 0.0)
    )
    retrieval_relevance = 0.0
    if retrieval_hit:
        retrieval_relevance = 0.01 + (0.05 * retrieval_raw if retrieval_raw is not None else 0.0)

    evolution_raw = _unit_interval(assessed.get("score"))
    evolution_gate = bool(
        evolution_candidate
        and (no_match or retrieval_raw is not None)
        and evolution_raw is not None
        and assessed.get("reusable") is True
        and assessed.get("substantive_improvement") is True
        and assessed.get("trajectory_grounded") is True
        and not hacked
    )
    evolution_quality = 0.05 * evolution_raw if evolution_gate else 0.0
    draft_shaping_reward = _draft_shaping(
        parsed["protocol_valid"], evolution_candidate, hacked, parsed["evolved_experience"]
    )

    # The semantic anti-hack assessment and deterministic checks are complete
    # before this gate. Hacked or unassessed answers never reach DeerFlow.
    anti_hack_assessment_available = evolution_assessment is not None
    outcome_retrieval_gate = bool(
        evolution_candidate
        and (
            not _REQUIRE_POSITIVE_RELEVANCE_FOR_BANK_HIT
            or retrieval_semantically_relevant
        )
    )
    outcome_gate = bool(
        outcome_retrieval_gate and anti_hack_assessment_available and not hacked
    )
    with_memory_em = 0.0
    with_memory_answer = ""
    with_memory_trace = ""
    with_memory_ok = False
    if outcome_gate:
        try:
            with_memory_em, with_memory_answer, with_memory_trace, with_memory_ok = (
                _v1._backend._get_outcome_score_via_deerflow(
                    parsed["evolved_experience"],
                    query,
                    ground_truth,
                    data_source=data_source,
                )
            )
        except Exception as exc:
            print(f"[MemE v7 outcome] unavailable: {exc}")

    # This experiment was prefiltered to baseline-false tasks. There is no
    # baseline lookup, cache read, or counterfactual subtraction in v7.
    outcome_reward = 2.0 if with_memory_ok and float(with_memory_em) > 0.0 else 0.0

    sid_reward = round(sid_reward, 6)
    retrieval_relevance = round(retrieval_relevance, 6)
    evolution_quality = round(evolution_quality, 6)
    process_scores = []
    if parsed["single_sid_action"]:
        tool_start, tool_end = _v1._span(
            tokenizer, input_ids, "<tool_call>", "</tool_call>", kwargs.get("total_len")
        )
        process_scores.append({
            "from_idx": tool_start,
            "to_idx": tool_end,
            "score_value": sid_reward,
            "token_type": "meme_sid_action_v7",
        })
    if parsed["protocol_valid"]:
        tool_start, tool_end = _v1._span(
            tokenizer, input_ids, "<tool_call>", "</tool_call>", kwargs.get("total_len")
        )
        answer_start, answer_end = _v1._span(
            tokenizer, input_ids, "<answer>", "</answer>", kwargs.get("total_len")
        )
        process_scores.extend([
            {
                "from_idx": tool_start,
                "to_idx": tool_end,
                "score_value": retrieval_relevance,
                "token_type": "meme_retrieval_relevance_v7",
            },
            {
                "from_idx": answer_start,
                "to_idx": answer_end,
                "score_value": evolution_quality,
                "token_type": "meme_evolution_quality_v7",
            },
        ])
    if draft_shaping_reward:
        process_scores.append({
            "from_idx": 0,
            "to_idx": 0,
            "score_value": draft_shaping_reward,
            "token_type": "meme_evolution_draft_shape_v7",
        })

    hack_penalty = 0.0
    hack_mask_applied = False
    if hacked:
        sid_reward = retrieval_relevance = evolution_quality = draft_shaping_reward = 0.0
        for item in process_scores:
            item["score_value"] = 0.0
        hack_penalty = _HACK_PENALTY
        hack_mask_applied = True
        outcome_gate = False
        outcome_reward = 0.0
        process_scores.append({
            "from_idx": 0,
            "to_idx": 0,
            "score_value": _HACK_PENALTY,
            "token_type": "meme_hack_penalty_v7",
        })

    process_total = round(
        sid_reward + retrieval_relevance + evolution_quality + draft_shaping_reward + hack_penalty,
        6,
    )
    total = round(process_total + outcome_reward, 6)
    if process_total > _MAX_PROCESS_REWARD + 1e-9 or total > _MAX_TOTAL_REWARD + 1e-9:
        raise AssertionError(f"MemE v7 reward cap violated: process={process_total}, total={total}")

    evolved_answer_tokens = _token_count(tokenizer, parsed["evolved_experience"])
    retrieved_experience_tokens = _token_count(tokenizer, parsed["retrieved_experience"])
    try:
        answer_token_cap = int(os.getenv("MEME_ANSWER_TOKEN_CAP", "0"))
    except ValueError:
        answer_token_cap = 0
    trace_payload = _v1._trace_for_rollout(with_memory_trace)
    breakdown = {
        **parsed,
        "reward_version": _REWARD_VERSION,
        "target_sid": target_sid,
        "retrieval_hit": retrieval_hit,
        "new_experience": no_match,
        "old_experience_used": old_experience,
        "sid_prefix_len": _contract.longest_correct_sid_prefix(parsed["predicted_sid"], target_sid),
        "sid_level_matches": list(sid_level_matches),
        "sid_matched_levels": sid_matched_levels,
        "sid_reward": sid_reward,
        "retrieval_relevance": retrieval_relevance,
        "retrieval_score_raw": retrieval_score_raw,
        "retrieval_semantically_relevant": retrieval_semantically_relevant,
        "require_positive_relevance_for_bank_hit": (
            _REQUIRE_POSITIVE_RELEVANCE_FOR_BANK_HIT
        ),
        "outcome_retrieval_gate": outcome_retrieval_gate,
        "retrieval_judge_raw": retrieval_judge_raw[:4000],
        "evolution_gate": evolution_gate and not hacked,
        "evolution_quality": evolution_quality,
        "evolution_score_raw": assessed.get("score"),
        "evolution_reusable": assessed.get("reusable") is True,
        "substantive_improvement": assessed.get("substantive_improvement") is True,
        "trajectory_grounded": assessed.get("trajectory_grounded") is True,
        "direct_answer_hack": direct_answer_hack,
        "copy_hack": copy_hack,
        "copy_ngram_ratio": copy_ratio,
        "copy_hack_threshold": _contract.COPY_HACK_THRESHOLD,
        "anti_hack_assessment_available": anti_hack_assessment_available,
        "hack_penalty": hack_penalty,
        "hack_mask_applied": hack_mask_applied,
        "draft_shaping_reward": draft_shaping_reward,
        "evolution_reason": assessed.get("reason", "judge unavailable or invalid response"),
        "evolution_judge_raw": evolution_judge_raw[:4000],
        "outcome_gate": outcome_gate,
        "outcome_reward_gate": outcome_gate,
        "outcome_available": bool(with_memory_ok),
        "with_memory_em_raw": float(with_memory_em),
        "with_memory_ok": bool(with_memory_ok),
        "outcome_score": outcome_reward,
        "baseline_mode": "disabled_direct_2_or_0",
        "baseline_em_raw": None,
        "baseline_ok": False,
        "outcome_delta_em": None,
        "process_reward_total": process_total,
        "total_scalar_reward": total,
        "evolved_answer_tokens": evolved_answer_tokens,
        "retrieved_experience_tokens": retrieved_experience_tokens,
        "answer_token_cap": answer_token_cap,
        "answer_cap_hit": bool(
            answer_token_cap > 0
            and evolved_answer_tokens is not None
            and evolved_answer_tokens >= answer_token_cap
        ),
        "observation_truncated": "...(truncated)..." in parsed["retrieved_experience"],
        "sid_reward_range": [0.0, 0.1],
        "retrieval_relevance_range": [0.0, 0.06],
        "evolution_quality_range": [0.0, 0.05],
        "process_reward_cap": _MAX_PROCESS_REWARD,
        "total_reward_cap": _MAX_TOTAL_REWARD,
        "outcome_backend": "deerflow_direct_2_or_0_no_baseline_meme_v7",
        "judge_backend": {
            "base_url": os.getenv("SUMMARY_BASE_URL", ""),
            "model": os.getenv("SUMMARY_MODEL", ""),
        },
        "outcome_model_backend": {
            "deerflow_url": os.getenv("DEERFLOW_BASE_URL", ""),
            "model_api": os.getenv("DEERFLOW_MODEL_API", ""),
            "model": os.getenv("DEERFLOW_MODEL", ""),
        },
        "deerflow_with_memory": {
            "answer": with_memory_answer,
            "em": float(with_memory_em),
            "ok": bool(with_memory_ok),
            "trace": trace_payload,
        },
        "deerflow_baseline": {"disabled": True, "ok": False},
    }
    print(
        f"[MemE reward {_REWARD_VERSION}] protocol={parsed['protocol_valid']} new={no_match} "
        f"hack={hacked} deerflow={outcome_gate} process={process_total:.3f} "
        f"outcome={outcome_reward:.1f}"
    )
    trace_envelope = json.dumps(
        {"trace_schema": "deerflow_direct_2_or_0_no_baseline_meme_v7", "with_memory": trace_payload},
        ensure_ascii=False,
    )
    return {
        "outcome_score": outcome_reward,
        "process_scores": process_scores,
        "model_answer": parsed["evolved_experience"],
        "evolved_experience": parsed["evolved_experience"],
        "retrieved_experience": parsed["retrieved_experience"],
        "sp_answer": with_memory_answer,
        "sp_trajectory": trace_envelope,
        "sp_raw_result": with_memory_trace,
        "deerflow_answer": with_memory_answer,
        "deerflow_result": with_memory_trace,
        "reward_breakdown": json.dumps(breakdown, ensure_ascii=False),
    }
