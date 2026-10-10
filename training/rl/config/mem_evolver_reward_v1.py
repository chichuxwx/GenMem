"""Nonnegative outcome-dominant reward for true MemE memory evolution.

The policy must perform exactly one four-layer memory lookup and emit exactly
one reusable evolved experience.  Low-weight process terms total at most 0.3;
the direct with-memory DeerFlow outcome is 2.0 or zero.  The outcome call is
gated by an anti-hacking judge so a bare task answer cannot earn outcome.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional


_SID_RE = re.compile(r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>")
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SID_BOUNDS = (48, 16, 8, 8)
_JUDGE_MAX_ATTEMPTS = 3
_JUDGE_ENDPOINT = "http://127.0.0.1:8010/v1"
_JUDGE_MODEL = "Qwen3-32B"
_SID_BASE_REWARD = 0.05
_SID_LEVEL_REWARD = 0.0125
_QUALITY_SCALE = 0.1


def _load_backend():
    configured = os.getenv("DEERFLOW_HELPERS_PATH", "").strip()
    path = Path(configured) if configured else Path(__file__).with_name("deerflow_outcome_helpers_v33.py")
    if not path.is_file():
        class MissingBackend:
            pass
        return MissingBackend()
    spec = importlib.util.spec_from_file_location("meme_deerflow_outcome_helpers_v33", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DeerFlow helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backend = _load_backend()


def _parse_sid(value: str) -> tuple[int, int, int, int] | None:
    match = _SID_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    codes = tuple(int(match.group(index)) for index in range(1, 5))
    if any(code < 0 or code >= bound for code, bound in zip(codes, _SID_BOUNDS)):
        return None
    return codes


def _sid_reward(predicted_sid: str, target_sid: str) -> float:
    predicted, target = _parse_sid(predicted_sid), _parse_sid(target_sid)
    if predicted is None or target is None:
        return 0.0
    return round(min(0.1, _SID_BASE_REWARD + _SID_LEVEL_REWARD * sum(a == b for a, b in zip(predicted, target))), 4)


def _retrieval_hit(value: str) -> bool:
    text = str(value or "").strip()
    return len(text) >= 50 and not text.lower().startswith("no matching experience found")


def _parse_protocol(solution: str) -> dict:
    tool_calls = _TOOL_CALL_RE.findall(solution or "")
    responses = [item.strip() for item in _TOOL_RESPONSE_RE.findall(solution or "")]
    answers = [item.strip() for item in _ANSWER_RE.findall(solution or "")]
    action_sids = [match.group(0) for body in tool_calls for match in _SID_RE.finditer(body)]
    predicted_sid = ""
    call_ok = False
    if len(tool_calls) == 1:
        try:
            payload = json.loads(tool_calls[0].strip())
            arguments = payload.get("arguments") or {}
            sid_list = arguments.get("sid_list")
            if (
                payload.get("name") == "memory_lookup"
                and isinstance(sid_list, list)
                and len(sid_list) == 1
                and isinstance(sid_list[0], str)
                and _parse_sid(sid_list[0]) is not None
            ):
                predicted_sid = sid_list[0]
                call_ok = True
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    protocol_valid = (
        call_ok
        and len(action_sids) == 1
        and len(responses) == 1
        and bool(responses[0])
        and len(answers) == 1
        and bool(answers[0])
    )
    return {
        "protocol_valid": protocol_valid,
        "single_sid_action": call_ok and len(tool_calls) == 1 and len(action_sids) == 1,
        "num_tool_calls": len(tool_calls),
        "num_tool_responses": len(responses),
        "num_answers": len(answers),
        "total_sids": len(action_sids),
        "predicted_sid": predicted_sid,
        "retrieved_experience": responses[0] if len(responses) == 1 else "",
        "evolved_experience": answers[0] if len(answers) == 1 else "",
    }


def _parse_unit_interval_score(value: object) -> Optional[float]:
    text = str(value or "").strip()
    match = re.search(r'"score"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', text)
    if not match:
        return None
    score = float(match.group(1))
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


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
    reusable = payload.get("reusable")
    direct = payload.get("direct_answer_hack")
    reason = payload.get("reason")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    if not isinstance(reusable, bool) or not isinstance(direct, bool) or not isinstance(reason, str):
        return None
    return {"score": score, "reusable": reusable, "direct_answer_hack": direct, "reason": reason.strip()}


def _judge_raw(prompt: str, label: str, max_tokens: int = 192) -> str | None:
    for attempt in range(1, _JUDGE_MAX_ATTEMPTS + 1):
        try:
            from openai import OpenAI

            endpoint = os.getenv("SUMMARY_BASE_URL", os.getenv("EVAL_LLM_BASE_URL", _JUDGE_ENDPOINT))
            model = os.getenv("SUMMARY_MODEL", os.getenv("EVAL_LLM_MODEL_NAME", _JUDGE_MODEL))
            api_key = os.getenv("SUMMARY_API_KEY", os.getenv("EVAL_LLM_API_KEY", "EMPTY"))
            client = OpenAI(api_key=api_key, base_url=endpoint, timeout=30, max_retries=0)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            raw = message.content or getattr(message, "reasoning_content", None) or ""
            if str(raw).strip():
                return str(raw).strip()
            raise ValueError("empty judge response")
        except Exception as exc:
            print(f"[MemE {label}] attempt {attempt}/{_JUDGE_MAX_ATTEMPTS} failed: {exc}")
            if attempt < _JUDGE_MAX_ATTEMPTS:
                time.sleep(1)
    return None


def _get_retrieval_assessment(trajectory: str, retrieved_experience: str) -> tuple[Optional[float], str]:
    prompt = (
        "You judge whether an existing memory is relevant to improving the failed trajectory. "
        "Score 0.0 for unrelated or unusable content, 0.2-0.5 for broadly related strategy, and 0.6-1.0 for directly applicable knowledge. "
        "Do not reward shared labels or verbosity. Return only JSON: {\"score\": 0.0}.\n\n"
        f"FAILED TRAJECTORY:\n{trajectory[:12000]}\n\nOLD EXPERIENCE:\n{retrieved_experience[:16000]}"
    )
    raw = _judge_raw(prompt, "retrieval relevance")
    return (_parse_unit_interval_score(raw), raw or "")


def _get_evolution_assessment(
    query: str,
    trajectory: str,
    retrieved_experience: str,
    evolved_experience: str,
) -> tuple[dict | None, str]:
    prompt = (
        "You are a strict anti-reward-hacking judge for memory evolution. Decide whether EVOLVED EXPERIENCE genuinely synthesizes useful old guidance with lessons from the failed trajectory into transferable, actionable knowledge. "
        "Set reusable=false for vague templates, copied text, unsupported claims, or content that cannot guide a similar future task. Set direct_answer_hack=true when it mainly gives the answer to ORIGINAL TASK instead of reusable methods, decision rules, failure recovery, or verification checks. "
        "Score quality from 0.0 to 1.0. Return only JSON with all fields: {\"score\":0.0,\"reusable\":false,\"direct_answer_hack\":false,\"reason\":\"brief reason\"}.\n\n"
        f"ORIGINAL TASK:\n{query[:4000]}\n\nFAILED TRAJECTORY:\n{trajectory[:12000]}\n\n"
        f"OLD EXPERIENCE:\n{retrieved_experience[:16000]}\n\nEVOLVED EXPERIENCE:\n{evolved_experience[:16000]}"
    )
    raw = _judge_raw(prompt, "evolution quality")
    return (_parse_evolution_assessment(raw or ""), raw or "")


def _normalize_direct_answer(text: str) -> str:
    value = re.sub(r"[*_`#]", "", str(text or "")).strip().casefold()
    value = re.sub(r"^(?:the\s+answer\s+is|answer)\s*[:：-]?\s*", "", value)
    return value.strip(" .,!?:;\"'()[]{}")


def _local_direct_answer_hack(evolved: str, ground_truth: dict) -> bool:
    normalized = _normalize_direct_answer(evolved)
    aliases = list(ground_truth.get("target_answer_aliases") or [])
    answer = ground_truth.get("target_answer")
    if answer:
        aliases.append(answer)
    normalized_aliases = {_normalize_direct_answer(alias) for alias in aliases if str(alias).strip()}
    return bool(normalized and normalized in normalized_aliases)


def _span(tokenizer, input_ids, start_tag: str, end_tag: str, total_len: int | None) -> tuple[int, int]:
    if tokenizer is not None and input_ids is not None:
        try:
            from verl.spr1_special_tokens.recognition import TokenRecognizer

            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            intervals = TokenRecognizer(tokenizer).find_intervals(ids, start_tag, end_tag, include_boundaries=True)
            if len(intervals) == 1:
                start, end = intervals[0]
                return int(start), max(int(start), int(end) - 1)
        except Exception:
            pass
    eos = max(int(total_len or 1) - 1, 0)
    return eos, eos


def _trace_for_rollout(trace):
    converter = getattr(_backend, "_trace_for_rollout", None)
    if converter is not None:
        return converter(trace)
    try:
        return json.loads(trace) if trace else None
    except (json.JSONDecodeError, TypeError):
        return {"trace_schema": "unparsed_text_v1", "raw": str(trace)}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    tokenizer=None,
    input_ids=None,
    **kwargs,
) -> dict:
    extra_info = extra_info or {}
    ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
    solution = solution_str or ""
    if tokenizer is not None and input_ids is not None:
        try:
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            solution = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            pass

    parsed = _parse_protocol(solution)
    query = str(extra_info.get("question") or ground_truth.get("original_query") or "")
    trajectory = str(extra_info.get("trajectory") or ground_truth.get("trajectory") or "")
    target_sid = str(ground_truth.get("sid_tokens") or "")
    process_scores: list[dict] = []
    sid_reward = retrieval_relevance = evolution_quality = 0.0
    retrieval_hit = bool(parsed["protocol_valid"] and _retrieval_hit(parsed["retrieved_experience"]))
    retrieval_score_raw = None
    evolution_assessment = None
    retrieval_judge_raw = evolution_judge_raw = ""
    direct_answer_hack = False
    outcome_gate = False

    if parsed["protocol_valid"]:
        tool_start, tool_end = _span(tokenizer, input_ids, "<tool_call>", "</tool_call>", kwargs.get("total_len"))
        answer_start, answer_end = _span(tokenizer, input_ids, "<answer>", "</answer>", kwargs.get("total_len"))
        if retrieval_hit:
            sid_reward = _sid_reward(parsed["predicted_sid"], target_sid)
            retrieval_score_raw, retrieval_judge_raw = _get_retrieval_assessment(
                trajectory, parsed["retrieved_experience"]
            )
            if retrieval_score_raw is not None:
                retrieval_relevance = round(_QUALITY_SCALE * retrieval_score_raw, 4)
        evolution_assessment, evolution_judge_raw = _get_evolution_assessment(
            query, trajectory, parsed["retrieved_experience"], parsed["evolved_experience"]
        )
        local_hack = _local_direct_answer_hack(parsed["evolved_experience"], ground_truth)
        if evolution_assessment is not None:
            direct_answer_hack = bool(evolution_assessment["direct_answer_hack"] or local_hack)
            outcome_gate = bool(
                retrieval_hit
                and evolution_assessment["reusable"]
                and not direct_answer_hack
            )
            if outcome_gate:
                evolution_quality = round(_QUALITY_SCALE * float(evolution_assessment["score"]), 4)
        else:
            direct_answer_hack = local_hack
        if direct_answer_hack:
            sid_reward = retrieval_relevance = evolution_quality = 0.0
        process_scores.extend([
            {"from_idx": tool_start, "to_idx": tool_end, "score_value": sid_reward, "token_type": "meme_sid_action"},
            {"from_idx": tool_start, "to_idx": tool_end, "score_value": retrieval_relevance, "token_type": "meme_retrieval_relevance"},
            {"from_idx": answer_start, "to_idx": answer_end, "score_value": evolution_quality, "token_type": "meme_evolution_quality"}
        ])

    with_memory_em = 0.0
    with_memory_answer = ""
    with_memory_trace = ""
    with_memory_ok = False
    if outcome_gate:
        try:
            with_memory_em, with_memory_answer, with_memory_trace, with_memory_ok = (
                _backend._get_outcome_score_via_deerflow(
                    parsed["evolved_experience"], query, ground_truth, data_source=data_source
                )
            )
        except Exception as exc:
            print(f"[MemE outcome] unavailable: {exc}")
            with_memory_ok = False
    outcome_reward = 2.0 if with_memory_ok and float(with_memory_em) > 0.0 else 0.0
    process_total = round(sum(float(item["score_value"]) for item in process_scores), 4)
    total = round(outcome_reward + process_total, 4)
    if total > 2.3 + 1e-9:
        raise AssertionError(f"MemE reward exceeded 2.3: {total}")

    predicted_codes, target_codes = _parse_sid(parsed["predicted_sid"]), _parse_sid(target_sid)
    matched_levels = sum(a == b for a, b in zip(predicted_codes or (), target_codes or ()))
    reason = evolution_assessment["reason"] if evolution_assessment else "judge unavailable or invalid response"
    trace_payload = _trace_for_rollout(with_memory_trace)
    breakdown = {
        "reward_version": "meme_v1_nonnegative_outcome_dominant_anti_hack",
        **parsed,
        "target_sid": target_sid,
        "retrieval_hit": retrieval_hit,
        "sid_matched_levels": matched_levels,
        "sid_reward": sid_reward,
        "sid_reward_range": [0.05, 0.1],
        "retrieval_relevance": retrieval_relevance,
        "retrieval_score_raw": retrieval_score_raw,
        "retrieval_judge_raw": retrieval_judge_raw[:4000],
        "evolution_quality": evolution_quality,
        "evolution_score_raw": evolution_assessment["score"] if evolution_assessment else None,
        "evolution_reusable": evolution_assessment["reusable"] if evolution_assessment else False,
        "direct_answer_hack": direct_answer_hack,
        "evolution_reason": reason,
        "evolution_judge_raw": evolution_judge_raw[:4000],
        "outcome_gate": outcome_gate,
        "with_memory_em_raw": float(with_memory_em),
        "with_memory_ok": bool(with_memory_ok),
        "outcome_available": bool(with_memory_ok),
        "outcome_score": outcome_reward,
        "process_reward_total": process_total,
        "total_scalar_reward": total,
        "outcome_backend": "deerflow_direct_with_evolved_memory_meme_v1",
        "deerflow_with_memory": {
            "answer": with_memory_answer,
            "em": float(with_memory_em),
            "ok": bool(with_memory_ok),
            "trace": trace_payload,
        },
    }
    print(
        f"[MemE reward] protocol={parsed['protocol_valid']} gate={outcome_gate} "
        f"sid={sid_reward:.3f} outcome={outcome_reward:.1f} available={bool(with_memory_ok)}"
    )
    trace_envelope = json.dumps(
        {"trace_schema": "deerflow_direct_with_evolved_memory_meme_v1", "with_memory": trace_payload},
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
