"""Retrieval-first nonnegative reward for the v2 hard-failure experiment.

Every usable retrieval receives a positive SID base reward. Semantic and
rewrite quality are nonnegative in ``[0, 0.1]``. The task outcome is scored
directly from the DeerFlow run that receives the model's rewritten answer when
available, falling back to the retrieved experience only when no answer is
present: correct is 2, while incorrect or unavailable is 0. The historical
dataset already establishes the no-memory baseline as a stable failure, so no
counterfactual baseline run is issued during training.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional


_SID_RE = re.compile(r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>")
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SID_BOUNDS = (48, 16, 8, 8)
_SID_BASE_REWARD = 0.05
_SID_LEVEL_REWARD = 0.0125
_QUALITY_REWARD_SCALE = 0.1
_JUDGE_MAX_ATTEMPTS = 3
_JUDGE_ENDPOINT = "http://127.0.0.1:8010/v1"
_JUDGE_MODEL = "Qwen3-32B"


def _parse_unit_interval_score(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r'"score"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', text)
    if not match:
        match = re.fullmatch(
            r"(?:score\s*[:=]\s*)?(-?(?:\d+(?:\.\d*)?|\.\d+))",
            text.splitlines()[0].strip(),
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    score = float(match.group(1))
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _load_backend():
    configured = os.getenv("DEERFLOW_HELPERS_PATH", "").strip()
    path = Path(configured) if configured else Path(__file__).with_name("deerflow_outcome_helpers_v33.py")
    if not path.is_file():
        class MissingBackend:
            pass
        return MissingBackend()
    spec = importlib.util.spec_from_file_location("deerflow_outcome_helpers_v33", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DeerFlow helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backend = _load_backend()


def _parse_sid(value: str) -> tuple[int, int, int, int] | None:
    match = _SID_RE.fullmatch((value or "").strip())
    if not match:
        return None
    codes = tuple(int(match.group(index)) for index in range(1, 5))
    return codes if all(code < bound for code, bound in zip(codes, _SID_BOUNDS)) else None


def sid_process_reward(predicted_sid: str, ground_truth_sid: str) -> float:
    predicted, target = _parse_sid(predicted_sid), _parse_sid(ground_truth_sid)
    if predicted is None or target is None:
        return 0.0
    return round(
        _SID_BASE_REWARD + _SID_LEVEL_REWARD * sum(a == b for a, b in zip(predicted, target)),
        4,
    )


def _tool_span(tokenizer, input_ids, total_len: int | None) -> tuple[int, int, str]:
    if tokenizer is not None and input_ids is not None:
        try:
            from verl.spr1_special_tokens.recognition import TokenRecognizer
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            intervals = TokenRecognizer(tokenizer).find_intervals(ids, "<tool_call>", "</tool_call>", include_boundaries=True)
            if intervals:
                start, end = intervals[0]
                return int(start), max(int(start), int(end) - 1), "sid_supervision"
        except Exception:
            pass
    eos = max(int(total_len or 1) - 1, 0)
    return eos, eos, "eos"


def _answer_span(tokenizer, input_ids, total_len: int | None) -> tuple[int, int, str]:
    if tokenizer is not None and input_ids is not None:
        try:
            from verl.spr1_special_tokens.recognition import TokenRecognizer
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            intervals = TokenRecognizer(tokenizer).find_intervals(ids, "<answer>", "</answer>", include_boundaries=True)
            if intervals:
                start, end = intervals[0]
                return int(start), max(int(start), int(end) - 1), "rewrite_quality"
        except Exception:
            pass
    eos = max(int(total_len or 1) - 1, 0)
    return eos, eos, "eos"


def _extract_model_answer(solution: str) -> str:
    """Return the final nonempty answer, tolerating a missing final close tag.

    The policy-authored content is never invented: if the final ``<answer>``
    has no matching closing delimiter before the response ends, its suffix is
    treated as the answer.  A closed final span retains the existing behavior.
    """

    answer_matches = list(_ANSWER_RE.finditer(solution))
    last_open = solution.rfind("<answer>")
    last_close = solution.rfind("</answer>")
    if last_open > last_close:
        return solution[last_open + len("<answer>") :].strip()
    return answer_matches[-1].group(1).strip() if answer_matches else ""


def _last_retrieval_attempt(solution: str) -> tuple[list[str], list[str], str, str]:
    """Return all calls/SIDs plus the last call's SID and observation.

    Rollouts may contain an exploratory call before their final choice.  The
    final call and final tool observation are the action judged by reward.
    """
    tool_calls = _TOOL_CALL_RE.findall(solution or "")
    sid_strings = [match.group(0) for call in tool_calls for match in _SID_RE.finditer(call)]
    if not tool_calls:
        return tool_calls, sid_strings, "", ""
    last_sids = [match.group(0) for match in _SID_RE.finditer(tool_calls[-1])]
    predicted_sid = last_sids[0] if len(last_sids) == 1 and _parse_sid(last_sids[0]) is not None else ""
    responses = [item.strip() for item in _TOOL_RESPONSE_RE.findall(solution or "") if item.strip()]
    return tool_calls, sid_strings, predicted_sid, (responses[-1] if responses else "")


def _judge_score(prompt: str, label: str) -> Optional[float]:
    for attempt in range(1, _JUDGE_MAX_ATTEMPTS + 1):
        try:
            from openai import OpenAI
            endpoint = os.getenv("SUMMARY_BASE_URL", os.getenv("EVAL_LLM_BASE_URL", _JUDGE_ENDPOINT))
            model = os.getenv("SUMMARY_MODEL", os.getenv("EVAL_LLM_MODEL_NAME", _JUDGE_MODEL))
            api_key = os.getenv("SUMMARY_API_KEY", os.getenv("EVAL_LLM_API_KEY", "EMPTY"))
            client = OpenAI(api_key=api_key, base_url=endpoint, timeout=30, max_retries=0)
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                max_tokens=128, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            score = _parse_unit_interval_score(message.content or getattr(message, "reasoning_content", None) or "")
            if score is not None:
                return score
            raise ValueError("judge response lacked a unit-interval score")
        except Exception as exc:
            print(f"[v2 {label}] attempt {attempt}/{_JUDGE_MAX_ATTEMPTS} failed: {exc}")
            if attempt < _JUDGE_MAX_ATTEMPTS:
                time.sleep(1)
    return None


def _get_semantic_score(query: str, content: str) -> Optional[float]:
    return _judge_score(
        "You are a retrieval-usefulness judge for a memory agent. Score how helpful the retrieved experience is for answering the query. Give 0.0 when it is irrelevant or unusable; give a modest positive score (about 0.2-0.5) for a broadly applicable strategy, domain knowledge, or verification procedure that can reasonably help; reserve 0.6-1.0 for query-specific facts, constraints, or directly applicable solution steps. Return only JSON: {\"score\": 0.0}.\n\n"
        f"QUERY:\n{query[:4000]}\n\nRETRIEVED EXPERIENCE:\n{content[:16000]}",
        "semantic reward",
    )


def _get_rewrite_score(query: str, content: str) -> Optional[float]:
    return _judge_score(
        "You are a strict answerability judge for a memory agent. Given a query and a rewritten experience, judge whether a downstream agent with only the query and this experience has enough query-specific facts, strategy, constraints, and steps to solve it. Do not reward generic boilerplate, shared task labels, verbosity, or plausible but unsupported claims. Return only JSON: {\"score\": 0.0}.\n\n"
        f"QUERY:\n{query[:4000]}\n\nREWRITTEN EXPERIENCE:\n{content[:16000]}",
        "rewrite reward",
    )


def _trace_for_rollout(trace: str):
    converter = getattr(_backend, "_trace_for_rollout", None)
    if converter is not None:
        return converter(trace)
    try:
        return json.loads(trace) if trace else None
    except (json.JSONDecodeError, TypeError):
        return {"trace_schema": "unparsed_text_v1", "raw": str(trace)}


def _quality_reward(get_score: Callable[[str, str], Optional[float]], query: str, text: str) -> tuple[float, Optional[float], bool]:
    raw = get_score(query, text)
    return (round(_QUALITY_REWARD_SCALE * raw, 4), raw, True) if raw is not None else (0.0, None, False)


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: Optional[dict] = None, tokenizer=None, input_ids=None, **kwargs) -> dict:
    extra_info = extra_info or {}
    ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
    solution = solution_str or ""
    if tokenizer is not None and input_ids is not None:
        try:
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            solution = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            pass
    query = str(extra_info.get("question") or "")
    tool_calls, sid_strings, predicted_sid, retrieved_experience = _last_retrieval_attempt(solution)
    valid_sids = [sid for sid in sid_strings if _parse_sid(sid) is not None]
    single_sid_action = bool(predicted_sid)
    target_sid = str(ground_truth.get("sid_tokens") or "")
    retrieval_attempted = single_sid_action and bool(query)
    eligible = retrieval_attempted and len(retrieved_experience) >= 50
    span_start, span_end, span_kind = _tool_span(tokenizer, input_ids, kwargs.get("total_len"))
    process_scores = []
    sid_reward = semantic_sid = rewrite_quality = 0.0
    semantic_score_raw = rewrite_score_raw = None
    semantic_judge_used = rewrite_judge_used = False
    if eligible:
        sid_reward = sid_process_reward(predicted_sid, target_sid)
        process_scores.append({"from_idx": span_start, "to_idx": span_end, "score_value": sid_reward, "token_type": span_kind})
        semantic_sid, semantic_score_raw, semantic_judge_used = _quality_reward(_get_semantic_score, query, retrieved_experience)
        process_scores.append({"from_idx": span_start, "to_idx": span_end, "score_value": semantic_sid, "token_type": "semantic_sid"})

    model_answer = _extract_model_answer(solution)
    answer_eligible = retrieval_attempted and bool(model_answer)
    if answer_eligible:
        rewrite_quality, rewrite_score_raw, rewrite_judge_used = _quality_reward(_get_rewrite_score, query, model_answer)
        answer_start, answer_end, _ = _answer_span(tokenizer, input_ids, kwargs.get("total_len"))
        process_scores.append({"from_idx": answer_start, "to_idx": answer_end, "score_value": rewrite_quality, "token_type": "rewrite_quality"})

    outcome_experience = model_answer if model_answer else (retrieved_experience if eligible else "")
    outcome_experience_source = "model_answer" if model_answer else ("retrieved_experience_fallback" if eligible else "none")
    with_memory_em = 0.0
    with_memory_answer = with_memory_trace = ""
    with_memory_ok = False
    outcome_available = False
    outcome_reward = 0.0
    if answer_eligible or eligible:
        with_memory_em, with_memory_answer, with_memory_trace, with_memory_ok = _backend._get_outcome_score_via_deerflow(outcome_experience, query, ground_truth, data_source=data_source)
        outcome_available = bool(with_memory_ok)
        if outcome_available:
            outcome_reward = 2.0 if float(with_memory_em) > 0.0 else 0.0
        else:
            outcome_reward = 0.0

    predicted_codes, target_codes = _parse_sid(predicted_sid), _parse_sid(target_sid)
    sid_matched_levels = sum(a == b for a, b in zip(predicted_codes or (), target_codes or ()))
    sid_prefix_len = 0
    if predicted_codes is not None and target_codes is not None:
        for index, (a, b) in enumerate(zip(predicted_codes, target_codes)):
            if a != b:
                break
            sid_prefix_len = index + 1
    process_total = sum(float(item["score_value"]) for item in process_scores)
    breakdown = {
        "reward_version": "v33_nonnegative_retrieval_first_v2", "eligible": eligible,
        "retrieval_attempted": retrieval_attempted, "answer_eligible": answer_eligible,
        "single_sid_action": single_sid_action, "num_tool_calls": len(tool_calls), "total_sids": len(sid_strings),
        "valid_sids": len(valid_sids), "predicted_sid": predicted_sid, "target_sid": target_sid,
        "used_tool_call_index": len(tool_calls) - 1, "retrieved_experience": retrieved_experience,
        "sid_prefix_len": sid_prefix_len, "sid_matched_levels": sid_matched_levels, "sid_reward": sid_reward,
        "sid_reward_range": [0.05, 0.1], "semantic_sid": semantic_sid, "semantic_score_raw": semantic_score_raw,
        "semantic_judge_used": semantic_judge_used, "rewrite_quality": rewrite_quality,
        "rewrite_score_raw": rewrite_score_raw, "rewrite_judge_used": rewrite_judge_used,
        "with_memory_em_raw": float(with_memory_em), "with_memory_ok": bool(with_memory_ok),
        "outcome_experience_source": outcome_experience_source,
        "outcome_available": outcome_available, "outcome_score": outcome_reward, "process_reward_total": process_total,
        "total_scalar_reward": outcome_reward + process_total, "outcome_backend": "deerflow_direct_with_memory_v2",
        "deerflow_with_memory": {"answer": with_memory_answer, "em": float(with_memory_em), "ok": bool(with_memory_ok), "trace": _trace_for_rollout(with_memory_trace)},
    }
    print(f"[v2 reward] eligible={eligible} sid={sid_reward:.3f} outcome={outcome_reward:.1f} available={outcome_available}")
    trace_envelope = json.dumps({"trace_schema": "deerflow_direct_with_memory_v2", "with_memory": _trace_for_rollout(with_memory_trace)}, ensure_ascii=False)
    return {"outcome_score": outcome_reward, "process_scores": process_scores, "model_answer": model_answer, "retrieved_experience": retrieved_experience, "sp_answer": with_memory_answer, "sp_trajectory": trace_envelope, "sp_raw_result": with_memory_trace, "deerflow_answer": with_memory_answer, "deerflow_result": with_memory_trace, "reward_breakdown": json.dumps(breakdown, ensure_ascii=False)}
