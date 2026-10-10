# Modified for GenMem; see verl/README.md for provenance.
"""
mem_retriever_reward.py

Reward function for MemRetriever training under the GRPO framework.

Reward structure
----------------
A. Process Reward (assigned to token spans):
   A1. Format reward   – SID tokens conform to <SID_L1_X><SID_L2_Y><SID_L3_Z>
                          with valid integer bounds.
   A2. Semantic matching reward – External LLM scores the relevance of the
                          tool_response (retrieved experience) to the user's query.
                          Reward assigned to the SID token span in <tool_call>.
B. Outcome Reward (assigned to the final EOS token):
   B1. StackPlanner QA – Injects the model-organised experience into
                          StackPlanner and measures answer accuracy vs ground truth.
                          Ground truth = target_answer from dataset (correct answer
                          to the question, not the experience text).

Returns
-------
dict:
  {
    "outcome_score": float,
    "process_scores": list[{
        "from_idx": int,
        "to_idx": int,
        "score_value": float,
        "token_type": str   # "format_sid" | "format_think" | "format_answer"
                            # | "semantic_sid" | "none"
    }],
    # extra fields saved to rollout jsonl:
    "retrieved_experience": str,
    "sp_answer": str,
    "sp_raw_result": str,
    "reward_breakdown": dict,
  }
"""

import asyncio
import json
import os
import sys
import re
import subprocess
import threading
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SID_RE = re.compile(r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)>")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)

_SID_BOUNDS = {"L1": 64, "L2": 32, "L3": 16}

_STACK_PLANNER_PATH = os.getenv("STACK_PLANNER_ROOT", "./external/Stack-Planner")
_SP_PYTHON = os.getenv("STACK_PLANNER_PYTHON", sys.executable)
_SP_SEMAPHORE = threading.Semaphore(3)  # max 3 concurrent SP processes

_QA_RUNNER_SCRIPT = """
import sys, json
sys.path.insert(0, '{stack_planner_path}')
from scripts.qa_api.qa_interface import ask_sync
result = ask_sync({question_repr})
print(json.dumps({{"answer": result.get("answer", ""), "trajectory": result.get("trajectory", [])}}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_sid(m: re.Match) -> bool:
    l1, l2, l3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (
        0 <= l1 < _SID_BOUNDS["L1"]
        and 0 <= l2 < _SID_BOUNDS["L2"]
        and 0 <= l3 < _SID_BOUNDS["L3"]
    )


def _extract_retrieved_experience(solution_str: str) -> str:
    responses = _TOOL_RESP_RE.findall(solution_str)
    return "\n\n".join(r.strip() for r in responses if r.strip())


# ---------------------------------------------------------------------------
# Outcome reward via StackPlanner
# Returns (score, sp_answer, sp_raw_json)
# ---------------------------------------------------------------------------

def _get_outcome_score_via_sp(
    experience: str, query: str, ground_truth: dict
) -> tuple[float, str, str]:
    """
    Call StackPlanner to answer the question given the retrieved experience,
    then compare against target_answer using flexible EM.
    Returns (score_in_0_1, sp_answer_str, sp_raw_json_str).
    """
    from verl.utils.reward_score.search_r1_like_qa_em import (
        compute_score as compute_em_score,
        subem_check,
    )

    target_answer = ground_truth.get("target_answer") if isinstance(ground_truth, dict) else None
    if not target_answer:
        print("[outcome reward] skipped: no target_answer in ground_truth")
        return 0.0, "", "", []

    question = (
        f"Based on the following relevant experience:\n{experience}\n\n"
        f"Please answer the question: {query}"
    )
    script = _QA_RUNNER_SCRIPT.format(
        stack_planner_path=_STACK_PLANNER_PATH,
        question_repr=repr(question),
    )
    _SILENCE_TIMEOUT = 30
    _TOTAL_TIMEOUT = 90

    with _SP_SEMAPHORE:
        try:
            proc = subprocess.Popen(
                [_SP_PYTHON, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=_STACK_PLANNER_PATH,
            )
            stdout_lines = []
            last_output_time = time.time()
            start_time = time.time()

            while True:
                line = proc.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    last_output_time = time.time()
                elif proc.poll() is not None:
                    break
                now = time.time()
                if now - last_output_time > _SILENCE_TIMEOUT:
                    proc.kill()
                    raise TimeoutError(f"SP silent for {_SILENCE_TIMEOUT}s, killed")
                if now - start_time > _TOTAL_TIMEOUT:
                    proc.kill()
                    raise TimeoutError(f"SP exceeded total {_TOTAL_TIMEOUT}s, killed")
                time.sleep(0.2)

            stderr_text = proc.stderr.read()
            stdout_text = "".join(stdout_lines)

            sp_raw = ""
            actual_answer = ""
            sp_trajectory = []
            for line in reversed(stdout_text.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        raw_ans = parsed.get("answer", "")
                        # SP sometimes returns {"action":"finish","result":"X"} instead of {"answer":"X"}
                        if not raw_ans and "result" in parsed:
                            raw_ans = parsed.get("result", "")
                        # Strip pre-wrapped <answer>...</answer> tags SP may include
                        tag_m = re.search(r"<answer>(.*?)</answer>", raw_ans, re.DOTALL)
                        raw_ans = tag_m.group(1).strip() if tag_m else raw_ans
                        # SP sometimes buries answer in {"action":"finish","reasoning":"..."} JSON
                        if raw_ans.strip().startswith("{"):
                            try:
                                action_obj = json.loads(raw_ans)
                                if isinstance(action_obj, dict) and action_obj.get("action") == "finish":
                                    reasoning = action_obj.get("reasoning", "") or action_obj.get("result", "")
                                    short = re.match(r"^[A-Za-z0-9\$\.\-\s]{1,60}", reasoning.strip())
                                    raw_ans = short.group(0).strip() if short else reasoning[:100]
                            except Exception:
                                # JSON truncated/incomplete: extract leading short token directly
                                stripped = re.sub(r'^\{["\s]*\w+["\s]*:["\s]*', "", raw_ans).strip('" ')
                                short = re.match(r"^[A-Za-z0-9\$\.\-\s]{1,60}", stripped)
                                raw_ans = short.group(0).strip() if short else ""
                        actual_answer = raw_ans
                        sp_trajectory = parsed.get("trajectory", [])
                        sp_raw = line
                        break
                    except Exception:
                        continue
            if proc.returncode != 0 and not actual_answer:
                print(f"[outcome reward] subprocess stderr: {stderr_text[-300:]}")
                return 0.0, "", sp_raw, sp_trajectory

            wrapped = f"<answer>{actual_answer}</answer>"
            gt_for_em = {"target": target_answer}
            score = float(compute_em_score(wrapped, gt_for_em, method="flexible"))
            # fallback: subEM — gt contained within sp_answer handles cases like
            # "8 points" vs "8", "results in 222.22" vs "222.22"
            if score == 0.0 and actual_answer:
                if subem_check(actual_answer, target_answer):
                    score = 1.0
            print(
                f"[outcome reward] query={query[:60]!r} | "
                f"sp_answer={actual_answer[:60]!r} | gt={str(target_answer)[:60]!r} | score={score}"
            )
            return score, actual_answer, sp_raw, sp_trajectory
        except Exception as e:
            print(f"[outcome reward] failed: {e}")
            return 0.0, "", "", []


# ---------------------------------------------------------------------------
# Semantic matching reward (LLM-based)
# ---------------------------------------------------------------------------

def _get_semantic_score(query: str, content: str) -> float:
    from verl.utils.reward_score.sp_r1.search_quality_score import AsyncLLMRelevanceScorer

    scorer = AsyncLLMRelevanceScorer(
        model=os.getenv("SUMMARY_MODEL", os.getenv("EVAL_LLM_MODEL_NAME")),
        api_key=os.getenv("SUMMARY_API_KEY", os.getenv("EVAL_LLM_API_KEY")),
        base_url=os.getenv("SUMMARY_BASE_URL", os.getenv("EVAL_LLM_BASE_URL")),
        max_workers=2,
    )

    async def _score():
        return await scorer.score_batch(query, [content])

    try:
        score = asyncio.run(_score())
        return float(score) - 0.5  # shift to [-0.5, 0.5]
    except Exception as e:
        print(f"[semantic reward] failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Main compute_score
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    tokenizer=None,
    input_ids=None,
    **kwargs,
) -> dict:
    """
    Compute process + outcome rewards for a MemRetriever trajectory.

    ground_truth fields used:
        target_answer  – correct answer to the question (for EM outcome reward)
        experience_text – reference experience (for semantic reward context)
        sid_tokens     – correct SID

    Returns dict with outcome_score, process_scores, plus extra fields for logging.
    """
    from verl.spr1_special_tokens.recognition import TokenRecognizer

    extra_info = extra_info or {}
    query = extra_info.get("question", "")
    process_scores = []
    reward_breakdown = {}
    _MIN_EXPERIENCE_LEN = 50  # skip SP if experience is trivially short

    # format_think reward removed — no reward for <think> blocks
    reward_breakdown["format_think"] = 0.0

    # ------------------------------------------------------------------ #
    # Build token-level intervals using TokenRecognizer                    #
    # Positions are token indices (correct for assign_slice).              #
    # Content is decoded from each span for SID/semantic parsing.         #
    # ------------------------------------------------------------------ #
    recognizer = TokenRecognizer(tokenizer) if tokenizer is not None else None

    ids_list = None
    if input_ids is not None:
        import torch
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)

    # Find tool_call and answer token spans
    tc_token_intervals = []   # list of (from_tok, to_tok) inclusive
    ans_token_intervals = []

    if recognizer is not None and ids_list is not None:
        tc_token_intervals = recognizer.find_intervals(
            ids_list, "<tool_call>", "</tool_call>", include_boundaries=True
        )
        ans_token_intervals = recognizer.find_intervals(
            ids_list, "<answer>", "</answer>", include_boundaries=True
        )
        # find_intervals returns [start, end) — convert to inclusive [start, end-1]
        tc_token_intervals  = [(s, e - 1) for s, e in tc_token_intervals]
        ans_token_intervals = [(s, e - 1) for s, e in ans_token_intervals]

    def _decode_span(s, e):
        """Decode token span [s, e] inclusive."""
        if tokenizer is None or ids_list is None:
            return ""
        return tokenizer.decode(ids_list[s: e + 1], skip_special_tokens=False)

    # ------------------------------------------------------------------ #
    # A1. Format reward: SID tokens inside <tool_call>                     #
    # ------------------------------------------------------------------ #
    total_sids = 0
    valid_sids = 0
    sid_fmt_total = 0.0
    n_tc = len(tc_token_intervals)

    if n_tc > 0:
        for s, e in tc_token_intervals:
            content = _decode_span(s, e)
            sid_matches = list(_SID_RE.finditer(content))
            for sid_m in sid_matches:
                total_sids += 1
                if _is_valid_sid(sid_m):
                    valid_sids += 1
            fmt_score = (valid_sids / max(total_sids, 1) * 0.9) if sid_matches else 0.0
            per_tc_score = fmt_score  # each tool_call scored independently, not averaged
            sid_fmt_total += per_tc_score
            process_scores.append({
                "from_idx": s,
                "to_idx": e,
                "score_value": per_tc_score,
                "token_type": "format_sid",
            })
    else:
        process_scores.append({"from_idx": 0, "to_idx": 1, "score_value": 0.0, "token_type": "none"})

    reward_breakdown["format_sid"] = sid_fmt_total
    reward_breakdown["total_sids"] = total_sids
    reward_breakdown["valid_sids"] = valid_sids

    # ------------------------------------------------------------------ #
    # A1. Format reward: <answer> block — only reward the FIRST one       #
    # ------------------------------------------------------------------ #
    # BPE can merge ">\n" into a single token (e.g. token 397), so the
    # token-level search for [27,9217,29] misses "<answer>\n...".
    # Use regex on the decoded string as authoritative fallback.
    _regex_ans_matches = list(_ANSWER_RE.finditer(solution_str))
    if not ans_token_intervals and _regex_ans_matches:
        # Build a dummy interval at end-of-sequence so downstream checks work.
        total_len = kwargs.get("total_len", None)
        proxy = max((total_len - 2), 0) if total_len else 0
        ans_token_intervals = [(proxy, proxy)]

    if ans_token_intervals:
        s, e = ans_token_intervals[0]
        process_scores.append({
            "from_idx": s,
            "to_idx": e,
            "score_value": 0.9,
            "token_type": "format_answer",
        })
        reward_breakdown["format_answer"] = 0.9
        reward_breakdown["num_answer_tags"] = max(len(ans_token_intervals), len(_regex_ans_matches))
    else:
        process_scores.append({"from_idx": 0, "to_idx": 1, "score_value": 0.0, "token_type": "none"})
        reward_breakdown["format_answer"] = 0.0
        reward_breakdown["num_answer_tags"] = 0

    # ------------------------------------------------------------------ #
    # A2. Semantic matching reward: retrieved experience vs query          #
    # Score whether the looked-up memory is relevant to the query.        #
    # Reward is assigned to the SID token span in <tool_call>.            #
    # ------------------------------------------------------------------ #
    sem_total = 0.0
    if query and n_tc > 0:
        # Extract all tool_responses from the decoded solution string
        tool_responses = _TOOL_RESP_RE.findall(solution_str)
        experience_text = "\n\n".join(r.strip() for r in tool_responses if r.strip())
        if experience_text:
            sem_score = _get_semantic_score(query, experience_text)
            sem_total = sem_score
            # Assign to the first tool_call span (the SID that caused the retrieval)
            s, e = tc_token_intervals[0]
            process_scores.append({
                "from_idx": s,
                "to_idx": e,
                "score_value": sem_score,
                "token_type": "semantic_sid",
            })
    reward_breakdown["semantic_sid"] = sem_total

    # ------------------------------------------------------------------ #
    # B. Outcome reward: StackPlanner QA vs target_answer                  #
    # ------------------------------------------------------------------ #
    # Prefer <answer> content if long enough, else fallback to tool_response
    retrieved_experience = ""
    if ans_token_intervals:
        s, e = ans_token_intervals[-1]
        content = _decode_span(s, e).strip()
        # strip the <answer>/</answer> tags themselves
        content = _ANSWER_RE.search(content)
        content = content.group(1).strip() if content else ""
        if len(content) >= _MIN_EXPERIENCE_LEN:
            retrieved_experience = content
    if not retrieved_experience:
        retrieved_experience = _extract_retrieved_experience(solution_str)

    # Extract model's <answer> content for logging
    answer_matches = list(_ANSWER_RE.finditer(solution_str))
    model_answer = answer_matches[-1].group(1).strip() if answer_matches else ""

    sp_answer = ""
    sp_raw = ""
    sp_trajectory = []
    if retrieved_experience and len(retrieved_experience) >= _MIN_EXPERIENCE_LEN and query:
        raw_score, sp_answer, sp_raw, sp_trajectory = _get_outcome_score_via_sp(
            retrieved_experience, query, ground_truth
        )
        outcome_score = raw_score * 5
    else:
        print(
            f"[outcome reward] skipped: "
            f"experience_found={bool(retrieved_experience)}, "
            f"exp_len={len(retrieved_experience)}, query={bool(query)}"
        )
        outcome_score = 0.0

    reward_breakdown["outcome_score_raw"] = outcome_score / 5 if outcome_score else 0.0
    reward_breakdown["outcome_score"] = outcome_score

    # ------------------------------------------------------------------ #
    # D. Penalty: repeated tool_calls with no <answer>                    #
    # ------------------------------------------------------------------ #
    _NO_ANSWER_TC_THRESHOLD = 3
    if not ans_token_intervals and n_tc >= _NO_ANSWER_TC_THRESHOLD:
        total_len = kwargs.get("total_len", None)
        eos_pos = (total_len - 1) if total_len else 1
        penalty = -1.0
        process_scores.append({
            "from_idx": eos_pos,
            "to_idx": eos_pos,
            "score_value": penalty,
            "token_type": "no_answer_penalty",
        })
        reward_breakdown["no_answer_penalty"] = penalty
    else:
        reward_breakdown["no_answer_penalty"] = 0.0

    print(f"outcome_reward: {outcome_score}")
    print(f"process_reward: {process_scores}")

    if not process_scores:
        process_scores.append({"from_idx": 0, "to_idx": 1, "score_value": 0.0, "token_type": "none"})

    return {
        "outcome_score": outcome_score,
        "process_scores": process_scores,
        # extra fields for rollout logging
        "model_answer": model_answer,
        "retrieved_experience": retrieved_experience,
        "sp_answer": sp_answer,
        "sp_trajectory": json.dumps(sp_trajectory, ensure_ascii=False),
        "sp_raw_result": sp_raw,
        "reward_breakdown": json.dumps(reward_breakdown, ensure_ascii=False),
    }
