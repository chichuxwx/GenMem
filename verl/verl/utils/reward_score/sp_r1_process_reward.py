# Modified for GenMem; see verl/README.md for provenance.
# verl/utils/reward_score/sp_r1_process_reward.py
import re
import random
from verl.utils.reward_score.search_r1_like_qa_em import compute_score as compute_outcome_score
import sys
import os
import asyncio

from verl.spr1_special_tokens.recognition import TokenRecognizer
from verl.utils.reward_score.sp_r1.search_quality_score import AsyncLLMRelevanceScorer

_STACK_PLANNER_PATH = os.getenv('STACK_PLANNER_ROOT', './external/Stack-Planner')
_SP_PYTHON = os.getenv('STACK_PLANNER_PYTHON', sys.executable)

# Inline script run inside the sp environment via subprocess
_QA_RUNNER_SCRIPT = """
import sys, json
sys.path.insert(0, '{stack_planner_path}')
from scripts.qa_api.qa_interface import ask_sync
result = ask_sync({question_repr})
print(json.dumps({{"answer": result.get("answer", "")}}))
"""


def _get_outcome_score_via_qa(experience: str, query: str, ground_truth: dict) -> float:
    """
    Call Stack-Planner QA system (sp conda env) with the model's experience as context.
    Uses subprocess so that Stack-Planner's own dependencies don't conflict with verl-ragas.
    Returns outcome score by comparing the QA system's actual answer to ground_truth.
    """
    import subprocess, json as _json

    question = (
        f"Based on the following relevant experience:\n{experience}\n\n"
        f"Please answer the question: {query}"
    )
    script = _QA_RUNNER_SCRIPT.format(
        stack_planner_path=_STACK_PLANNER_PATH,
        question_repr=repr(question),
    )
    try:
        proc = subprocess.run(
            [_SP_PYTHON, '-c', script],
            capture_output=True, text=True, timeout=120,
            cwd=_STACK_PLANNER_PATH,
        )
        # find the last JSON line in stdout
        actual_answer = ""
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    actual_answer = _json.loads(line).get("answer", "")
                    break
                except Exception:
                    continue
        if proc.returncode != 0 and not actual_answer:
            print(f"[outcome reward] subprocess stderr: {proc.stderr[-500:]}")
            return 0.0
        wrapped = f"<answer>{actual_answer}</answer>"
        score = compute_outcome_score(wrapped, ground_truth, method="flexible")
        print(f"[outcome reward] query={query[:80]!r} | answer={actual_answer[:80]!r} | score={score}")
        return float(score)
    except Exception as e:
        print(f"[outcome reward] failed: {e}")
        return 0.0

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reflect_reward=1.0,
    **kwargs
):
    """
    Compute both outcome reward and process reward for a QA example.

    Args:
        data_source: str, data source name
        solution_str: str, model-generated text
        ground_truth: dict, ground truth answer info
        extra_info: dict, optional extra context
        reflect_reward: float, reward to give for <reflect> tags
        mock_tool_call: bool, if True, mock tool_call reward instead of calling real evaluator
        kwargs: additional args

    Returns:
        dict:
            {
                "outcome_score": float,
                "process_scores": list[dict]  # each dict: {"from_idx", "to_idx", "score_value", "token_type"}
            }
    """
    extra_info = extra_info or {}
    process_scores = []

    # -------------------------
    # Outcome reward: extract experience from <answer> tags, call Stack-Planner QA system
    # -------------------------
    query = (extra_info or {}).get("question", "")
    experience_matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if experience_matches and query:
        experience = experience_matches[-1].group(1).strip()
        outcome_score = _get_outcome_score_via_qa(experience, query, ground_truth) * 5
    else:
        print(f"[outcome reward] skipped: experience_found={bool(experience_matches)}, query={bool(query)!r}")
        outcome_score = 0.0

    # -------------------------
    # Process reward: 
    # -------------------------
    tokenizer = kwargs.get('tokenizer')
    input_ids = kwargs.get('input_ids')

    tokenizer = None # FIXME 
    
    ## NOTE 这里不需要tokenizer
    if not (tokenizer is not None and input_ids is not None and TokenRecognizer is not None):

        # 0. Action Format
        matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.9 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": per_score,
                    "token_type": "format"
                })
        if not re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })
        
        matches = list(re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.3 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": per_score,
                    "token_type": "format"
                })

        if not re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })

        # 2. Reflect
        matches = list(re.finditer(r"<reflect>(.*?)</reflect>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.7 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": 0, 
                    "token_type": "reflect"
                })  
        if not re.search(r"<reflect>(.*?)</reflect>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })

        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL):
            start, end = match.span()
            content = match.group(1).strip()

            #### jr
            import os
            from dotenv import load_dotenv
            load_dotenv()
            scorer = AsyncLLMRelevanceScorer(
                model=os.getenv("EVAL_LLM_MODEL_NAME"),
                api_key=os.getenv("EVAL_LLM_API_KEY"),
                base_url=os.getenv("EVAL_LLM_BASE_URL"),
                max_workers=2,
            )
            # 用户输入和候选内容列表
            user_input = extra_info.get("question", "")
            retrieved_contexts = [content]  # 可以是单条也可以是多条
            # 异步计算相关性分数
            async def compute_score():
                print(f"user_input: {user_input}")
                print(f"retrieved_contexts: {retrieved_contexts}")

                score = await scorer.score_batch(user_input, retrieved_contexts)
                return score

            try:
                # raise ValueError("Mock tool_call reward not implemented.")
                score_value = asyncio.run(compute_score())
            except Exception as e:
                print(f"Error evaluating tool_call: {e}")
                score_value = 0.0

            process_scores.append({
                "from_idx": start,
                "to_idx": end - 1,
                # "score_value": 0,
                "score_value": score_value-0.5, # 由于返回的是0和1，需要变为负数
                "token_type": "tool_call"
            })

    if len(process_scores) == 0:
        process_scores.append({
            "from_idx": 0,
            "to_idx": 1,
            "score_value": 0.0,
            "token_type": "none"
        })

    return {
        "outcome_score": outcome_score,
        "process_scores": process_scores
    }
