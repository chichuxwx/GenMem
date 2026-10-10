"""
sp_runner.py

Calls StackPlanner (SP) as a subprocess to obtain a trajectory for a given question.
SP is called with `question + curated_experience` as context (same pattern as reward fn).

Returns:
    {
        "trajectory_text": str,   # formatted trajectory for MemE input
        "sp_answer":       str,   # SP's final answer
        "sp_success":      bool,  # whether SP's answer matches ground_truth
        "memory_stack_log": list, # raw SP memory_stack_log (if available)
    }
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import select
import signal
import subprocess
import threading
import time
import urllib.error

from answer_scoring import answers_match

logger = logging.getLogger(__name__)

_STACK_PLANNER_PATH = os.getenv("STACK_PLANNER_PATH", "./private_inputs/Stack-Planner")
_SP_PYTHON = os.getenv("SP_PYTHON", "./private_inputs/python")
_SP_SEMAPHORE = threading.Semaphore(4)  # max concurrent SP processes

_SP_LLM_ENDPOINTS = [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8010/v1",
    "http://127.0.0.1:8030/v1",
]
_SP_ENDPOINT_LOCK = threading.Lock()
_SP_ENDPOINT_IDX = 0

_DIRECT_LLM_URL = os.getenv("DIRECT_LLM_URL", "http://127.0.0.1:8000/v1")
_DIRECT_LLM_MODEL = os.getenv("DIRECT_LLM_MODEL", "Qwen3-32B")
_DIRECT_LLM_BACKEND = os.getenv("DIRECT_LLM_BACKEND", "qwen")
_MINIMAX_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
_MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")
_REMOTE_SOLVER_LOCK = threading.Lock()
_REMOTE_SOLVER_INDEX = 0


def _next_remote_solver_url() -> str:
    """Round-robin remote DeerFlow endpoints when a pool is configured."""
    global _REMOTE_SOLVER_INDEX
    raw = os.getenv("REMOTE_DEERFLOW_SOLVER_URLS", "").strip()
    urls = [item.strip() for item in raw.split(",") if item.strip()]
    if not urls:
        return os.getenv("REMOTE_DEERFLOW_SOLVER_URL", "http://127.0.0.1:18160/solve")
    with _REMOTE_SOLVER_LOCK:
        url = urls[_REMOTE_SOLVER_INDEX % len(urls)]
        _REMOTE_SOLVER_INDEX += 1
    return url


def _direct_api_config() -> tuple[str, str, str, bool]:
    """Return (base_url, model, api_key, is_minimax) without logging secrets."""
    if _DIRECT_LLM_BACKEND.lower() == "minimax":
        return (
            _MINIMAX_URL,
            _MINIMAX_MODEL,
            os.getenv("MINIMAX_API_KEY", ""),
            True,
        )
    return (
        _DIRECT_LLM_URL,
        _DIRECT_LLM_MODEL,
        os.getenv("DIRECT_LLM_API_KEY", "EMPTY"),
        False,
    )


def _api_body(model: str, messages: list[dict], max_tokens: int, is_minimax: bool) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if is_minimax:
        # MiniMax's OpenAI-compatible API uses max_completion_tokens.
        body["max_completion_tokens"] = max_tokens
        # Keep chain-of-thought out of message.content so the existing
        # Answer:/Reasoning: parser receives the final answer text.
        body["reasoning_split"] = True
    else:
        body["max_tokens"] = max_tokens
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _content_without_thinking(content: str) -> str:
    """Remove MiniMax/Qwen inline reasoning before parsing structured output."""
    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()


def _post_json_with_retry(request, timeout: int) -> tuple[dict, int]:
    """POST one API request, retrying transient API/network failures."""
    import urllib.request

    max_retries = max(0, int(os.getenv("DIRECT_LLM_MAX_RETRIES", "2")))
    base_delay = max(0.1, float(os.getenv("DIRECT_LLM_RETRY_DELAY", "2")))
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), retries
        except urllib.error.HTTPError as exc:
            # Remote DeerFlow may briefly return 503 while its worker pool is
            # saturated. Treat the transient gateway/service errors like 429
            # so one busy request does not abort an otherwise healthy run.
            if exc.code not in (429, 500, 502, 503, 504) or attempt >= max_retries:
                setattr(exc, "retry_count", retries)
                raise
            retries += 1
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
            logger.warning(
                "LLM request returned HTTP %s; retry %s/%s in %.1fs",
                exc.code,
                retries,
                max_retries,
                delay,
            )
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                setattr(exc, "retry_count", retries)
                raise
            retries += 1
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
            logger.warning(
                "Direct %s request failed (%s); retry %s/%s in %.1fs",
                _DIRECT_LLM_BACKEND,
                exc,
                retries,
                max_retries,
                delay,
            )
            time.sleep(delay)


def _next_sp_endpoint() -> str:
    global _SP_ENDPOINT_IDX
    with _SP_ENDPOINT_LOCK:
        url = _SP_LLM_ENDPOINTS[_SP_ENDPOINT_IDX % len(_SP_LLM_ENDPOINTS)]
        _SP_ENDPOINT_IDX += 1
    return url


_QA_RUNNER_SCRIPT = """
import sys, json, threading
sys.path.insert(0, '{stack_planner_path}')

_hb_stop = threading.Event()
def _heartbeat():
    while not _hb_stop.wait(30):
        print('SP_HEARTBEAT', flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()

from scripts.qa_api.qa_interface import ask_sync
result = ask_sync({question_repr}, max_step_num=2)
_hb_stop.set()
print(json.dumps({{
    "answer": result.get("answer", ""),
    "trajectory": result.get("trajectory", []),
    "memory_stack_log": result.get("experience_data", {{}}).get("memory_stack_log", []),
}}, ensure_ascii=False))
"""


def _parse_sp_stdout(text: str) -> tuple[dict | None, str]:
    """Extract last JSON object from SP stdout."""
    stripped = text.strip()
    # Strategy 1: last single-line JSON
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line), line
            except (json.JSONDecodeError, ValueError):
                continue
    # Strategy 2: last balanced brace block
    depth = 0
    end_ci = -1
    for ci in range(len(stripped) - 1, -1, -1):
        ch = stripped[ci]
        if ch == "}":
            if depth == 0:
                end_ci = ci
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end_ci >= 0:
                candidate = stripped[ci: end_ci + 1]
                try:
                    return json.loads(candidate), candidate
                except (json.JSONDecodeError, ValueError):
                    pass
                end_ci = -1
    return None, ""


def _format_memory_stack_log(msl: list, max_steps: int = 10) -> str:
    """Format SP memory_stack_log into readable trajectory text."""
    if not msl:
        return ""
    parts = []
    for i, entry in enumerate(msl[:max_steps]):
        action = entry.get("action", "unknown")
        agent_type = entry.get("agent_type", "")
        content = entry.get("content", "")
        if isinstance(content, dict):
            content = content.get("content", json.dumps(content, ensure_ascii=False))
        elif not isinstance(content, str):
            content = str(content)
        prefix = f"[{action}]" if not agent_type else f"[{action} -> {agent_type}]"
        step_text = f"Step {i+1}: {prefix} {content[:400]}"

        result = entry.get("result")
        if result and isinstance(result, dict):
            observations = result.get("observations", [])
            obs_parts = []
            for obs in observations[:2]:
                obs_str = obs if isinstance(obs, str) else obs.get("content", str(obs))
                obs_parts.append(f"  → {str(obs_str)[:200]}")
            if obs_parts:
                step_text += "\n" + "\n".join(obs_parts)
        parts.append(step_text)
    return "\n".join(parts)


def run_sp(
    question: str,
    curated_experience: str,
    ground_truth_answer: str = "",
    silence_timeout: int = 180,
    total_timeout: int = 360,
    max_retries: int = 2,
) -> dict:
    """
    Call SP with question + curated_experience context.

    Returns dict with: trajectory_text, sp_answer, sp_success, memory_stack_log
    """
    # Build the question fed to SP
    if curated_experience and len(curated_experience.strip()) > 20:
        sp_question = (
            f"Based on the following relevant experience:\n{curated_experience}\n\n"
            f"Please answer the question: {question}"
        )
    else:
        sp_question = question

    script = _QA_RUNNER_SCRIPT.format(
        stack_planner_path=_STACK_PLANNER_PATH,
        question_repr=repr(sp_question),
    )

    empty_result = {
        "trajectory_text": "",
        "sp_answer": "",
        "sp_success": False,
        "memory_stack_log": [],
    }

    # Lightweight solver used for offline experiments when the DeerFlow
    # gateway is unavailable. This talks to the existing OpenAI-compatible
    # Qwen service directly and does not invoke the SP 2.0 interface.
    sp_backend = os.getenv("SP_BACKEND", "").lower()
    if sp_backend == "direct_llm":
        return _run_direct_llm(
            question, curated_experience, ground_truth_answer, total_timeout
        )

    # Run the solver through an isolated DeerFlow/StackPlanner worker.  This is
    # used for search-agent checkpoints whose behaviour depends on the native
    # control/tool protocol and therefore cannot be evaluated as plain chat.
    if sp_backend == "remote_deerflow":
        return _run_remote_deerflow(
            question, curated_experience, ground_truth_answer, total_timeout
        )

    # The migrated Stack-Planner environment may not have its optional
    # LangChain integrations, while the local DeerFlow service is already
    # available. Use it explicitly for small evolution smoke tests so the
    # MemE stage still receives a real solver trajectory.
    if os.getenv("SP_FALLBACK_DEERFLOW", "0") == "1":
        return _run_deerflow_fallback(
            question, curated_experience, ground_truth_answer, total_timeout
        )

    def _kill_pg(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    for attempt in range(max_retries):
        with _SP_SEMAPHORE:
            try:
                endpoint = _next_sp_endpoint()
                env = os.environ.copy()
                env["BASIC_MODEL__base_url"] = endpoint
                env["MAX_SEARCH_RESULTS"] = "2"

                proc = subprocess.Popen(
                    [_SP_PYTHON, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=_STACK_PLANNER_PATH,
                    env=env,
                    start_new_session=True,
                )

                stdout_lines = []
                last_out = time.time()
                start = time.time()
                timed_out = False

                try:
                    while True:
                        ready, _, _ = select.select([proc.stdout], [], [], 0.2)
                        if ready:
                            line = proc.stdout.readline()
                            if line:
                                stdout_lines.append(line)
                                last_out = time.time()
                            elif proc.poll() is not None:
                                break
                        else:
                            if proc.poll() is not None:
                                break
                        now = time.time()
                        if now - last_out > silence_timeout:
                            timed_out = True
                            break
                        if now - start > total_timeout:
                            timed_out = True
                            break
                finally:
                    _kill_pg(proc)
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        proc.stderr.close()
                    except Exception:
                        pass

                if timed_out:
                    logger.warning(f"SP timed out (attempt {attempt+1})")
                    continue

                stdout_text = "".join(stdout_lines)
                parsed, _ = _parse_sp_stdout(stdout_text)
                if parsed is None:
                    logger.warning(f"SP output parse failed (attempt {attempt+1}): {stdout_text[-200:]!r}")
                    continue

                sp_answer = parsed.get("answer", "")
                # Strip <answer> tags if present
                m = re.search(r"<answer>(.*?)</answer>", sp_answer, re.DOTALL)
                if m:
                    sp_answer = m.group(1).strip()

                memory_stack_log = parsed.get("memory_stack_log", [])

                # Build trajectory text for MemE
                trace = _format_memory_stack_log(memory_stack_log) if memory_stack_log else ""
                if not trace:
                    # Fallback: use curated_experience as proxy trace
                    trace = curated_experience

                traj_parts = []
                if question:
                    traj_parts.append(f"Question: {question}")
                if trace:
                    traj_parts.append(f"Execution:\n{trace}")
                if sp_answer:
                    # GT is evaluator-only.  Do not annotate the trajectory
                    # with (correct)/(incorrect): that text is passed to
                    # MemE and would leak the reference answer status.
                    sp_success = (
                        _check_answer(sp_answer, ground_truth_answer)
                        if ground_truth_answer else False
                    )
                    traj_parts.append(f"Answer given: {sp_answer}")
                else:
                    sp_success = False

                trajectory_text = "\n".join(traj_parts)

                logger.debug(f"SP answer: {sp_answer[:80]!r}")
                return {
                    "trajectory_text": trajectory_text,
                    "sp_answer": sp_answer,
                    "sp_success": sp_success,
                    "memory_stack_log": memory_stack_log,
                }

            except Exception as e:
                logger.warning(f"SP attempt {attempt+1} failed: {e}")

    return empty_result


def _run_direct_llm(
    question: str,
    curated_experience: str,
    ground_truth_answer: str,
    timeout: int,
) -> dict:
    """Run a compact QA solver through Qwen or MiniMax."""
    import urllib.request

    prompt = (
        "Answer the question directly and concisely. Use the memory only when it "
        "is relevant; do not repeat generic memory text. Return exactly two lines:\n"
        "Answer: <final answer>\n"
        "Reasoning: <short reusable reasoning or verification lesson>\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved experience:\n{curated_experience or '(none)'}"
    )
    messages = [
            {"role": "system", "content": "You are a precise question-answering solver."},
            {"role": "user", "content": prompt},
        ]
    base_url, model, api_key, is_minimax = _direct_api_config()
    body = _api_body(model, messages, 384, is_minimax)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    empty_result = {
        "trajectory_text": "",
        "sp_answer": "",
        "sp_success": False,
        "sp_request_success": False,
        "sp_error": "",
        "sp_retry_count": 0,
        "memory_stack_log": [],
    }
    try:
        payload, retry_count = _post_json_with_retry(request, timeout)
        message = payload["choices"][0]["message"]
        # The served Qwen endpoint may place non-thinking output in the
        # OpenAI ``reasoning`` field when thinking is disabled.
        content = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )
        content = _content_without_thinking(content)
        answer_match = re.search(
            r"(?ims)^\s*Answer:\s*(.+?)(?=\n\s*Reasoning\s*:|\Z)", content
        )
        answer = answer_match.group(1).strip() if answer_match else content.strip()
        reason_match = re.search(r"(?ims)^\s*Reasoning:\s*(.+)$", content)
        reasoning = reason_match.group(1).strip() if reason_match else content.strip()
        if not answer:
            empty_result["sp_error"] = "empty_response"
            empty_result["sp_retry_count"] = retry_count
            return empty_result
        success = _check_answer(answer, ground_truth_answer) if ground_truth_answer else False
        # Keep the reference answer strictly outside the trajectory.  It is
        # used only below for evaluator bookkeeping.
        trajectory = (
            f"Question: {question}\n"
            f"Execution:\n[Direct {_DIRECT_LLM_BACKEND} solver] {reasoning}\n"
            f"Answer given: {answer}"
        )
        return {
            "trajectory_text": trajectory,
            "sp_answer": answer,
            "sp_success": success,
            "sp_request_success": True,
            "sp_error": "",
            "sp_retry_count": retry_count,
            "memory_stack_log": [],
        }
    except Exception as exc:
        logger.warning("Direct %s solver failed: %s", _DIRECT_LLM_BACKEND, exc)
        empty_result["sp_error"] = f"{type(exc).__name__}: {exc}"
        empty_result["sp_retry_count"] = int(getattr(exc, "retry_count", 0))
        return empty_result


def _run_remote_deerflow(
    question: str,
    curated_experience: str,
    ground_truth_answer: str,
    timeout: int,
) -> dict:
    """Run one solver trajectory through the remote native DeerFlow worker."""
    import urllib.request

    endpoint = _next_remote_solver_url()
    request_timeout = max(
        30, int(os.getenv("REMOTE_DEERFLOW_TIMEOUT", str(timeout)))
    )
    body = {
        "question": question,
        "curated_experience": curated_experience,
        "profile": os.getenv("REMOTE_DEERFLOW_PROFILE", "stackplanner"),
        "max_iterations": int(os.getenv("REMOTE_DEERFLOW_MAX_ITERATIONS", "12")),
    }
    if os.getenv('SP2_HOP_VERIFY', '0') == '1':
        body['question'] = (
            'Solve this multi-hop question with StackPlanner2. Decompose every hop, '
            'verify each intermediate entity with web_search, perform at least two '
            'independent searches when needed, resolve conflicting names/dates, and '
            'do not stop after the first entity. Return only the final answer.\n\n'
            + question
        )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    empty_result = {
        "trajectory_text": "",
        "sp_answer": "",
        "sp_success": False,
        "sp_request_success": False,
        "sp_error": "",
        "sp_retry_count": 0,
        "memory_stack_log": [],
    }
    try:
        payload, retry_count = _post_json_with_retry(request, request_timeout)
        answer = _content_without_thinking(str(payload.get("answer") or ""))
        trajectory = str(payload.get("trajectory_text") or "").strip()
        # DeerFlow can return a human-readable provider outage as a normal
        # HTTP-200 answer.  Treat it as a failed solver request instead of
        # feeding the outage text into MemE or the evolving memory bank.
        provider_error = str(payload.get("error") or "")
        outage_markers = (
            "temporarily unavailable",
            "provider is unavailable",
            "configured llm provider",
            "multiple retries",
            "upstream unavailable",
        )
        if provider_error or any(marker in answer.lower() for marker in outage_markers):
            empty_result["sp_error"] = provider_error or answer[:500]
            empty_result["sp_retry_count"] = retry_count
            return empty_result
        if not answer:
            empty_result["sp_error"] = str(payload.get("error") or "empty_response")
            empty_result["sp_retry_count"] = retry_count
            return empty_result
        success = _check_answer(answer, ground_truth_answer) if ground_truth_answer else False
        if not trajectory:
            # Do not append correctness labels: this trajectory is consumed by
            # MemE and GT must remain evaluator-only during online evolution.
            trajectory = (
                f"Question: {question}\n"
                f"Execution:\n[Remote DeerFlow] {answer}\n"
                f"Answer given: {answer}"
            )
        return {
            "trajectory_text": trajectory,
            "sp_answer": answer,
            "sp_success": success,
            "sp_request_success": True,
            "sp_error": "",
            "sp_retry_count": retry_count,
            "memory_stack_log": payload.get("messages") or [],
            "deerflow_profile": payload.get("profile", body["profile"]),
            "search_call_count": int(payload.get("search_call_count") or 0),
        }
    except Exception as exc:
        logger.warning("Remote DeerFlow solver failed: %s", exc)
        empty_result["sp_error"] = f"{type(exc).__name__}: {exc}"
        empty_result["sp_retry_count"] = int(getattr(exc, "retry_count", 0))
        return empty_result


def run_direct_memory_evolution(
    trajectory: str,
    existing_experience: str,
    timeout: int = 15,
) -> str:
    """Synthesize one reusable memory card through Qwen or MiniMax."""
    import urllib.request

    prompt = (
        "Synthesize an improved reusable memory card from the execution trajectory. "
        "Keep correct transferable details from the existing experience, add lessons "
        "supported by the trajectory, and do not mention this prompt. Return only the "
        "card with these labels: Problem type:, Reusable insight:, Procedure:, "
        "Failure recovery:, Verification:.\n\n"
        f"Existing experience:\n{existing_experience or '(none)'}\n\n"
        f"Execution trajectory:\n{trajectory or '(empty)'}"
    )
    messages = [
            {"role": "system", "content": "You are a careful memory editor."},
            {"role": "user", "content": prompt},
        ]
    base_url, model, api_key, is_minimax = _direct_api_config()
    body = _api_body(model, messages, 500, is_minimax)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        payload, _ = _post_json_with_retry(request, timeout)
        message = payload["choices"][0]["message"]
        content = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )
        return _content_without_thinking(content)
    except Exception as exc:
        logger.warning("Direct %s memory evolution failed: %s", _DIRECT_LLM_BACKEND, exc)
        return ""


def _run_deerflow_fallback(
    question: str,
    curated_experience: str,
    ground_truth_answer: str,
    timeout: int,
) -> dict:
    """Generate a solver trajectory through the local DeerFlow API."""
    import argparse
    import uuid

    from eval_deerflow_retrieval_baselines import DeerFlowClient

    fallback_timeout = min(timeout, int(os.getenv("SP_FALLBACK_TIMEOUT", "90")))

    prompt = question
    if curated_experience and len(curated_experience.strip()) > 20:
        prompt = (
            f"Based on the following relevant experience:\n{curated_experience}\n\n"
            f"Please answer the question: {question}"
        )
    client_args = argparse.Namespace(
        deerflow_url="http://127.0.0.1:2026/api/langgraph",
        timeout=fallback_timeout,
        assistant_id="lead_agent",
        model_name="qwen3-32b",
        recursion_limit=30,
        thinking_enabled=False,
        subagent_enabled=False,
        auth_username="",
        auth_password="",
        auth_env_pid=int(os.getenv("DEERFLOW_AUTH_ENV_PID", "3909878")),
    )
    empty_result = {
        "trajectory_text": "",
        "sp_answer": "",
        "sp_success": False,
        "memory_stack_log": [],
    }
    try:
        client = DeerFlowClient(client_args)
        answer = client.run(
            prompt,
            task="offline-evo-fallback",
            thread_id=f"offline-evo-fallback-{uuid.uuid4().hex[:12]}",
        )
    except Exception as exc:
        logger.warning("DeerFlow fallback failed: %s", exc)
        return empty_result
    success = _check_answer(answer, ground_truth_answer) if ground_truth_answer else False
    trajectory = (
        f"Question: {question}\n"
        f"Execution:\n[DeerFlow fallback] {answer}\n"
        f"Answer given: {answer}"
    )
    return {
        "trajectory_text": trajectory,
        "sp_answer": answer,
        "sp_success": success,
        "memory_stack_log": [],
    }


def _check_answer(predicted: str, ground_truth: str) -> bool:
    """Compare answer payloads after structured extraction and normalization."""
    return answers_match(predicted, ground_truth)


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------

def run_sp_batch(
    samples: list[dict],
    max_workers: int = 4,
) -> list[dict]:
    """
    Run SP for a batch of samples in parallel threads.

    Each sample dict must have:
        question            : str
        curated_experience  : str
        ground_truth_answer : str (optional)

    Returns list of SP result dicts in same order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(samples)

    def _run_one(idx: int, s: dict) -> tuple[int, dict]:
        r = run_sp(
            question=s["question"],
            curated_experience=s.get("curated_experience", ""),
            ground_truth_answer=s.get("ground_truth_answer", ""),
        )
        return idx, r

    logger.info(f"Running SP for {len(samples)} samples (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_run_one, i, s): i for i, s in enumerate(samples)}
        for completed, fut in enumerate(as_completed(futures), start=1):
            idx, r = fut.result()
            results[idx] = r
            if completed % 10 == 0 or completed == len(samples):
                logger.info(f"  SP progress: {completed}/{len(samples)}")

    # A busy remote solver can time out a minority of requests even though the
    # service itself is healthy.  Do not checkpoint those transport failures as
    # wrong answers: retry only the failed rows at a deliberately lower
    # concurrency.  Successful first-pass trajectories are kept byte-for-byte.
    retry_rounds = max(0, int(os.getenv("SP_FAILED_RETRY_ROUNDS", "0")))
    retry_workers = max(1, int(os.getenv("SP_FAILED_RETRY_WORKERS", "4")))
    for retry_round in range(1, retry_rounds + 1):
        failed = [
            idx for idx, result in enumerate(results)
            if result is not None and result.get("sp_request_success") is False
        ]
        if not failed:
            break
        logger.info(
            "Retrying %s failed SP requests (round %s/%s, max_workers=%s)",
            len(failed), retry_round, retry_rounds, retry_workers,
        )
        recovered = 0
        with ThreadPoolExecutor(max_workers=min(retry_workers, len(failed))) as exe:
            futures = {
                exe.submit(_run_one, idx, samples[idx]): idx for idx in failed
            }
            for fut in as_completed(futures):
                idx, retried = fut.result()
                if retried.get("sp_request_success") is True:
                    results[idx] = retried
                    recovered += 1
                else:
                    # Preserve the latest diagnostic while retaining failure.
                    results[idx] = retried
        logger.info(
            "SP retry round %s recovered %s/%s requests",
            retry_round, recovered, len(failed),
        )

    return results
