"""Task-aware DeerFlow outcome backend with a persistent baseline cache.

Only final task success is scored.  The no-memory result is persisted in a
local SQLite database so a query keeps the same baseline across workers and
training restarts.  Credentials remain in environment variables and are never
written to traces.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import requests


_DEERFLOW_DEFAULT_BASE_URL = "http://127.0.0.1:2026/api/langgraph"
_DEERFLOW_DEFAULT_MODEL = "qwen3-32b"
_DEERFLOW_CONNECT_TIMEOUT = 10
_DEERFLOW_READ_TIMEOUT = 600
_DEERFLOW_MAX_RETRIES = 2
_DEERFLOW_TRACE_MAX_CHARS = 1_000_000
_BASELINE_LEASE_SECONDS = 60.0
_BASELINE_POLL_SECONDS = 0.25
_BASELINE_DB_RETRY_SECONDS = 60.0
_ALFWORLD_FILENAME_RE = re.compile(r"File:\s*([^\s]+\.tw-pddl)", re.IGNORECASE)

_SESSION_LOCK = threading.Lock()
_SESSION: requests.Session | None = None
_SESSION_IDENTITY: tuple[str, str, str] | None = None
_BASELINE_LOCK = threading.Lock()
_BASELINE_MEMORY: dict[str, tuple[float, str, str, bool]] = {}
_BASELINE_INFLIGHT: dict[str, threading.Event] = {}
_ALFWORLD_INDEX_LOCK = threading.Lock()
_ALFWORLD_INDEX: dict[str, str] | None = None


def _extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = (text or "").rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def _canonical_target_answer(data_source: str, target_answer):
    if not isinstance(target_answer, str):
        return target_answer, "identity"
    source = (data_source or "").lower()
    if "gsm8k" in source:
        matches = re.findall(r"####\s*([^\r\n]+)", target_answer)
        if matches:
            return matches[-1].strip(), "gsm8k_final_after_hashes"
    if "math" in source:
        boxed = _extract_last_boxed(target_answer)
        if boxed:
            return boxed, "math_last_boxed"
    return target_answer, "identity"


def _normalise_decimal(text: str) -> Decimal | None:
    candidate = str(text or "").strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def _decimal_candidates(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for numerator, denominator in re.findall(
        r"\\(?:d|t)?frac\s*\{\s*(-?[\d,.]+)\s*\}\s*\{\s*(-?[\d,.]+)\s*\}",
        text or "",
    ):
        left, right = _normalise_decimal(numerator), _normalise_decimal(denominator)
        if left is not None and right not in (None, Decimal(0)):
            values.append(left / right)
    for number in re.findall(r"-?(?:\d[\d,]*)(?:\.\d+)?", text or ""):
        value = _normalise_decimal(number)
        if value is not None:
            values.append(value)
    return values


def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?(?:\d[\d,]*)(?:\.\d+)?", (text or "")[-800:])
    return matches[-1] if matches else None


def _normalise_latex(text: str) -> str:
    value = str(text or "").strip().lower()
    for token in ("$", r"\left", r"\right", r"\,", r"\!", r"\;", r"\quad"):
        value = value.replace(token, "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return re.sub(r"\s+", "", value).rstrip(".,;:")


def _extract_final_math_answer(answer: str) -> str:
    boxed = _extract_last_boxed(answer or "")
    if boxed:
        return boxed
    labelled = re.findall(
        r"(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)", answer or "", re.IGNORECASE
    )
    if labelled:
        return labelled[-1].strip()
    bold = re.findall(r"\*\*([^*\n]+)\*\*", answer or "")
    if bold:
        return bold[-1].strip()
    return (answer or "").strip()


def _score_canonical_answer(data_source: str, answer: str, target_answer) -> float | None:
    source = (data_source or "").lower()
    if "gsm8k" in source:
        predicted = _extract_last_number(answer)
        target = _normalise_decimal(str(target_answer))
        return float(
            predicted is not None
            and target is not None
            and _normalise_decimal(predicted) == target
        )
    if "math" in source:
        final_answer = _extract_final_math_answer(answer)
        target_number = _normalise_decimal(str(target_answer))
        if target_number is not None:
            return float(target_number in _decimal_candidates(final_answer))
        return float(
            bool(final_answer)
            and _normalise_latex(final_answer) == _normalise_latex(str(target_answer))
        )
    return None


def _deerflow_endpoints() -> tuple[str, str]:
    raw = os.getenv("DEERFLOW_BASE_URL", _DEERFLOW_DEFAULT_BASE_URL).strip().rstrip("/")
    if not raw:
        raise ValueError("DEERFLOW_BASE_URL is empty")
    suffix = "/api/langgraph"
    return (raw[: -len(suffix)], raw) if raw.endswith(suffix) else (raw, f"{raw}{suffix}")


def _credentials() -> tuple[str, str]:
    email = os.getenv("DEERFLOW_EMAIL", "").strip()
    password = os.getenv("DEERFLOW_PASSWORD", "")
    if not email or not password:
        raise RuntimeError("DEERFLOW_EMAIL and DEERFLOW_PASSWORD are required")
    return email, password


def _login(session: requests.Session, origin: str, email: str, password: str) -> None:
    response = session.post(
        f"{origin}/api/v1/auth/login/local",
        data={"username": email, "password": password},
        timeout=(_DEERFLOW_CONNECT_TIMEOUT, 30),
    )
    response.raise_for_status()
    csrf = session.cookies.get("csrf_token") or session.cookies.get("csrf-token")
    if not csrf:
        raise RuntimeError("DeerFlow login succeeded without a CSRF cookie")
    session.headers.update({"X-CSRF-Token": csrf})


def _get_session() -> tuple[requests.Session, str]:
    global _SESSION, _SESSION_IDENTITY
    origin, langgraph_base = _deerflow_endpoints()
    email, password = _credentials()
    identity = (origin, email, password)
    if _SESSION is None or _SESSION_IDENTITY != identity:
        session = requests.Session()
        session.trust_env = False
        _login(session, origin, email, password)
        _SESSION, _SESSION_IDENTITY = session, identity
    return _SESSION, langgraph_base


def _reset_session() -> None:
    global _SESSION, _SESSION_IDENTITY
    if _SESSION is not None:
        _SESSION.close()
    _SESSION = None
    _SESSION_IDENTITY = None


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    pieces.append(str(value))
        return "\n".join(pieces).strip()
    return str(content or "").strip()


def _serialise_trace(payload: dict, thread_id: str, request_body: dict) -> str:
    envelope = {
        "trace_schema": "deerflow_langgraph_request_response_v1",
        "thread_id": thread_id,
        "request": request_body,
        "response": payload,
        "trace_truncated": False,
    }
    raw = json.dumps(envelope, ensure_ascii=False, default=str)
    max_chars = max(
        10_000,
        int(os.getenv("DEERFLOW_TRACE_MAX_CHARS", str(_DEERFLOW_TRACE_MAX_CHARS))),
    )
    if len(raw) <= max_chars:
        return raw
    head = max_chars * 3 // 4
    return json.dumps(
        {
            "trace_schema": "deerflow_langgraph_request_response_v1",
            "thread_id": thread_id,
            "trace_truncated": True,
            "original_chars": len(raw),
            "raw_head": raw[:head],
            "raw_tail": raw[-(max_chars - head):],
        },
        ensure_ascii=False,
    )


def _trace_for_rollout(raw_trace: str):
    if not raw_trace:
        return None
    try:
        return json.loads(raw_trace)
    except (json.JSONDecodeError, TypeError):
        return {"trace_schema": "unparsed_text_v1", "raw": str(raw_trace)}


def _extract_deerflow_answer(payload: dict, thread_id: str, request_body: dict) -> tuple[str, str]:
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        raise ValueError("DeerFlow response has no messages list")
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        kind = str(message.get("type") or message.get("role") or "").lower()
        if kind not in {"ai", "assistant", "aimessage"}:
            continue
        answer = _message_text(message.get("content"))
        if answer:
            return answer, _serialise_trace(payload, thread_id, request_body)
    raise ValueError("DeerFlow response has no non-empty AI answer")


def _alfworld_root() -> Path:
    configured = os.getenv("ALFWORLD_RAW_ROOT", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "assets" / "alfworld_raw"


def _alfworld_file_index() -> dict[str, str]:
    global _ALFWORLD_INDEX
    with _ALFWORLD_INDEX_LOCK:
        if _ALFWORLD_INDEX is None:
            index: dict[str, str] = {}
            for split in ("train", "eval_in_distribution", "eval_out_of_distribution"):
                split_dir = _alfworld_root() / split
                if not split_dir.is_dir():
                    continue
                for path in split_dir.iterdir():
                    if path.is_file() and path.name.endswith(".tw-pddl"):
                        index.setdefault(path.name, str(path))
            _ALFWORLD_INDEX = index
        return _ALFWORLD_INDEX


def _resolve_alfworld_task_path(query: str) -> str:
    match = _ALFWORLD_FILENAME_RE.search(query or "")
    if not match:
        raise ValueError("ALFWorld query has no .tw-pddl filename")
    filename = Path(match.group(1)).name
    path = _alfworld_file_index().get(filename)
    if not path:
        raise FileNotFoundError(f"ALFWorld task not found under local raw root: {filename}")
    return path


def _json_tool_payload(message: dict) -> dict | None:
    kind = str(message.get("type") or message.get("role") or "").lower()
    if kind not in {"tool", "toolmessage"}:
        return None
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _score_alfworld_execution(payload: dict) -> tuple[float, bool, str]:
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return 0.0, False, "missing LangGraph messages"
    reset_seen = False
    step_outputs = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        name = str(message.get("name") or "").lower()
        tool_payload = _json_tool_payload(message)
        if tool_payload is None:
            continue
        if name == "alfworld_reset":
            if tool_payload.get("error"):
                return 0.0, False, "reset error"
            reset_seen = bool(tool_payload.get("reset"))
            step_outputs = 0
        elif name == "alfworld_step" and reset_seen:
            if tool_payload.get("error"):
                return 0.0, False, "step error"
            step_outputs += 1
            if tool_payload.get("won") is True:
                return 1.0, True, f"environment won after {step_outputs} steps"
    if not reset_seen:
        return 0.0, True, "agent never called alfworld_reset"
    return 0.0, True, f"environment not won after {step_outputs} observed steps"


def _deerflow_request_body(
    thread_id: str,
    experience: str | None,
    query: str,
    data_source: str = "",
    alfworld_task_file: str = "",
    alfworld_split: str = "auto",
) -> dict:
    source = (data_source or "").lower()
    memory_section = (
        f"Retrieved reusable memory:\n{experience}\n\n"
        if experience
        else "No retrieved memory is provided.\n\n"
    )
    if "webshop" in source:
        prompt = (
            "Solve this WebShop product-identification task. Identify the single product "
            "that satisfies every requested attribute. Return only its 10-character ASIN.\n\n"
            f"{memory_section}WebShop request:\n{query}"
        )
    elif "alfworld" in source:
        task_request = query.split("\n\nEnvironment/task excerpt:", 1)[0]
        prompt = (
            "Solve this ALFWorld task using the real TextWorld tools. First call "
            "alfworld_reset with exactly the task_file and split below, then call "
            "alfworld_step using commands from admissible_commands until won=true or "
            "30 steps. A printed plan is not success.\n\n"
            f"{memory_section}Task request:\n{task_request}\n\n"
            f"task_file: {alfworld_task_file}\nsplit: {alfworld_split}"
        )
    elif experience:
        prompt = (
            "Use the retrieved reusable strategy and verification guidance while solving "
            "the question. Do not assume it contains the final answer. Return only the "
            f"final answer.\n\nRetrieved memory:\n{experience}\n\nQuestion:\n{query}"
        )
    else:
        prompt = (
            "Solve the question using your normal capabilities. No retrieved memory is "
            f"provided. Return only the final answer.\n\nQuestion:\n{query}"
        )
    return {
        "assistant_id": "lead_agent",
        "input": {"messages": [{"type": "human", "content": prompt}]},
        "config": {"configurable": {"thread_id": thread_id}},
        "context": {
            "model_name": os.getenv("DEERFLOW_MODEL", _DEERFLOW_DEFAULT_MODEL),
            "mode": os.getenv("DEERFLOW_MODE", "flash"),
            "thinking_enabled": "alfworld" in source,
            "is_plan_mode": False,
            "subagent_enabled": False,
            "memory_enabled": False,
            "title_enabled": False,
            "harbor_environment": "alfworld" if "alfworld" in source else None,
        },
    }


def _delete_thread(thread_id: str) -> None:
    try:
        with _SESSION_LOCK:
            session, langgraph = _get_session()
            response = session.delete(
                f"{langgraph}/threads/{thread_id}",
                timeout=(_DEERFLOW_CONNECT_TIMEOUT, 30),
            )
            if response.status_code == 401:
                _reset_session()
                session, langgraph = _get_session()
                response = session.delete(
                    f"{langgraph}/threads/{thread_id}",
                    timeout=(_DEERFLOW_CONNECT_TIMEOUT, 30),
                )
            response.raise_for_status()
    except Exception as exc:
        print(f"[v33 DeerFlow] thread cleanup failed: {exc}")


def _generic_em(answer: str, target_answer) -> float:
    try:
        from verl.utils.reward_score.search_r1_like_qa_em import compute_score, subem_check

        wrapped = f"<answer>{answer}</answer>"
        score = float(compute_score(wrapped, {"target": target_answer}, method="flexible"))
        if score == 0.0 and answer and subem_check(answer, target_answer):
            score = 1.0
        return score
    except ImportError:
        normalise = lambda value: re.sub(r"\W+", "", str(value or "").lower())
        return float(bool(answer) and normalise(answer) == normalise(target_answer))


def _get_outcome_score_via_deerflow(
    experience: str,
    query: str,
    ground_truth: dict,
    data_source: str = "",
) -> tuple[float, str, str, bool]:
    source = (data_source or "").lower()
    is_alfworld = "alfworld" in source
    alfworld_file = ""
    alfworld_split = "auto"
    if is_alfworld:
        try:
            path = Path(_resolve_alfworld_task_path(query))
            alfworld_file = path.name
            alfworld_split = path.parent.name
        except Exception as exc:
            print(f"[v33 DeerFlow] ALFWorld task resolution failed: {exc}")
            return 0.0, "", "", False

    raw_target = ground_truth.get("target_answer") if isinstance(ground_truth, dict) else None
    target, normalizer = _canonical_target_answer(data_source, raw_target)
    if not is_alfworld and not target:
        print("[v33 DeerFlow] skipped because target_answer is empty")
        return 0.0, "", "", False

    read_timeout = int(os.getenv("DEERFLOW_READ_TIMEOUT", str(_DEERFLOW_READ_TIMEOUT)))
    max_retries = int(os.getenv("DEERFLOW_MAX_RETRIES", str(_DEERFLOW_MAX_RETRIES)))
    for attempt in range(max_retries):
        thread_id = f"memr-v33-{uuid.uuid4().hex}"
        try:
            body = _deerflow_request_body(
                thread_id,
                experience,
                query,
                data_source=data_source,
                alfworld_task_file=alfworld_file,
                alfworld_split=alfworld_split,
            )
            started = time.monotonic()
            with _SESSION_LOCK:
                session, langgraph = _get_session()
                response = session.post(
                    f"{langgraph}/runs/wait",
                    json=body,
                    timeout=(_DEERFLOW_CONNECT_TIMEOUT, read_timeout),
                )
                if response.status_code == 401:
                    _reset_session()
                    session, langgraph = _get_session()
                    response = session.post(
                        f"{langgraph}/runs/wait",
                        json=body,
                        timeout=(_DEERFLOW_CONNECT_TIMEOUT, read_timeout),
                    )
                response.raise_for_status()
                payload = response.json()

            if is_alfworld:
                raw_trace = _serialise_trace(payload, thread_id, body)
                try:
                    answer, _ = _extract_deerflow_answer(payload, thread_id, body)
                except ValueError:
                    answer = ""
                score, execution_ok, reason = _score_alfworld_execution(payload)
                if not execution_ok:
                    raise RuntimeError(reason)
            else:
                answer, raw_trace = _extract_deerflow_answer(payload, thread_id, body)
                canonical = _score_canonical_answer(data_source, answer, target)
                score = canonical if canonical is not None else _generic_em(answer, target)
            print(
                f"[v33 DeerFlow] attempt={attempt + 1} "
                f"elapsed={time.monotonic() - started:.1f}s score={score} "
                f"normalizer={normalizer}"
            )
            return float(score), answer, raw_trace, True
        except Exception as exc:
            print(f"[v33 DeerFlow] attempt {attempt + 1}/{max_retries} failed: {exc}")
            if attempt + 1 == max_retries:
                return 0.0, "", "", False
            time.sleep(min(2**attempt, 5))
        finally:
            _delete_thread(thread_id)
    return 0.0, "", "", False


def _baseline_key(query: str, ground_truth: dict, data_source: str) -> str:
    target = ground_truth.get("target_answer") if isinstance(ground_truth, dict) else None
    payload = {
        "data_source": data_source,
        "query": query,
        "target": target,
        "model": os.getenv("DEERFLOW_MODEL", _DEERFLOW_DEFAULT_MODEL),
        "mode": os.getenv("DEERFLOW_MODE", "flash"),
    }
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _cache_path() -> Path:
    configured = os.getenv("BASELINE_CACHE_DB", "").strip()
    return Path(configured) if configured else Path(__file__).parents[1] / "cache" / "baseline.sqlite3"


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _db_connect() -> sqlite3.Connection:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _BASELINE_DB_RETRY_SECONDS
    while True:
        remaining = max(0.1, deadline - time.monotonic())
        connection = sqlite3.connect(path, timeout=min(5.0, remaining))
        connection.execute(f"PRAGMA busy_timeout={int(min(5.0, remaining) * 1000)}")
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS baselines (
                cache_key TEXT PRIMARY KEY,
                score REAL NOT NULL,
                answer TEXT NOT NULL,
                trace TEXT NOT NULL,
                healthy INTEGER NOT NULL,
                created_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS baseline_claims (
                cache_key TEXT PRIMARY KEY,
                owner_token TEXT NOT NULL,
                lease_until REAL NOT NULL,
                updated_at REAL NOT NULL
                )"""
            )
            return connection
        except sqlite3.OperationalError as exc:
            connection.close()
            if (
                "locked" not in str(exc).lower()
                and "busy" not in str(exc).lower()
            ) or time.monotonic() >= deadline:
                raise
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _read_disk_baseline(key: str) -> tuple[float, str, str, bool] | None:
    connection = _db_connect()
    try:
        row = connection.execute(
            """SELECT score, answer, trace, healthy FROM baselines
            WHERE cache_key = ? AND healthy = 1""",
            (key,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return float(row[0]), str(row[1]), str(row[2]), bool(row[3])


def _persist_baseline(key: str, result: tuple[float, str, str, bool]):
    score, answer, trace, healthy = result
    if not healthy:
        return result
    connection = _db_connect()
    try:
        with connection:
            connection.execute(
                """INSERT INTO baselines
                (cache_key, score, answer, trace, healthy, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    score = excluded.score,
                    answer = excluded.answer,
                    trace = excluded.trace,
                    healthy = 1,
                    created_at = excluded.created_at
                WHERE baselines.healthy = 0""",
                (key, float(score), str(answer), str(trace), time.time()),
            )
    finally:
        connection.close()
    return _read_disk_baseline(key) or result


def _try_claim_baseline(key: str, owner_token: str, lease_seconds: float) -> bool:
    now = time.time()
    connection = _db_connect()
    try:
        with connection:
            cursor = connection.execute(
                """INSERT INTO baseline_claims
                (cache_key, owner_token, lease_until, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    lease_until = excluded.lease_until,
                    updated_at = excluded.updated_at
                WHERE baseline_claims.lease_until <= excluded.updated_at""",
                (key, owner_token, now + lease_seconds, now),
            )
        return cursor.rowcount == 1
    finally:
        connection.close()


def _renew_baseline_claim(key: str, owner_token: str, lease_seconds: float) -> bool:
    now = time.time()
    connection = _db_connect()
    try:
        with connection:
            cursor = connection.execute(
                """UPDATE baseline_claims
                SET lease_until = ?, updated_at = ?
                WHERE cache_key = ? AND owner_token = ?""",
                (now + lease_seconds, now, key, owner_token),
            )
        return cursor.rowcount == 1
    finally:
        connection.close()


def _release_baseline_claim(key: str, owner_token: str) -> None:
    connection = _db_connect()
    try:
        with connection:
            connection.execute(
                "DELETE FROM baseline_claims WHERE cache_key = ? AND owner_token = ?",
                (key, owner_token),
            )
    finally:
        connection.close()


def _heartbeat_baseline_claim(
    key: str,
    owner_token: str,
    lease_seconds: float,
    stop_event: threading.Event,
) -> None:
    interval = max(0.01, min(10.0, lease_seconds / 3.0))
    while not stop_event.wait(interval):
        try:
            if not _renew_baseline_claim(key, owner_token, lease_seconds):
                return
        except sqlite3.Error as exc:
            print(f"[v33 baseline cache] lease heartbeat failed: {exc}")


def _get_disk_or_compute_baseline(
    key: str,
    query: str,
    ground_truth: dict,
    data_source: str,
) -> tuple[float, str, str, bool]:
    lease_seconds = _positive_float_env(
        "BASELINE_LEASE_SECONDS", _BASELINE_LEASE_SECONDS
    )
    poll_seconds = _positive_float_env(
        "BASELINE_POLL_SECONDS", _BASELINE_POLL_SECONDS
    )
    owner_token = f"{os.getpid()}-{uuid.uuid4().hex}"

    while True:
        disk = _read_disk_baseline(key)
        if disk is not None:
            return disk
        if _try_claim_baseline(key, owner_token, lease_seconds):
            break
        time.sleep(poll_seconds)

    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_baseline_claim,
        args=(key, owner_token, lease_seconds, heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    try:
        # Close the race where a healthy result was committed between our last
        # read and acquisition of a newly released/expired claim.
        disk = _read_disk_baseline(key)
        if disk is not None:
            return disk
        computed = _get_outcome_score_via_deerflow(
            "", query, ground_truth, data_source=data_source
        )
        return _persist_baseline(key, computed) if computed[3] else computed
    finally:
        heartbeat_stop.set()
        _release_baseline_claim(key, owner_token)
        heartbeat.join(timeout=1.0)


def _get_cached_baseline_score(
    query: str,
    ground_truth: dict,
    data_source: str = "",
) -> tuple[float, str, str, bool]:
    key = _baseline_key(query, ground_truth, data_source)
    # Training workers must not block on a second external DeerFlow request.
    # A cache-only mode lets a prefilled baseline DB provide counterfactual
    # supervision while treating misses as unavailable for this rollout.
    if os.getenv("BASELINE_CACHE_ONLY", "0").lower() in {"1", "true", "yes"}:
        return _read_disk_baseline(key) or (0.0, "", "", False)
    with _BASELINE_LOCK:
        cached = _BASELINE_MEMORY.get(key)
        if cached is not None:
            return cached
        event = _BASELINE_INFLIGHT.get(key)
        owner = event is None
        if owner:
            event = threading.Event()
            _BASELINE_INFLIGHT[key] = event
    assert event is not None
    if not owner:
        event.wait(
            timeout=int(os.getenv("DEERFLOW_READ_TIMEOUT", str(_DEERFLOW_READ_TIMEOUT)))
            * int(os.getenv("DEERFLOW_MAX_RETRIES", str(_DEERFLOW_MAX_RETRIES)))
            + 60
        )
        with _BASELINE_LOCK:
            return _BASELINE_MEMORY.get(key, (0.0, "", "", False))

    result = (0.0, "", "", False)
    try:
        result = _get_disk_or_compute_baseline(
            key, query, ground_truth, data_source
        )
        if result[3]:
            with _BASELINE_LOCK:
                _BASELINE_MEMORY[key] = result
        return result
    finally:
        with _BASELINE_LOCK:
            _BASELINE_INFLIGHT.pop(key, None)
            event.set()
