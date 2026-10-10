"""
mem_agent_infer.py

Offline inference for MemRetriever (MemR) and MemEvolver (MemE).
Uses vLLM for fast batched generation with a manual tool-call loop.

Tool-call loop (per sample):
  1. Generate from prompt until </tool_call> or </answer> stop string.
  2. If stopped at </tool_call>: parse SID → call memory_lookup → inject <tool_response> → continue.
  3. If stopped at </answer>: extract answer text and predicted SID → done.
  4. Repeat up to max_turns.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from memory_manager import lookup_sid

logger = logging.getLogger(__name__)


def _append_tool_response(prompt: str, tool_result: str) -> str:
    """Append an observation using the Qwen/LlamaFactory tool role format."""
    return (
        prompt
        + "<|im_end|>\n<|im_start|>user\n"
        + f"<tool_response>\n{tool_result}\n</tool_response>"
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

_SID_RE = re.compile(
    r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# ---------------------------------------------------------------------------
# Prompt templates (copied from preprocess scripts)
# ---------------------------------------------------------------------------

MEMR_SYSTEM_PROMPT = (
    "You are MemRetriever, a memory retrieval agent. For every query, choose exactly "
    "one four-layer Semantic ID (SID) in the exact form "
    "<SID_L1_X><SID_L2_Y><SID_L3_Z><SID_L4_W>, then make exactly one memory_lookup "
    "call whose sid_list contains only that SID. You may reason briefly in "
    "<think>...</think>. After the tool response, output exactly one non-empty "
    "<answer>...</answer> containing a concise, actionable summary of the retrieved "
    "experience. Do not call the tool again, output multiple SIDs, or copy the entire "
    "memory. If no relevant memory is returned, output "
    "<answer>No relevant experience found.</answer>."
)

MEMORY_LOOKUP_TOOLS = [{
    "type": "function",
    "function": {
        "name": "memory_lookup",
        "description": (
            "Look up reusable experience memories using one four-layer "
            "Semantic ID (SID)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sid_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 1,
                    "description": (
                        "Exactly one SID such as "
                        "<SID_L1_5><SID_L2_3><SID_L3_2><SID_L4_1>. "
                        "Natural-language descriptions are invalid."
                    ),
                }
            },
            "required": ["sid_list"],
        },
    },
}]

# This is intentionally byte-for-byte aligned with the system message used to
# build the v39 RL training parquet (``scripts/build_meme_training_data.py``).
# Keeping the inference contract identical matters here: the old, looser
# prompt often made the checkpoint continue the embedded external task instead
# of emitting a four-layer SID/tool call, which was then counted as
# ``invalid_sid`` by the bootstrap gate.
MEME_SYSTEM_PROMPT = (
    "You are MemEvolver, a memory evolution agent. Given a failed execution "
    "trajectory, update the memory bank with evolved experience using this exact "
    "ReAct order. (1) In a first <think>...</think>, analyse the attempted task, "
    "its knowledge domain, what failed, and which SID cluster best captures the "
    "transferable lesson. (2) Select exactly one valid four-layer Semantic ID in "
    "the form <SID_L1_X><SID_L2_Y><SID_L3_Z><SID_L4_W>, then call memory_lookup "
    "exactly once with a sid_list containing only that SID. (3) After reading the "
    "tool response, use a second <think>...</think> to compare the failed trajectory "
    "with the retrieved experience: identify what is already captured, what is "
    "missing or contradicted, and how they should be merged. If no existing "
    "experience is found, derive the reusable lesson from the trajectory. (4) "
    "Output exactly one non-empty <answer>...</answer> containing the evolved "
    "experience. Preserve useful prior guidance, correct the failure, and add "
    "actionable decision rules and verification checks. Do not merely answer the "
    "original task, copy the old memory verbatim, emit multiple SIDs, or call the "
    "tool more than once."
)


# ---------------------------------------------------------------------------
# Core inference engine
# ---------------------------------------------------------------------------

class MemAgentInfer:
    """
    Runs MemR and MemE inference using vLLM.

    Usage:
        infer = MemAgentInfer(model_path, bank, index)
        memr_results = infer.run_memr_batch(questions)
        meme_results = infer.run_meme_batch(trajectories)
        infer.unload()
    """

    def __init__(
        self,
        model_path: str,
        bank: list[dict],
        index: dict,
        top_k: int = 3,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 4096,
        max_turns: int = 5,
        max_new_tokens: int = 1024,
        memr_sid_beams: int = 1,
        memr_beam_fusion: str = "merge_then_summarize",
    ):
        self.bank = bank
        self.index = index
        self.top_k = top_k
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.memr_sid_beams = max(1, int(memr_sid_beams))
        self.memr_beam_fusion = memr_beam_fusion

        logger.info(f"Loading model from {model_path} ...")
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("Model loaded.")

    def unload(self):
        """Release GPU memory."""
        try:
            import gc
            import torch
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Single-sample inference with tool loop
    # ------------------------------------------------------------------

    def _run_one(self, system_prompt: str, user_content: str) -> dict:
        """
        Run one MemR or MemE sample through the tool-call loop.

        Returns:
            {
                "answer": str,           # content inside <answer>...</answer>
                "sid_l1": int | None,    # predicted SID codes
                "sid_l2": int | None,
                "sid_l3": int | None,
                "sid_l4": int | None,
                "full_text": str,        # entire accumulated generation
            }
        """
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Build initial prompt text using chat template
        full_text: str = self.tokenizer.apply_chat_template(
            messages,
            tools=MEMORY_LOOKUP_TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            stop=["</tool_call>", "</answer>"],
            include_stop_str_in_output=True,
        )

        predicted_sid = None

        for turn in range(self.max_turns):
            outputs = self.llm.generate([full_text], sp, use_tqdm=False)
            generated: str = outputs[0].outputs[0].text
            full_text += generated

            # Check for </answer> stop
            if "</answer>" in generated:
                break

            # Check for </tool_call> stop
            if "</tool_call>" in generated:
                # Extract SID from the tool_call block
                sid_m = _SID_RE.search(generated)
                if sid_m and predicted_sid is None:
                    predicted_sid = tuple(map(int, sid_m.groups()))

                # Extract sid_list from tool_call JSON
                sid_list = _parse_sid_list(generated)
                if not sid_list:
                    # Can't execute tool, stop
                    break

                # Execute memory lookup
                tool_result_parts = []
                for sid_str in sid_list:
                    result = lookup_sid(
                        sid_str,
                        self.bank,
                        self.index,
                        self.top_k,
                        allow_fallback=system_prompt == MEMR_SYSTEM_PROMPT,
                    )
                    tool_result_parts.append(result)
                tool_result = "\n\n".join(tool_result_parts)

                # Inject tool_response and continue
                full_text = _append_tool_response(full_text, tool_result)
            else:
                # No boundary hit — model stopped generating (EOS or max_tokens)
                break

        # Extract answer
        answer = ""
        ans_matches = list(_ANSWER_RE.finditer(full_text))
        if ans_matches:
            answer = ans_matches[-1].group(1).strip()

        # Extract first predicted SID from full_text (if not captured above)
        if predicted_sid is None:
            for tc_m in _TOOL_CALL_RE.finditer(full_text):
                sid_m = _SID_RE.search(tc_m.group(1))
                if sid_m:
                    predicted_sid = tuple(map(int, sid_m.groups()))
                    break

        l1 = predicted_sid[0] if predicted_sid else None
        l2 = predicted_sid[1] if predicted_sid else None
        l3 = predicted_sid[2] if predicted_sid else None
        l4 = predicted_sid[3] if predicted_sid else None

        return {
            "answer": answer,
            "sid_l1": l1,
            "sid_l2": l2,
            "sid_l3": l3,
            "sid_l4": l4,
            "full_text": full_text,
        }

    # ------------------------------------------------------------------
    # Batch interfaces
    # ------------------------------------------------------------------

    def run_memr_batch(self, questions: list[str]) -> list[dict]:
        """
        Run MemR inference for a list of questions.
        Each question → {answer, sid_l1, sid_l2, sid_l3, sid_l4, full_text}
        """
        logger.info(f"MemR inference: {len(questions)} questions")
        results = []
        for i, q in enumerate(questions):
            user_content = f"<query> {q} </query>"
            r = self._run_one(MEMR_SYSTEM_PROMPT, user_content)
            results.append(r)
            if (i + 1) % 20 == 0 or (i + 1) == len(questions):
                logger.info(f"  MemR progress: {i+1}/{len(questions)}")
        return results

    def run_meme_batch(self, trajectory_texts: list[str]) -> list[dict]:
        """
        Run MemE inference for a list of trajectory texts.
        Each trajectory → {answer, sid_l1, sid_l2, sid_l3, sid_l4, full_text}
        """
        logger.info(f"MemE inference: {len(trajectory_texts)} trajectories")
        results = []
        for i, traj in enumerate(trajectory_texts):
            user_content = f"<trajectory> {traj} </trajectory>"
            r = self._run_one(MEME_SYSTEM_PROMPT, user_content)
            results.append(r)
            if (i + 1) % 20 == 0 or (i + 1) == len(trajectory_texts):
                logger.info(f"  MemE progress: {i+1}/{len(trajectory_texts)}")
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sid_list(tool_call_text: str) -> list[str]:
    """Extract sid_list from <tool_call>...</tool_call> content."""
    # Find tool_call content
    tc_m = _TOOL_CALL_RE.search(tool_call_text)
    if not tc_m:
        # Try partial: text might be just the content after <tool_call>
        content = tool_call_text
    else:
        content = tc_m.group(1)

    try:
        obj = json.loads(content.strip())
        args = obj.get("arguments", {})
        sid_list = args.get("sid_list", [])
        if isinstance(sid_list, str):
            sid_list = [sid_list]
        return [s for s in sid_list if isinstance(s, str)]
    except (json.JSONDecodeError, AttributeError):
        # Fallback: extract SID tokens directly
        return [match.group(0) for match in _SID_RE.finditer(tool_call_text)]
