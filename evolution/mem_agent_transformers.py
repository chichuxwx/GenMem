"""Transformers backend for MemR/MemE inference.

Used when the local vLLM environment cannot load the Qwen3 architecture.
The public methods intentionally match ``MemAgentInfer``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from mem_agent_infer import (
    MEMORY_LOOKUP_TOOLS,
    MEME_SYSTEM_PROMPT,
    MEMR_SYSTEM_PROMPT,
    _ANSWER_RE,
    _SID_RE,
    _TOOL_CALL_RE,
    _parse_sid_list,
)
from memory_manager import lookup_sid

logger = logging.getLogger(__name__)


def make_sid_prefix_allowed_tokens(
    prompt_width: int,
    sid_token_levels: list[list[int]],
):
    """Restrict each generated position to the matching SID codebook level."""
    def allowed_tokens(_batch_id, input_ids):
        level = len(input_ids) - prompt_width
        if not 0 <= level < len(sid_token_levels):
            raise ValueError(f"SID constraint called at invalid level: {level}")
        return sid_token_levels[level]

    return allowed_tokens


def validate_sid_beam_token_ids(
    sequences,
    sid_token_levels: list[set[int]],
    expected_count: int,
) -> list[list[int]]:
    """Require legal, unique, four-token SID beam paths."""
    validated = [[int(token_id) for token_id in row] for row in sequences]
    if len(validated) != expected_count:
        raise ValueError(
            f"expected {expected_count} SID beams, received {len(validated)}"
        )
    for row in validated:
        if len(row) != len(sid_token_levels) or any(
            token_id not in sid_token_levels[level]
            for level, token_id in enumerate(row)
        ):
            raise ValueError(f"invalid SID beam token path: {row}")
    if len({tuple(row) for row in validated}) != expected_count:
        raise ValueError(f"expected {expected_count} unique SID beams")
    return validated


def build_sid_token_levels(
    tokenizer,
    level_sizes: tuple[int, ...] = (48, 16, 8, 8),
) -> list[list[int]]:
    """Resolve every symbolic SID label to one unique tokenizer token ID."""
    levels: list[list[int]] = []
    flattened: list[int] = []
    for level, size in enumerate(level_sizes, start=1):
        token_ids = []
        for code in range(size):
            label = f"<SID_L{level}_{code}>"
            encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(f"SID label is not one atomic token: {label}")
            token_ids.append(int(encoded[0]))
        levels.append(token_ids)
        flattened.extend(token_ids)
    if len(flattened) != len(set(flattened)):
        raise ValueError("SID labels do not map to unique token IDs")
    return levels


def split_before_first_sid_token(
    generated_token_ids: list[int],
    l1_token_ids: set[int],
) -> tuple[list[int], int | None]:
    """Keep one shared reasoning/tool prefix and remove its first L1 SID token."""
    for index, token_id in enumerate(generated_token_ids):
        if token_id in l1_token_ids:
            return generated_token_ids[:index], token_id
    return generated_token_ids, None


def _append_tool_response(prompt: str, tool_result: str) -> str:
    """Append an observation using the exact Qwen/LlamaFactory role boundaries."""
    return (
        prompt
        + "<|im_end|>\n<|im_start|>user\n"
        + f"<tool_response>\n{tool_result}\n</tool_response>"
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


class _BoundaryStoppingCriteria(StoppingCriteria):
    """Stop generation once every row reaches a MemR/MemE boundary marker.

    ``generate(max_new_tokens=8192)`` otherwise waits for the hard cap even
    after the model has emitted ``</tool_call>`` or ``</answer>``.  That made
    external bootstrap inference appear stalled and wasted most of the GPU
    time.  Matching token IDs is faster and safer than decoding every step.
    """

    def __init__(self, marker_ids: list[list[int]]):
        super().__init__()
        self.marker_ids = [tuple(ids) for ids in marker_ids if ids]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        del scores, kwargs
        if not self.marker_ids:
            return False
        for row in input_ids:
            row_ids = row.tolist()
            if not any(len(row_ids) >= len(marker)
                       and tuple(row_ids[-len(marker):]) == marker
                       for marker in self.marker_ids):
                return False
        return True


class MemAgentTransformersInfer:
    """Drop-in Transformers implementation of the MemAgent inference loop."""

    def __init__(
        self,
        model_path: str,
        bank: list[dict],
        index: dict,
        top_k: int = 3,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 4096,
        max_turns: int = 5,
        max_new_tokens: int = 1024,
        memr_sid_beams: int = 1,
        memr_beam_fusion: str = "merge_then_summarize",
        memr_reranker_path: str | None = None,
        memr_rerank_top_k: int = 0,
        memr_reranker_batch_size: int = 32,
        memr_reranker_max_length: int = 512,
    ):
        del tensor_parallel_size, gpu_memory_utilization, max_model_len
        self.bank = bank
        self.index = index
        self.top_k = top_k
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        # MemR's first pass only needs to reach the first SID token.  A
        # separate cap prevents a malformed/overthinking sample from
        # consuming the full answer budget before the structured SID beam
        # search starts.  The main answer budget remains ``max_new_tokens``.
        think_cap = int(os.getenv("MEMR_THINK_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.memr_think_max_new_tokens = max(64, min(self.max_new_tokens, think_cap))
        fusion_cap = int(os.getenv("MEMR_FUSION_MAX_NEW_TOKENS", "256"))
        self.memr_fusion_max_new_tokens = max(64, min(self.max_new_tokens, fusion_cap))
        self.memr_sid_beams = max(1, int(memr_sid_beams))
        self.memr_rerank_top_k = int(memr_rerank_top_k)
        if bool(memr_reranker_path) != (self.memr_rerank_top_k > 0):
            raise ValueError(
                "memr_reranker_path and a positive memr_rerank_top_k must be set together"
            )
        if self.memr_rerank_top_k > self.memr_sid_beams:
            raise ValueError("memr_rerank_top_k cannot exceed memr_sid_beams")
        if memr_beam_fusion not in {"merge_then_summarize", "summarize_then_merge"}:
            raise ValueError(f"Unsupported MemR beam fusion strategy: {memr_beam_fusion}")
        self.memr_beam_fusion = memr_beam_fusion

        logger.info("Loading Transformers model from %s ...", model_path)
        # The exported LlamaFactory tokenizer stores this field as a list,
        # while newer Transformers expects a mapping.  The SID tokens are
        # already present in tokenizer.json, so an empty mapping preserves
        # them and avoids the incompatible config parser path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            extra_special_tokens={},
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self._boundary_stopping = StoppingCriteriaList([
            _BoundaryStoppingCriteria([
                self.tokenizer(marker, add_special_tokens=False)["input_ids"]
                for marker in ("</tool_call>", "</answer>")
            ])
        ])
        self._think_stopping = StoppingCriteriaList([
            _BoundaryStoppingCriteria([
                self.tokenizer("</think>", add_special_tokens=False)["input_ids"]
            ])
        ])
        self._answer_stopping = StoppingCriteriaList([
            _BoundaryStoppingCriteria([
                self.tokenizer("</answer>", add_special_tokens=False)["input_ids"]
            ])
        ])
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        # Some lightweight remote inference environments do not ship
        # accelerate.  With one visible GPU, loading on CPU and moving the
        # replica explicitly is equivalent to device_map=auto and avoids that
        # optional dependency.  The normal local path keeps device_map=auto.
        if os.getenv("MEMR_NO_DEVICE_MAP", "0") == "1":
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            self.model.to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", **load_kwargs
            )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        self.memr_reranker = None
        if memr_reranker_path:
            from sid_memory_reranker import BgeSidMemoryReranker

            logger.info("Loading MemR SID reranker from %s ...", memr_reranker_path)
            self.memr_reranker = BgeSidMemoryReranker(
                memr_reranker_path,
                self.input_device,
                batch_size=memr_reranker_batch_size,
                max_length=memr_reranker_max_length,
            )
        logger.info("Transformers model loaded; input device=%s", self.input_device)

    def unload(self):
        try:
            import gc

            if self.memr_reranker is not None:
                self.memr_reranker.unload()
                self.memr_reranker = None
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _meme_sampling_kwargs() -> dict:
        """Use the v39 rollout temperature when requested by the caller."""
        if os.getenv("MEM_AGENT_DO_SAMPLE", "0") != "1":
            return {"do_sample": False}
        kwargs = {
            "do_sample": True,
            "temperature": float(os.getenv("MEM_AGENT_TEMPERATURE", "0.8")),
            "top_p": float(os.getenv("MEM_AGENT_TOP_P", "1.0")),
        }
        top_k = int(os.getenv("MEM_AGENT_TOP_K", "0"))
        if top_k > 0:
            kwargs["top_k"] = top_k
        return kwargs

    def _generate_until_boundary(
        self,
        prompt: str,
        num_beams: int = 1,
        num_return_sequences: int = 1,
    ) -> list[str]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                **self._meme_sampling_kwargs(),
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self._boundary_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_texts = []
        prompt_len = encoded["input_ids"].shape[1]
        for row in output:
            generated = self.tokenizer.decode(
                row[prompt_len:], skip_special_tokens=False
            )
            boundaries = [
                (generated.find("</tool_call>"), "</tool_call>"),
                (generated.find("</answer>"), "</answer>"),
            ]
            boundaries = [(pos, marker) for pos, marker in boundaries if pos >= 0]
            if boundaries:
                pos, marker = min(boundaries, key=lambda item: item[0])
                generated = generated[: pos + len(marker)]
            generated_texts.append(generated)
        return generated_texts

    def _generate_until_think_boundary(self, prompt: str, max_new_tokens: int) -> str:
        """Generate the first MemE thought with the same cap as the RL loop.

        The verl RL loop caps the pre-lookup thought (v39 uses 1280 tokens) and
        then continues with the structured tool prefix.  The old Transformers
        evaluator decoded the whole response in one call, so a checkpoint
        could spend all 6K tokens repeating a thought and never reach a SID.
        """
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                **self._meme_sampling_kwargs(),
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=max(1, int(max_new_tokens)),
                stopping_criteria=self._think_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_len = encoded["input_ids"].shape[1]
        self._record_generation_budget('think', prompt_len, output.shape[1] - prompt_len, max_new_tokens)
        return self.tokenizer.decode(
            output[0, prompt_len:], skip_special_tokens=False
        )

    def _generate_until_answer_boundary(self, prompt: str, max_new_tokens: int) -> str:
        """Generate a bounded post-lookup answer continuation."""
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        generation_kwargs = self._meme_sampling_kwargs()
        # A malformed post-tool continuation can enter a token loop.  Keep
        # these conservative and configurable: they are only a last-resort
        # decoder guard, not a semantic quality score.
        repetition_penalty = float(os.getenv("MEME_REPETITION_PENALTY", "1.05"))
        no_repeat_ngram_size = int(os.getenv("MEME_NO_REPEAT_NGRAM_SIZE", "4"))
        if repetition_penalty > 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                **generation_kwargs,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=max(1, int(max_new_tokens)),
                stopping_criteria=self._answer_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_len = encoded["input_ids"].shape[1]
        self._record_generation_budget('answer', prompt_len, output.shape[1] - prompt_len, max_new_tokens)
        return self.tokenizer.decode(
            output[0, prompt_len:], skip_special_tokens=False
        )

    def _generate_meme_first_turn(self, prompt: str) -> str:
        """Run the RL-compatible bounded thought -> one SID protocol.

        After the thought cap, only the four registered SID tokens are allowed
        for the action.  This is the same parser-side structural continuation
        used by the v39 RL loop: it does not choose a SID from the bank and it
        never inserts a ground-truth SID.
        """
        think_cap = int(os.getenv("MEME_THINK_MAX_TOKENS", "1280"))
        thought = self._generate_until_think_boundary(prompt, think_cap)
        if "</tool_call>" in thought or "<answer>" in thought:
            return thought
        if "</think>" not in thought:
            thought += "</think>"
        elif thought.rfind("</think>") < thought.rfind("<think>"):
            thought += "</think>"

        tool_prefix = (
            '<tool_call>{"name":"memory_lookup","arguments":{"sid_list":["'
        )
        sid_prompt = prompt + thought + tool_prefix
        encoded = self.tokenizer(sid_prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        sid_token_levels = build_sid_token_levels(self.tokenizer)
        allowed_tokens = make_sid_prefix_allowed_tokens(prompt_width, sid_token_levels)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=len(sid_token_levels),
                min_new_tokens=len(sid_token_levels),
                early_stopping=False,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        sid = self.tokenizer.decode(
            output[0, prompt_width:], skip_special_tokens=False
        ).strip()
        return thought + tool_prefix + sid + '"]}}</tool_call>'

    def _generate_meme_post_tool(self, prompt: str) -> str:
        """Match the RL post-tool caps and force one closed answer block."""
        think_cap = int(os.getenv("MEME_POST_THINK_MAX_TOKENS", "768"))
        answer_cap = int(os.getenv("MEME_ANSWER_MAX_TOKENS", "2048"))
        # The native RL loop inserts this non-trainable prefix after the tool
        # response.  Without it, the Transformers evaluator leaves the model
        # at a bare assistant boundary and some checkpoints fall into numeric
        # or punctuation loops instead of entering the comparison thought.
        post_prompt = prompt
        if not post_prompt.rstrip().endswith("<think>"):
            post_prompt += "<think>"
        raw_thought = self._generate_until_think_boundary(post_prompt, think_cap)
        # The prefix is structural; remove a duplicate one if the model emits
        # it despite being prefilled.
        thought_body = raw_thought.lstrip()
        if thought_body.startswith("<think>"):
            thought_body = thought_body[len("<think>"):].lstrip()
        thought = "<think>" + thought_body
        if "</think>" not in thought:
            thought += "</think>"
        # The opening answer tag is structural, as in the RL parser's answer
        # prefix; only the evolved experience is generated by the checkpoint.
        answer = self._generate_until_answer_boundary(
            prompt + thought + "<answer>", answer_cap
        )
        if answer.lstrip().startswith("<answer>"):
            answer = answer.lstrip()[len("<answer>") :]
        if "</answer>" in answer:
            answer = answer[: answer.find("</answer>") + len("</answer>")]
        else:
            answer += "</answer>"
        return thought + "<answer>" + answer

    def _generate_many_until_boundary(self, prompts: list[str]) -> list[str]:
        """Generate one deterministic continuation for each prompt in a batch."""
        if not prompts:
            return []
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=self.memr_fusion_max_new_tokens,
                stopping_criteria=self._boundary_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        generated_texts = []
        for row in output:
            generated = self.tokenizer.decode(row[prompt_len:], skip_special_tokens=False)
            boundaries = [
                (generated.find("</tool_call>"), "</tool_call>"),
                (generated.find("</answer>"), "</answer>"),
            ]
            boundaries = [(pos, marker) for pos, marker in boundaries if pos >= 0]
            if boundaries:
                pos, marker = min(boundaries, key=lambda item: item[0])
                generated = generated[: pos + len(marker)]
            generated_texts.append(generated)
        return generated_texts

    def _generate_shared_think_sid_beams(self, prompt: str) -> tuple[str, list[str]]:
        """Generate reasoning once, then beam-search only the four SID tokens."""
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        sid_token_levels = build_sid_token_levels(self.tokenizer)
        stop_ids = [self.tokenizer.eos_token_id, *sid_token_levels[0]]
        stop_ids = [token_id for token_id in stop_ids if token_id is not None]

        with torch.inference_mode():
            thinking_output = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=self.memr_think_max_new_tokens,
                eos_token_id=stop_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        thinking_ids = thinking_output[0, prompt_width:].tolist()
        shared_ids, first_sid = split_before_first_sid_token(
            thinking_ids,
            set(sid_token_levels[0]),
        )
        shared_text = self.tokenizer.decode(shared_ids, skip_special_tokens=False)
        self.last_memr_routing = {
            "thinking_tokens": len(thinking_ids),
            "thinking_limit": self.memr_think_max_new_tokens,
            "reached_sid": first_sid is not None,
            "shared_text": shared_text,
        }
        if first_sid is None:
            # Preserve the original query, but discard the looping continuation.
            # Only the tool-call format is supplied; all four SID values still
            # come from model-scored constrained beam search below.
            logger.warning('MemR reasoning exhausted without SID; recover with constrained SID routing')
            self.last_memr_routing['recovery'] = 'query_preserved_discard_loop_force_tool_prefix_v1'
            shared_text = ('<think>Retrieve relevant experience for the query.</think>\n'
                           '<tool_call>{"name":"memory_lookup","arguments":{"sid_list":["')
            shared_ids = self.tokenizer.encode(shared_text, add_special_tokens=False)

        shared_tensor = torch.tensor(
            [shared_ids],
            dtype=encoded["input_ids"].dtype,
            device=encoded["input_ids"].device,
        )
        stage2_input_ids = torch.cat([encoded["input_ids"], shared_tensor], dim=1)
        stage2_attention_mask = torch.cat(
            [encoded["attention_mask"], torch.ones_like(shared_tensor)], dim=1
        )
        stage2_width = stage2_input_ids.shape[1]
        allowed_tokens = make_sid_prefix_allowed_tokens(stage2_width, sid_token_levels)
        if os.getenv("MEMR_SEQUENTIAL_SID_BEAM", "0") == "1":
            # All SID candidates have exactly four tokens. Compute the long
            # shared prefix once, then score each short prefix sequentially.
            # This is the same cumulative-log-probability beam objective as
            # constrained beam search without materializing 50 long prefixes.
            import copy
            with torch.inference_mode():
                base = self.model(
                    input_ids=stage2_input_ids,
                    attention_mask=stage2_attention_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
            first_scores = torch.log_softmax(base.logits[0, -1].float(), dim=-1)
            beams = [(float(first_scores[t]), [t]) for t in sid_token_levels[0]]
            beams.sort(key=lambda x: x[0], reverse=True)
            beams = beams[:self.memr_sid_beams]
            base_cache = base.past_key_values
            del base
            for level_ids in sid_token_levels[1:]:
                expanded = []
                for score, prefix in beams:
                    cache = copy.deepcopy(base_cache)
                    prefix_tensor = torch.tensor([prefix], dtype=stage2_input_ids.dtype,
                                                 device=stage2_input_ids.device)
                    mask = torch.ones((1, stage2_width + len(prefix)),
                                      dtype=stage2_attention_mask.dtype,
                                      device=stage2_attention_mask.device)
                    with torch.inference_mode():
                        output = self.model(input_ids=prefix_tensor, attention_mask=mask,
                                            past_key_values=cache, use_cache=True,
                                            logits_to_keep=1)
                    scores = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                    expanded.extend((score + float(scores[t]), prefix + [t]) for t in level_ids)
                    del output, cache
                expanded.sort(key=lambda x: x[0], reverse=True)
                beams = expanded[:self.memr_sid_beams]
            sid_rows = [prefix for _, prefix in beams]
            del base_cache
        else:
            with torch.inference_mode():
                sid_outputs = self.model.generate(
                input_ids=stage2_input_ids,
                attention_mask=stage2_attention_mask,
                do_sample=False,
                num_beams=self.memr_sid_beams,
                num_return_sequences=self.memr_sid_beams,
                max_new_tokens=len(sid_token_levels),
                min_new_tokens=len(sid_token_levels),
                early_stopping=False,
                prefix_allowed_tokens_fn=allowed_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                )
            sid_rows = validate_sid_beam_token_ids(
                sid_outputs[:, stage2_width:].tolist(),
                [set(level) for level in sid_token_levels],
                self.memr_sid_beams,
            )
        candidates = [
            self.tokenizer.decode(row, skip_special_tokens=False).strip()
            for row in sid_rows
        ]
        return shared_text, candidates

    @staticmethod
    def _answer_from_text(text: str) -> str:
        matches = list(_ANSWER_RE.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
        return text.strip()

    def _beam_fusion_summary_prompt(self, query: str, memory_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You summarize retrieved experience for a downstream agent. "
                    "Do not call any tool and do not generate SIDs. Extract only actionable "
                    "guidance relevant to the current query. Output exactly one non-empty "
                    "<answer>...</answer> block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current query:\n{query}\n\n"
                    f"Retrieved experience:\n{memory_text}\n\n"
                    "Summarize the useful experience now."
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def _fuse_beam_memories(
        self,
        base_text: str,
        query: str,
        generateds: list[str],
    ) -> dict | None:
        """Apply one of the two controlled Beam-memory fusion strategies."""
        branches: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate in generateds:
            for sid in _parse_sid_list(candidate):
                if sid not in seen:
                    seen.add(sid)
                    branches.append((sid, candidate))
                    break
            if len(branches) >= self.memr_sid_beams:
                break
        if not branches:
            return None

        raw_candidate_sids = [sid for sid, _ in branches]
        memories = [lookup_sid(sid, self.bank, self.index, self.top_k) for sid, _ in branches]
        rerank_scores = []
        if self.memr_reranker is not None:
            ranked = self.memr_reranker.rank(
                query,
                list(zip(raw_candidate_sids, memories)),
                top_k=len(branches),
            )
            rerank_scores = [
                {"sid": row["sid"], "score": row["score"]} for row in ranked
            ]
            generated_by_sid = {sid: generated for sid, generated in branches}
            selected = ranked[: self.memr_rerank_top_k]
            branches = [(row["sid"], generated_by_sid[row["sid"]]) for row in selected]
            memories = [row["memory"] for row in selected]
        branch_summaries = []
        summary_prompts = []
        if self.memr_beam_fusion == "merge_then_summarize":
            merged_memory = "\n\n".join(
                f"[Beam {rank} | {sid}]\n{memory}"
                for rank, ((sid, _), memory) in enumerate(zip(branches, memories), start=1)
            )
            summary_prompts = [self._beam_fusion_summary_prompt(query, merged_memory)]
            generated_summaries = self._generate_many_until_boundary(summary_prompts)
            merged = self._answer_from_text(generated_summaries[0]) if generated_summaries else ""
        else:
            summary_prompts = [
                self._beam_fusion_summary_prompt(query, memory) for memory in memories
            ]
            generated_summaries = self._generate_many_until_boundary(summary_prompts)
            for (sid, _), generated in zip(branches, generated_summaries):
                branch_summaries.append({
                    "sid": sid,
                    "summary": self._answer_from_text(generated),
                })
            merged = "\n\n".join(
                f"[Beam {rank} | {item['sid']}]\n{item['summary']}"
                for rank, item in enumerate(branch_summaries, start=1)
                if item["summary"]
            )
        sid_m = _SID_RE.search(branches[0][0])
        predicted_sid = tuple(map(int, sid_m.groups())) if sid_m else None
        return {
            "answer": merged,
            "sid_l1": predicted_sid[0] if predicted_sid else None,
            "sid_l2": predicted_sid[1] if predicted_sid else None,
            "sid_l3": predicted_sid[2] if predicted_sid else None,
            "sid_l4": predicted_sid[3] if predicted_sid else None,
            "candidate_sids": raw_candidate_sids,
            "retrieved_sids": [sid for sid, _ in branches],
            "rerank_scores": rerank_scores,
            "branch_summaries": branch_summaries,
            "fusion_strategy": self.memr_beam_fusion,
            "full_text": (
                base_text
                + "\n\n===== FUSION SUMMARY =====\n\n"
                + "\n\n===== SUMMARY BRANCH =====\n\n".join(
                    prompt + generated
                    for prompt, generated in zip(summary_prompts, generated_summaries)
                )
            ),
        }

    def _run_one(self, system_prompt: str, user_content: str) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # The RL checkpoint was trained with the explicit XML ``<think>`` /
        # ``<tool_call>`` protocol.  Qwen's chat template has two modes:
        # ``enable_thinking=True`` leaves the assistant turn unconstrained,
        # while ``False`` pre-fills an empty ``<think>`` block.  Keep the
        # historical default for normal evaluations, but allow a diagnostic
        # or external-bootstrap run to select the trained, explicit mode
        # without editing the inference code again.
        enable_thinking = os.getenv("MEM_AGENT_ENABLE_THINKING", "1") != "0"
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tools=MEMORY_LOOKUP_TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        predicted_sid: Optional[tuple[int, int, int, int]] = None
        generated_parts: list[str] = []
        candidate_sids: list[str] = []

        for turn in range(self.max_turns):
            beam_count = self.memr_sid_beams if turn == 0 and system_prompt == MEMR_SYSTEM_PROMPT else 1
            generateds: list[str] | None = None
            if turn == 0 and system_prompt == MEMR_SYSTEM_PROMPT and beam_count > 1:
                shared_text, generateds = self._generate_shared_think_sid_beams(full_text)
                if generateds:
                    result = self._fuse_beam_memories(
                        full_text + shared_text,
                        user_content,
                        generateds,
                    )
                    if result is not None:
                        return result
                else:
                    full_text += shared_text
                    generateds = [""]
            if generateds is None:
                if turn == 0 and system_prompt == MEME_SYSTEM_PROMPT:
                    # Match the v39 RL agent loop: cap the first thought and
                    # continue with one structurally valid SID tool call.
                    generateds = [self._generate_meme_first_turn(full_text)]
                elif turn > 0 and system_prompt == MEME_SYSTEM_PROMPT:
                    generateds = [self._generate_meme_post_tool(full_text)]
                else:
                    generateds = self._generate_until_boundary(
                        full_text,
                        num_beams=beam_count,
                        num_return_sequences=beam_count,
                    )
            if turn == 0 and system_prompt == MEMR_SYSTEM_PROMPT and beam_count > 1:
                result = self._fuse_beam_memories(full_text, user_content, generateds)
                if result is not None:
                    return result
            generated = generateds[0] if generateds else ""
            generated_parts.append(generated)
            full_text += generated
            if "</answer>" in generated or "</tool_call>" not in generated:
                break
            sid_m = _SID_RE.search(generated)
            if sid_m and predicted_sid is None:
                predicted_sid = tuple(map(int, sid_m.groups()))
            sid_list = []
            for candidate in generateds:
                for sid in _parse_sid_list(candidate):
                    if sid not in sid_list:
                        sid_list.append(sid)
            candidate_sids = sid_list[: self.memr_sid_beams]
            if not sid_list:
                break
            tool_result = "\n\n".join(
                lookup_sid(sid, self.bank, self.index, self.top_k)
                for sid in candidate_sids
            )
            full_text = _append_tool_response(full_text, tool_result)

        # Never parse the system prompt here: both prompt templates contain a
        # worked arithmetic example with an <answer> block.  The previous
        # implementation selected that example as MemE's answer.
        generated_text = "\n".join(generated_parts)
        answer = ""
        answer_matches = list(_ANSWER_RE.finditer(generated_text))
        if answer_matches:
            answer = answer_matches[-1].group(1).strip()
        elif generated_text.strip():
            answer = generated_text.strip()
        if predicted_sid is None:
            for match in _TOOL_CALL_RE.finditer(generated_text):
                sid_m = _SID_RE.search(match.group(1))
                if sid_m:
                    predicted_sid = tuple(map(int, sid_m.groups()))
                    break
        return {
            "answer": answer,
            "sid_l1": predicted_sid[0] if predicted_sid else None,
            "sid_l2": predicted_sid[1] if predicted_sid else None,
            "sid_l3": predicted_sid[2] if predicted_sid else None,
            "sid_l4": predicted_sid[3] if predicted_sid else None,
            "candidate_sids": candidate_sids,
            "retrieved_sids": candidate_sids,
            "rerank_scores": [],
            "branch_summaries": [],
            "fusion_strategy": self.memr_beam_fusion,
            "full_text": full_text,
        }

    def run_memr_batch(self, questions: list[str]) -> list[dict]:
        logger.info("MemR inference: %s questions", len(questions))
        results = []
        for i, question in enumerate(questions):
            results.append(
                self._run_one(
                    MEMR_SYSTEM_PROMPT,
                    f"<query> {question} </query>",
                )
            )
            if (i + 1) % 20 == 0 or i + 1 == len(questions):
                logger.info("MemR progress: %s/%s", i + 1, len(questions))
        return results

    def run_meme_batch(self, trajectory_texts: list[str]) -> list[dict]:
        logger.info("MemE inference: %s trajectories", len(trajectory_texts))
        results = []
        for i, trajectory in enumerate(trajectory_texts):
            self.generation_budget_trace = []
            results.append(
                self._run_one(MEME_SYSTEM_PROMPT, f"<trajectory> {trajectory} </trajectory>")
            )
            results[-1]['generation_budget_trace'] = list(self.generation_budget_trace)
            if (i + 1) % 20 == 0 or i + 1 == len(trajectory_texts):
                logger.info("MemE progress: %s/%s", i + 1, len(trajectory_texts))
        return results

    def _record_generation_budget(self, phase, input_tokens, output_tokens, cap):
        if not hasattr(self, 'generation_budget_trace'):
            self.generation_budget_trace = []
        self.generation_budget_trace.append({
            'phase': phase, 'input_tokens': int(input_tokens),
            'output_tokens': int(output_tokens), 'output_cap': int(cap),
            'hit_cap': int(output_tokens) >= int(cap),
            'model_context_limit': getattr(self.model.config, 'max_position_embeddings', None),
        })
