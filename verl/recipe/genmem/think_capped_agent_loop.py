"""Experiment-local ToolAgentLoop with a hard cap for an unclosed think block."""

from __future__ import annotations

from copy import deepcopy
import html
import json
import re

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.profiler import simple_timer


@register("think_capped_tool_agent")
class ThinkCappedToolAgentLoop(ToolAgentLoop):
    """Parse one model-selected SID, execute it, then reserve room for its answer.

    The legacy path never appends a user/system follow-up instruction.  The
    optional causal MemE path appends a hidden failed trajectory only after
    the model has committed its SID and the lookup has executed.  Parser code
    can supply missing *syntax* around an already generated complete SID, but
    it never fabricates or substitutes a SID.
    """

    think_max_tokens = 2048
    post_tool_think_max_tokens = 768
    post_tool_answer_reserve_tokens = 1024
    inject_hidden_trajectory_after_lookup = False
    oracle_sid_route = False
    post_lookup_evolution_instruction = ""
    post_think_parser_tokens = 512
    _structured_tool_prefix = (
        '<tool_call>{"name":"memory_lookup","arguments":{"sid_list":["'
    )
    _structured_tool_suffix = '"]}}</tool_call>'
    _sid_pattern = re.compile(
        r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
    )
    _sid_bounds = (48, 16, 8, 8)
    _sid_terminal_tokens = tuple(f"<SID_L4_{index}>" for index in range(8))

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)
        cls.think_max_tokens = int(kwargs.get("think_max_tokens", 2048))
        cls.post_tool_think_max_tokens = int(
            kwargs.get("post_tool_think_max_tokens", 768)
        )
        cls.post_tool_answer_reserve_tokens = int(
            kwargs.get("post_tool_answer_reserve_tokens", 1024)
        )
        cls.inject_hidden_trajectory_after_lookup = bool(
            kwargs.get("inject_hidden_trajectory_after_lookup", False)
        )
        cls.oracle_sid_route = bool(kwargs.get("oracle_sid_route", False))
        cls.post_lookup_evolution_instruction = str(
            kwargs.get("post_lookup_evolution_instruction", "") or ""
        )

    @staticmethod
    def _hidden_trajectory(agent_data: AgentData) -> str:
        memory_kwargs = (agent_data.tools_kwargs or {}).get("memory_lookup") or {}
        create_kwargs = memory_kwargs.get("create_kwargs") or {}
        ground_truth = create_kwargs.get("ground_truth") or {}
        return str(ground_truth.get("trajectory") or "")

    @classmethod
    def _oracle_sid_tool_call(cls, agent_data: AgentData) -> FunctionCall | None:
        """Return a target-SID lookup for an isolated MemE upper-bound probe.

        This is intentionally disabled by default and must never be enabled for
        policy training: the target SID is privileged ground-truth evidence.
        """

        memory_kwargs = (agent_data.tools_kwargs or {}).get("memory_lookup") or {}
        create_kwargs = memory_kwargs.get("create_kwargs") or {}
        ground_truth = create_kwargs.get("ground_truth") or {}
        sid = str(ground_truth.get("sid_tokens") or "").strip()
        if cls._sid_pattern.fullmatch(sid) is None:
            return None
        codes = tuple(int(match) for match in cls._sid_pattern.fullmatch(sid).groups())
        if any(code < 0 or code >= bound for code, bound in zip(codes, cls._sid_bounds)):
            return None
        return FunctionCall(
            name="memory_lookup",
            arguments=json.dumps({"sid_list": [sid]}, ensure_ascii=False),
        )

    @staticmethod
    def _post_lookup_messages(
        retrieved: str, trajectory: str, instruction: str = ""
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "tool", "content": retrieved},
            {
                "role": "user",
                "content": (
                    "<failed_trajectory>\n"
                    + html.escape(trajectory, quote=False)
                    + "\n</failed_trajectory>"
                ),
            },
        ]
        if instruction.strip():
            messages.append({"role": "user", "content": html.escape(instruction, quote=False)})
        return messages

    @staticmethod
    def _needs_lookup_continuation(
        *, response_token_count: int, tool_calls: list[FunctionCall], think_max_tokens: int
    ) -> bool:
        """Whether a parser-only continuation may finish an interrupted call."""

        return response_token_count >= think_max_tokens and not tool_calls

    @staticmethod
    def _keep_last_tool_call(agent_data: AgentData) -> None:
        if agent_data.tool_calls:
            agent_data.tool_calls = [agent_data.tool_calls[-1]]
            agent_data.metrics["last_tool_call_used"] = 1.0

    @classmethod
    def _recover_truncated_lookup(cls, generated_text: str) -> FunctionCall | None:
        """Recover only a complete SID from one trailing, malformed lookup.

        A cap may cut the JSON closing syntax after the model has already
        emitted the complete four-layer SID.  The SID itself remains the
        model's action; this helper supplies only the deterministic JSON
        scaffolding required by the executor.  It refuses partial SIDs,
        another tool name, multiple SIDs, and out-of-range codes.
        """

        start = (generated_text or "").rfind("<tool_call>")
        if start < 0:
            return None
        trailing_call = generated_text[start + len("<tool_call>") :]
        if '"name":"memory_lookup"' not in trailing_call.replace(" ", ""):
            return None
        matches = list(cls._sid_pattern.finditer(trailing_call))
        if not matches:
            return None
        # Take the first SID the model emitted after the tool-call opening bracket.
        codes = tuple(int(matches[0].group(index)) for index in range(1, 5))
        if any(code >= bound for code, bound in zip(codes, cls._sid_bounds)):
            return None
        sid = matches[0].group(0)
        return FunctionCall(
            name="memory_lookup",
            arguments=json.dumps({"sid_list": [sid]}, ensure_ascii=False),
        )

    @classmethod
    def _recover_raw_sid_lookup(cls, generated_text: str) -> FunctionCall | None:
        """Recover the final complete, in-range SID from unstructured output.

        Hermes can reject a model-produced bare JSON function object even
        though the model has already selected a complete SID.  A thought can
        contain one or more earlier candidates, so the last complete SID is
        the model's final action.  Missing, partial, and out-of-range values
        remain rejected; this helper never fabricates a SID.
        """

        matches = list(cls._sid_pattern.finditer(generated_text or ""))
        if not matches:
            return None
        final_match = matches[-1]
        codes = tuple(int(final_match.group(index)) for index in range(1, 5))
        if any(code >= bound for code, bound in zip(codes, cls._sid_bounds)):
            return None
        sid = final_match.group(0)
        return FunctionCall(
            name="memory_lookup",
            arguments=json.dumps({"sid_list": [sid]}, ensure_ascii=False),
        )

    async def _append_structured_sid_prefix(
        self, agent_data: AgentData, *, close_open_think: bool
    ) -> None:
        """Append parser syntax only; the model still emits the four SID levels.

        This is an assistant-side continuation, not a message or an added
        natural-language instruction.  In particular, it never changes the
        dataset's system/user prompt.
        """

        prefix = ("</think>" if close_open_think else "") + self._structured_tool_prefix
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(prefix, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.extra_fields["parser_structured_sid_prefix"] = prefix
        agent_data.metrics["parser_structured_sid_continuation"] = 1.0

    async def _append_structured_tool_suffix(self, agent_data: AgentData) -> None:
        """Close a parser-opened tool call after its model-emitted SID."""

        injected = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(
                self._structured_tool_suffix, add_special_tokens=False
            ),
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.metrics["parser_closed_structured_tool_call"] = 1.0

    async def _append_parser_tool_call(self, agent_data: AgentData, sid: str) -> None:
        """Add only parser syntax after a model-selected SID for reward replay.

        The original model text stays intact.  The appended canonical call is
        non-trainable structural syntax, allowing the reward reader to replay
        the same lookup when Hermes rejected the original formatting.
        """

        canonical = (
            '<tool_call>{"name":"memory_lookup","arguments":{"sid_list":["'
            + sid
            + '"]}}</tool_call>'
        )
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(canonical, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.metrics["parser_normalized_tool_call"] = 1.0

    async def _recover_tool_call_from_text(
        self, agent_data: AgentData, generated_text: str
    ) -> bool:
        """Restore a parser-rejected lookup only from a complete model SID."""

        if agent_data.tool_calls:
            return False
        recovered = self._recover_truncated_lookup(generated_text)
        if recovered is None:
            recovered = self._recover_raw_sid_lookup(generated_text)
        if recovered is None:
            return False
        agent_data.tool_calls = [recovered]
        sid = json.loads(recovered.arguments)["sid_list"][0]
        structured_prefix = agent_data.extra_fields.get("parser_structured_sid_prefix")
        if structured_prefix:
            start = generated_text.rfind(structured_prefix)
            trailing_text = generated_text[start:] if start >= 0 else generated_text
            if "</tool_call>" not in trailing_text:
                await self._append_structured_tool_suffix(agent_data)
        else:
            await self._append_parser_tool_call(agent_data, sid)
        agent_data.metrics["parser_sid_recovered"] = 1.0
        return True

    async def _append_answer_prefix(self, agent_data: AgentData) -> None:
        """Seed a non-trainable answer span before the post-lookup decode."""

        if getattr(agent_data, "extra_fields", {}).get("parser_answer_prefix"):
            return
        prefix = "<answer>"
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(prefix, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.extra_fields["parser_answer_prefix"] = prefix
        agent_data.metrics["parser_answer_prefix"] = 1.0

    async def _append_post_tool_think_prefix(self, agent_data: AgentData) -> None:
        """Open the MemE comparison-thought phase after the tool observation."""

        prefix = "<think>"
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(prefix, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.metrics["parser_post_tool_think_prefix"] = 1.0

    async def _append_post_tool_think_suffix(self, agent_data: AgentData) -> None:
        """Close a capped comparison thought without assigning policy loss."""

        suffix = "</think>"
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(suffix, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.metrics["parser_closed_post_tool_think"] = 1.0

    async def _append_answer_suffix(self, agent_data: AgentData) -> None:
        """Close a parser-opened answer span without assigning it policy loss."""

        suffix = "</answer>"
        injected = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(suffix, add_special_tokens=False)
        )
        agent_data.prompt_ids += injected
        agent_data.response_mask += [0] * len(injected)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(injected)
        agent_data.metrics["parser_closed_answer"] = 1.0

    async def _wrap_last_model_answer(self, agent_data: AgentData) -> None:
        """Close a seeded answer span or wrap a legacy untagged completion."""

        generated_ids = list(agent_data.response_ids)
        if not generated_ids:
            agent_data.metrics["answer_missing_after_lookup"] = 1.0
            return
        generated_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        )
        if getattr(agent_data, "extra_fields", {}).get("parser_answer_prefix"):
            if "</answer>" not in generated_text:
                await self._append_answer_suffix(agent_data)
            return
        if re.search(r"<answer>\s*.+?\s*</answer>", generated_text, re.S):
            return
        wrapper_open, wrapper_close = await self.loop.run_in_executor(
            None,
            lambda: (
                self.tokenizer.encode("<answer>", add_special_tokens=False),
                self.tokenizer.encode("</answer>", add_special_tokens=False),
            ),
        )
        prior_len = len(agent_data.response_mask) - len(generated_ids)
        prior_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(generated_ids)]
        prior_mask = agent_data.response_mask[:prior_len]
        generated_mask = agent_data.response_mask[prior_len:]
        agent_data.prompt_ids = prior_ids + wrapper_open + generated_ids + wrapper_close
        agent_data.response_mask = (
            prior_mask + [0] * len(wrapper_open) + generated_mask + [0] * len(wrapper_close)
        )
        if agent_data.response_logprobs:
            prior_logprobs = agent_data.response_logprobs[:prior_len]
            generated_logprobs = agent_data.response_logprobs[prior_len:]
            agent_data.response_logprobs = (
                prior_logprobs
                + [0.0] * len(wrapper_open)
                + generated_logprobs
                + [0.0] * len(wrapper_close)
            )
        agent_data.metrics["parser_wrapped_answer"] = 1.0

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict, ignore_termination: bool = False
    ):
        if self.oracle_sid_route and not agent_data.metrics.get("lookup_executed"):
            oracle_call = self._oracle_sid_tool_call(agent_data)
            if oracle_call is None:
                agent_data.metrics["oracle_sid_missing"] = 1.0
                return AgentState.TERMINATED
            sid = json.loads(oracle_call.arguments)["sid_list"][0]
            canonical = (
                "<think>Oracle SID route for diagnostic.</think>"
                + self._structured_tool_prefix
                + sid
                + self._structured_tool_suffix
            )
            injected = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(canonical, add_special_tokens=False)
            )
            agent_data.prompt_ids += injected
            agent_data.response_mask += [0] * len(injected)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(injected)
            agent_data.tool_calls = [oracle_call]
            agent_data.messages.append({"role": "assistant", "content": canonical})
            agent_data.metrics["oracle_sid_route"] = 1.0
            agent_data.extra_fields["oracle_sid"] = sid
            return AgentState.PROCESSING_TOOLS
        if agent_data.metrics.get("lookup_executed"):
            if not agent_data.extra_fields.get("post_tool_think_done"):
                structural_tokens = await self.loop.run_in_executor(
                    None,
                    lambda: len(
                        self.tokenizer.encode(
                            "<think></think><answer></answer>",
                            add_special_tokens=False,
                        )
                    ),
                )
                remaining = (
                    self.response_length
                    - len(agent_data.response_mask)
                    - self.post_tool_answer_reserve_tokens
                    - structural_tokens
                )
                if remaining <= 0:
                    agent_data.metrics["post_tool_think_missing_no_budget"] = 1.0
                    return AgentState.TERMINATED
                await self._append_post_tool_think_prefix(agent_data)
                comparison_params = deepcopy(sampling_params)
                comparison_params["max_new_tokens"] = min(
                    self.post_tool_think_max_tokens, remaining
                )
                comparison_params["min_tokens"] = 1
                comparison_params["stop"] = ["</think>"]
                comparison_params["include_stop_str_in_output"] = True
                await super()._handle_generating_state(
                    agent_data, comparison_params, ignore_termination=True
                )
                generated_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(
                        agent_data.response_ids, skip_special_tokens=False
                    ),
                )
                if "</think>" not in generated_text:
                    await self._append_post_tool_think_suffix(agent_data)
                agent_data.extra_fields["post_tool_think_done"] = True
                agent_data.metrics["post_tool_think_generated"] = 1.0
                return AgentState.GENERATING

            answer_delimiter_len = await self.loop.run_in_executor(
                None,
                lambda: len(self.tokenizer.encode("<answer></answer>", add_special_tokens=False)),
            )
            remaining_before_prefix = self.response_length - len(agent_data.response_mask)
            if remaining_before_prefix <= answer_delimiter_len:
                agent_data.metrics["answer_missing_no_budget"] = 1.0
                return AgentState.TERMINATED
            await self._append_answer_prefix(agent_data)
            suffix_len = await self.loop.run_in_executor(
                None, lambda: len(self.tokenizer.encode("</answer>", add_special_tokens=False))
            )
            remaining = self.response_length - len(agent_data.response_mask) - suffix_len
            if remaining <= 0:
                agent_data.metrics["answer_missing_no_budget"] = 1.0
                return AgentState.TERMINATED
            answer_params = deepcopy(sampling_params)
            answer_params["max_new_tokens"] = min(
                self.post_tool_answer_reserve_tokens, remaining
            )
            # The content between structural answer tags remains entirely
            # model-generated; this only prevents an immediate EOS response.
            answer_params["min_tokens"] = 1
            await super()._handle_generating_state(
                agent_data, answer_params, ignore_termination=True
            )
            await self._wrap_last_model_answer(agent_data)
            return AgentState.TERMINATED

        continuation_active = bool(
            agent_data.extra_fields.get("parser_sid_continuation_active")
        )
        prior_ids = list(agent_data.extra_fields.get("pre_lookup_response_ids", []))
        remaining = self.response_length - len(agent_data.response_mask)
        if continuation_active:
            if remaining <= 0:
                return AgentState.TERMINATED
            generation_params = deepcopy(sampling_params)
            generation_params["max_new_tokens"] = min(
                self.post_think_parser_tokens, remaining
            )
            # The model owns all four SID choices.  Stop as soon as its L4
            # token completes the action, then append the canonical JSON/XML
            # suffix as non-trainable parser syntax.  Without this stop,
            # Qwen sometimes continues with an OpenAI-style ``id/type``
            # envelope that the executor accepts but the reward parser must
            # reject as malformed JSON.
            # SID symbols are special tokens and are skipped by vLLM's text
            # detokenizer, so string stops cannot see them.  Stop on their
            # token ids; ``include_stop_str_in_output`` also retains the
            # terminating L4 token in the sampled action.
            terminal_token_ids = [
                self.tokenizer.convert_tokens_to_ids(token)
                for token in self._sid_terminal_tokens
            ]
            if (
                any(not isinstance(token_id, int) or token_id < 0 for token_id in terminal_token_ids)
                or len(set(terminal_token_ids)) != len(self._sid_terminal_tokens)
            ):
                raise RuntimeError("MemE L4 SID tokens are missing from the tokenizer")
            generation_params["stop_token_ids"] = terminal_token_ids
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params = deepcopy(sampling_params)
            generation_params["max_new_tokens"] = self.think_max_tokens
            # The MemE SFT target always ends its first reasoning phase at
            # </think> and emits the lookup next.  Stop there so an early
            # closing tag cannot drift into an untagged answer/memory dump;
            # the parser then supplies only the deterministic tool syntax.
            generation_params["stop"] = ["</think>"]
            generation_params["include_stop_str_in_output"] = True

        state = await super()._handle_generating_state(
            agent_data, generation_params, ignore_termination=True
        )
        combined_ids = prior_ids + list(agent_data.response_ids)
        combined_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(combined_ids, skip_special_tokens=False),
        )
        structured_prefix = agent_data.extra_fields.get("parser_structured_sid_prefix", "")
        generated_text = combined_text
        if continuation_active:
            prior_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(prior_ids, skip_special_tokens=False),
            )
            continuation_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    agent_data.response_ids, skip_special_tokens=False
                ),
            )
            generated_text = prior_text + structured_prefix + continuation_text
        await self._recover_tool_call_from_text(agent_data, generated_text)
        self._keep_last_tool_call(agent_data)
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        if not continuation_active:
            agent_data.extra_fields["pre_lookup_response_ids"] = combined_ids
            agent_data.extra_fields["parser_sid_continuation_active"] = True
            close_open_think = (
                combined_text.rfind("<think>") > combined_text.rfind("</think>")
            )
            await self._append_structured_sid_prefix(
                agent_data, close_open_think=close_open_think
            )
            return AgentState.GENERATING
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Execute one lookup, retain a budgeted observation, then generate answer.

        The stock loop drops the complete tool response when it does not fit
        in the remaining response budget.  That silently turns a successful
        lookup into an ineligible trajectory.  Here, the agent's first 2K
        thought is followed by a head/tail-preserved observation and a fixed
        answer reserve inside the 4K global response limit.
        """

        self._keep_last_tool_call(agent_data)
        if not agent_data.tool_calls:
            return AgentState.TERMINATED
        tool_call = agent_data.tool_calls[0]
        with simple_timer("tool_calls", agent_data.metrics):
            tool_response, tool_reward, _ = await self._call_tool(
                tool_call, agent_data.tools_kwargs
            )
        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

        raw_text = str(tool_response.text or "")
        safe_text = re.sub(
            r"</?tool_(?:call|response)>",
            lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
            raw_text,
        )
        post_tool_structure_tokens = await self.loop.run_in_executor(
            None,
            lambda: len(
                self.tokenizer.encode(
                    "<think></think><answer></answer>", add_special_tokens=False
                )
            ),
        )
        observation_budget = (
            self.response_length
            - len(agent_data.response_mask)
            - self.post_tool_think_max_tokens
            - self.post_tool_answer_reserve_tokens
            - post_tool_structure_tokens
        )
        available = max(0, observation_budget)

        def encode_messages(messages: list[dict[str, str]]) -> list[int]:
            if self.processor is not None:
                rendered = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                model_inputs = self.processor(
                    text=[rendered], images=None, return_tensors="pt"
                )
                return model_inputs.pop("input_ids").squeeze(0).tolist()
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            return token_ids[len(self.system_prompt) :]

        def encode_tool_message(content: str) -> list[int]:
            return encode_messages([{"role": "tool", "content": content}])

        if self.inject_hidden_trajectory_after_lookup:
            trajectory = self._hidden_trajectory(agent_data)
            if not trajectory.strip():
                agent_data.metrics["trajectory_context_missing"] = 1.0
                return AgentState.TERMINATED
            context_messages = self._post_lookup_messages(
                safe_text, trajectory, self.post_lookup_evolution_instruction
            )
            context_ids = await self.loop.run_in_executor(
                None, lambda: encode_messages(context_messages)
            )
            if len(context_ids) > available:
                agent_data.metrics["trajectory_context_budget_failure"] = 1.0
                agent_data.metrics["trajectory_context_required_tokens"] = float(
                    len(context_ids)
                )
                agent_data.metrics["trajectory_context_available_tokens"] = float(
                    available
                )
                return AgentState.TERMINATED
            tool_only_ids = await self.loop.run_in_executor(
                None, lambda: encode_tool_message(safe_text)
            )
            retrieved_tokens = min(len(tool_only_ids), len(context_ids))
            agent_data.prompt_ids += context_ids
            agent_data.response_mask += [0] * len(context_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(context_ids)
            agent_data.messages.extend(
                [
                    {"role": "tool", "content": raw_text},
                    context_messages[1],
                ]
            )
            agent_data.user_turns += 1
            agent_data.extra_fields["tool_response_raw"] = raw_text
            agent_data.metrics["lookup_executed"] = 1.0
            agent_data.metrics["trajectory_context_injected"] = 1.0
            agent_data.metrics["retrieved_context_tokens"] = float(retrieved_tokens)
            agent_data.metrics["trajectory_context_tokens"] = float(
                len(context_ids) - retrieved_tokens
            )
            agent_data.metrics["tool_observation_truncated"] = 0.0
            return AgentState.GENERATING

        def fit_observation() -> tuple[list[int], str] | None:
            full_ids = encode_tool_message(safe_text)
            if len(full_ids) <= available:
                return full_ids, safe_text
            marker = "...(truncated)..."

            def clipped_content(characters_to_keep: int) -> str:
                head = (characters_to_keep * 3 + 3) // 4
                tail = characters_to_keep - head
                suffix = safe_text[-tail:] if tail else ""
                return safe_text[:head] + marker + suffix

            smallest_content = clipped_content(0)
            smallest_ids = encode_tool_message(smallest_content)
            if len(smallest_ids) > available:
                return None
            low, high = 0, len(safe_text)
            best_content, best_ids = smallest_content, smallest_ids
            while low <= high:
                midpoint = (low + high) // 2
                candidate_content = clipped_content(midpoint)
                candidate_ids = encode_tool_message(candidate_content)
                if len(candidate_ids) <= available:
                    best_content, best_ids = candidate_content, candidate_ids
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            return best_ids, best_content

        fitted = await self.loop.run_in_executor(None, fit_observation)
        if fitted is None:
            agent_data.metrics["tool_observation_dropped_no_budget"] = 1.0
            return AgentState.TERMINATED
        observation_ids, retained_text = fitted
        agent_data.prompt_ids += observation_ids
        agent_data.response_mask += [0] * len(observation_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(observation_ids)
        agent_data.messages.append({"role": "tool", "content": retained_text})
        agent_data.user_turns += 1
        agent_data.metrics["lookup_executed"] = 1.0
        agent_data.metrics["tool_observation_truncated"] = float(
            retained_text != safe_text
        )
        return AgentState.GENERATING
