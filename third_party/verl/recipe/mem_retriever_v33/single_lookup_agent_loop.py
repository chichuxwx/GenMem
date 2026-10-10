# Modified for GenMem; see third_party/verl/README.md for provenance.
"""A one-generation, one-lookup agent loop for MemRetriever v33.

The stock :class:`ToolAgentLoop` transitions back to model generation after a
tool response.  That is useful for general agents, but it makes retrieval
credit ambiguous: later assistant tokens can issue another lookup and receive
the same trajectory-level outcome.  This recipe keeps the generated tool call
and its observation in the trajectory, then terminates before a second model
turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.utils.profiler import simple_timer


_FIXED_RESPONSE_LENGTH = 4096
_LOOKUP_TOOL_NAME = "memory_lookup"
_SID_BOUNDS = (48, 16, 8, 8)
_FOUR_LAYER_SID = re.compile(
    r"^<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>$"
)
_RESERVED_TOOL_DELIMITER = re.compile(r"</?tool_(?:call|response)>")


def _is_valid_four_layer_sid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    match = _FOUR_LAYER_SID.fullmatch(value.strip())
    if match is None:
        return False
    codes = tuple(int(match.group(index)) for index in range(1, 5))
    return all(0 <= code < bound for code, bound in zip(codes, _SID_BOUNDS))


def _escape_reserved_tool_delimiters(text: str) -> str:
    """Keep observation text from impersonating structural trajectory tags."""

    return _RESERVED_TOOL_DELIMITER.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text,
    )


class SingleLookupAgentLoopV33(ToolAgentLoop):
    """Generate once, execute only the first parsed lookup, and stop."""

    # ToolAgentLoop caches heavyweight setup on the concrete class.  Giving the
    # recipe its own flag prevents another ToolAgentLoop subclass initialized in
    # the same worker from accidentally suppressing this class's setup.
    _class_initialized = False
    first_turn_max_tokens = 512

    @classmethod
    def init_class(
        cls,
        config,
        tokenizer,
        processor,
        *,
        first_turn_max_tokens: int = 512,
        required_response_length: int = 4096,
        **kwargs,
    ):
        actual_response_length = int(config.actor_rollout_ref.rollout.response_length)
        if int(required_response_length) != _FIXED_RESPONSE_LENGTH:
            raise ValueError(
                "SingleLookupAgentLoopV33 has a fixed 4096-token response; "
                "required_response_length is not an override"
            )
        if actual_response_length != _FIXED_RESPONSE_LENGTH:
            raise ValueError(
                "SingleLookupAgentLoopV33 requires rollout.response_length="
                f"{_FIXED_RESPONSE_LENGTH}, got {actual_response_length}"
            )
        if not 0 < int(first_turn_max_tokens) < actual_response_length:
            raise ValueError(
                "first_turn_max_tokens must be positive and smaller than the "
                "total response length"
            )

        super().init_class(config, tokenizer, processor, **kwargs)
        cls.first_turn_max_tokens = int(first_turn_max_tokens)

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        """Attach authoritative action/execution facts for the reward worker."""

        self._execution_audit = {
            "selector_version": "first_valid_direct_lookup_v1",
            "selected_tool_name": None,
            "selected_arguments": None,
            "selected_sid": None,
            "execution_attempted": False,
            "execution_completed": False,
            "lookup_hit_count": 0,
            "observation_present": False,
        }
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["single_lookup_v33"] = dict(self._execution_audit)
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Keep only the short, credit-bearing first assistant turn.

        The async vLLM server computes ``max_tokens`` itself and passes it as
        an explicit ``SamplingParams`` argument.  Supplying another
        ``max_tokens`` in ``sampling_params`` therefore raises a duplicate
        keyword error.  Clip the returned token stream locally instead: this
        preserves the stock server boundary while reserving deterministic
        space in the 4K response for the lookup observation.
        """

        del ignore_termination  # This recipe always ends after its first turn.

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        retained_ids = output.token_ids[: self.first_turn_max_tokens]
        agent_data.response_ids = retained_ids
        agent_data.prompt_ids += retained_ids
        agent_data.response_mask += [1] * len(retained_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs[: len(retained_ids)]

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts[: len(retained_ids)]

        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            retained_ids
        )
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    @staticmethod
    def _first_valid_direct_lookup(tool_calls):
        """Select one causal action without letting malformed calls consume it."""

        for tool_call in tool_calls:
            if tool_call.name != _LOOKUP_TOOL_NAME:
                continue
            try:
                arguments = json.loads(tool_call.arguments)
            except (TypeError, json.JSONDecodeError):
                continue
            sid_list = arguments.get("sid_list") if isinstance(arguments, dict) else None
            if (
                isinstance(sid_list, list)
                and len(sid_list) == 1
                and _is_valid_four_layer_sid(sid_list[0])
            ):
                return tool_call
        return None

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Execute the first parsed call, retain its response, then terminate."""

        # This state is reachable only when the parser found at least one call.
        # Intentionally ignore every later call so a rollout can cause at most
        # one environment action.  The raw generated calls remain in response_ids
        # for the reward function to penalize malformed/multiple-call formats.
        first_tool_call = self._first_valid_direct_lookup(agent_data.tool_calls)
        if first_tool_call is None:
            return AgentState.TERMINATED
        selected_arguments = json.loads(first_tool_call.arguments)
        self._execution_audit.update(
            {
                "selected_tool_name": first_tool_call.name,
                "selected_arguments": selected_arguments,
                "selected_sid": selected_arguments["sid_list"][0],
                "execution_attempted": True,
            }
        )
        with simple_timer("tool_calls", agent_data.metrics):
            tool_response, tool_reward, tool_result = await self._call_tool(
                first_tool_call, agent_data.tools_kwargs
            )

        result_has_status = isinstance(tool_result, dict) and "hit_count" in tool_result
        self._execution_audit.update(
            {
                "execution_completed": result_has_status,
                "lookup_hit_count": int(tool_result.get("hit_count", 0))
                if result_has_status
                else 0,
                "observation_present": bool((tool_response.text or "").strip()),
            }
        )

        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

        remaining = max(0, self.response_length - len(agent_data.response_mask))
        tool_response_text = _escape_reserved_tool_delimiters(
            tool_response.text or ""
        )

        def encode_tool_message(content: str) -> list[int]:
            message = {"role": "tool", "content": content}
            if self.processor is not None:
                rendered = self.processor.apply_chat_template(
                    [message],
                    add_generation_prompt=False,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                model_inputs = self.processor(
                    text=[rendered], images=None, return_tensors="pt"
                )
                return model_inputs.pop("input_ids").squeeze(0).tolist()
            token_ids = self.tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            return token_ids[len(self.system_prompt) :]

        def encode_tool_message_with_budget() -> tuple[list[int], str]:
            full_ids = encode_tool_message(tool_response_text)
            if len(full_ids) <= remaining:
                return full_ids, tool_response_text

            marker = "...(truncated)..."

            def clipped_content(characters_to_keep: int) -> str:
                head = (characters_to_keep * 3 + 3) // 4
                tail = characters_to_keep - head
                suffix = tool_response_text[-tail:] if tail else ""
                return tool_response_text[:head] + marker + suffix

            smallest_content = clipped_content(0)
            smallest_ids = encode_tool_message(smallest_content)
            if len(smallest_ids) > remaining:
                raise RuntimeError(
                    "response budget is too small to retain a parseable tool response"
                )

            low, high = 0, len(tool_response_text)
            best_content, best_ids = smallest_content, smallest_ids
            while low <= high:
                midpoint = (low + high) // 2
                candidate_content = clipped_content(midpoint)
                candidate_ids = encode_tool_message(candidate_content)
                if len(candidate_ids) <= remaining:
                    best_content, best_ids = candidate_content, candidate_ids
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            return best_ids, best_content

        (
            retained_tool_response_ids,
            retained_tool_response_text,
        ) = await self.loop.run_in_executor(None, encode_tool_message_with_budget)
        agent_data.prompt_ids += retained_tool_response_ids
        agent_data.response_mask += [0] * len(retained_tool_response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(retained_tool_response_ids)
        agent_data.user_turns += 1
        agent_data.messages.append(
            {"role": "tool", "content": retained_tool_response_text}
        )

        # Crucial difference from ToolAgentLoop: never transition back to
        # GENERATING after the observation.
        return AgentState.TERMINATED
