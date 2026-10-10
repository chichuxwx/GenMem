# Modified for GenMem; see third_party/verl/README.md for provenance.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from verl.workers.actor.decouple_layer import *

from verl.spr1_special_tokens.recognition import TokenRecognizer

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_tokenizer():
    import os
    tokenizer_path = os.getenv("TOKENIZER_PATH")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Initialize TokenRecognizer
        print("debug load_tokenizer")
        try:
            self.recognizer = TokenRecognizer(tokenizer=load_tokenizer())
            print(f"TokenRecognizer initialized successfully with tokenizer.")
        except Exception as e:
            if torch.distributed.get_rank() == 0:
                print(f"[Warning] Failed to initialize TokenRecognizer: {e}")
            self.recognizer = None

        # print("TokenRecognizer [debug] ===============================")
        # print("self.tokenizer=", self.tokenizer)
        # print("TokenRecognizer [debug] ===============================")
        # if TokenRecognizer is not None and hasattr(self.config, 'model_config') and self.config.model_config.tokenizer is not None:
        #      try:
        #         self.recognizer = TokenRecognizer(tokenizer=self.config.model_config.tokenizer)
        #         if torch.distributed.get_rank() == 0:
        #             print(f"TokenRecognizer initialized successfully with tokenizer.")
        #      except Exception as e:
        #         if torch.distributed.get_rank() == 0:
        #             print(f"[Warning] Failed to initialize TokenRecognizer: {e}")
        # else:
        #      if torch.distributed.get_rank() == 0:
        #         print(f"[Warning] TokenRecognizer or Tokenizer not available. Action/Args masks will be empty.")
        #         print(f"[Warning] hasattr(self.config, 'model_config')={hasattr(self.config, 'model_config')}")
        #         print(f"[Warning] self.config.model_config.tokenizer={self.config.model_config.tokenizer}")

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # [UPDATE] use 2d mask to construct 4d mask
                # if micro_batch.get("attention_mask_4d", None) is not None:
                #     attention_mask = micro_batch["attention_mask_4d"]

                if "2d_attention_mask" in micro_batch.keys():
                    batch_size = len(input_ids)
                    max_len = len(input_ids[0])
                    bool_mask = torch.tril(torch.ones(batch_size, max_len, max_len, dtype=torch.bool))
                    spans = micro_batch["2d_attention_mask"]

                    for i in range(batch_size):
                        spansi = spans[i]
                        print(f"[zzx debug] create 4d-attn-mask with 2d-attn-mask = {spansi}")
                        for l, r, L, R in spansi:
                            if l >= L:
                                bool_mask[i][L:l][l:r+1] = False
                                bool_mask[i][l:r+1][L:l] = False
                                bool_mask[i][r+1:R+1][l:r+1] = False
                                bool_mask[i][l:r+1][r+1:R+1] = False
                            else:
                                bool_mask[i][l:r+1][L:R+1] = False
                                bool_mask[i][L:R+1][l:r+1] = False

                    attention_mask = torch.zeros_like(bool_mask, dtype=torch.bfloat16)
                    attention_mask.masked_fill_(~bool_mask, float("-inf"))
                    attention_mask = attention_mask.unsqueeze(1)

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if "2d_attention_mask" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("2d_attention_mask")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto) -> dict:
        is_decoupling = os.environ.get("DECOUPLING", "0") == "1"
        if is_decoupling:
            return self.decouple_update_policy(data)
        else:
            return self.old_update_policy(data)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def decouple_update_policy(self, data: DataProto) -> dict:
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        ### NOTE SP-R1
        # build action / args parameter groups
        print(self.actor_module.__repr__())
# FullyShardedDataParallel( [repeated 2x across cluster]
#          (_fsdp_wrapped_module): Qwen2DecoderLayer( [repeated 4x across cluster]
#      (model): Qwen2Model( [repeated 2x across cluster]
#       (embed_tokens): Embedding(151936, 2048, padding_idx=151643) [repeated 2x across cluster]
#      (layers): ModuleList( [repeated 2x across cluster]
#         (0-35): 36 x FullyShardedDataParallel( [repeated 2x across cluster]
#            (self_attn): Qwen2Attention( [repeated 2x across cluster]
#              (q_proj): Linear(in_features=2048, out_features=2048, bias=True) [repeated 2x across cluster]
#                (k_proj): Linear(in_features=2048, out_features=256, bias=True) [repeated 2x across cluster]
#               (v_proj): Linear(in_features=2048, out_features=256, bias=True) [repeated 2x across cluster]
#               (o_proj): Linear(in_features=2048, out_features=2048, bias=False) [repeated 2x across cluster]
#  ) [repeated 16x across cluster]
#              (mlp): Qwen2MLP( [repeated 2x across cluster]
#                (gate_proj): Linear(in_features=2048, out_features=11008, bias=False) [repeated 2x across cluster]
#                (up_proj): Linear(in_features=2048, out_features=11008, bias=False) [repeated 2x across cluster]
#                (down_proj): Linear(in_features=11008, out_features=2048, bias=False) [repeated 2x across cluster]
#                (act_fn): SiLUActivation() [repeated 2x across cluster]
#              (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06) [repeated 2x across cluster]
#              (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06) [repeated 2x across cluster]
#        (norm): Qwen2RMSNorm((2048,), eps=1e-06) [repeated 2x across cluster]
#        (rotary_emb): Qwen2RotaryEmbedding() [repeated 2x across cluster]
#      (lm_head): Linear(in_features=2048, out_features=151936, bias=False) [repeated 2x across cluster]

        action_params, args_params = generate_grad_control_dicts(
            model=self.actor_module,
            param_importance_path=str(Path(__file__).resolve().parent / "neuron" / "neuron_active_ratio.json"),
            top_k=8
        )
        ### NOTE SP-R1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss += kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    ### NOTE SP-R1
                    # ---- build action / args masks ----
                    action_mask, args_mask = build_action_args_masks_from_model_inputs(model_inputs, recognizer=self.recognizer)
                    loss_action = agg_loss(policy_loss, loss_mask=response_mask*action_mask, loss_agg_mode=loss_agg_mode)
                    loss_args = agg_loss(policy_loss, loss_mask=response_mask*args_mask, loss_agg_mode=loss_agg_mode)

                    # # 修复闭包迟绑定：用默认参数固化param的值
                    # def hook_factory(param):
                    #     def hook(grad, param=param):  # 新增默认参数param=param，固化当前值
                    #         # if param.numel() == 0:  # size=0 shard 不处理
                    #         #     return grad
                    #         return torch.zeros_like(param)
                    #     return hook

                    # # action梯度钩子
                    # handles = []
                    # for p in self.actor_module.parameters():
                    #     if p not in action_params:
                    #         handles.append(p.register_hook(hook_factory(p)))

                    # loss_action.backward(retain_graph=True)
                    # for h in handles:
                    #     h.remove()

                    # # args梯度钩子
                    # handles = []
                    # for p in self.actor_module.parameters():
                    #     if p not in args_params:
                    #         handles.append(p.register_hook(hook_factory(p)))

                    # loss_args.backward()
                    # for h in handles:
                    #     h.remove()

                    ### NOTE SP-R1 (soft update, single backward)

                    policy_loss.backward()

                    import torch.distributed as dist
                    u = torch.rand((), device=policy_loss.device)
                    if dist.is_available() and dist.is_initialized():
                        dist.broadcast(u, src=0)
                    u = float(u.item())

                    hi, lo = 2.0, 0.5

                    if u > 0.5:
                        action_scale, args_scale = hi, lo
                    else:
                        action_scale, args_scale = lo, hi

                    other_scale = 0.1

                    def _to_param_id_set(x):
                        if x is None:
                            return set()
                        if isinstance(x, dict):
                            it = x.keys()
                        else:
                            it = x
                        return {id(p) for p in it}

                    action_ids = _to_param_id_set(action_params)
                    args_ids = _to_param_id_set(args_params)

                    for p in self.actor_module.parameters():
                        g = p.grad
                        if g is None:
                            continue
                        if g.numel() == 0:
                            continue

                        pid = id(p)
                        if pid in action_ids:
                            g.mul_(action_scale)
                        elif pid in args_ids:
                            g.mul_(args_scale)
                        else:
                            g.mul_(other_scale)

                    micro_batch_metrics["actor/sp_r1_u"] = u
                    micro_batch_metrics["actor/sp_r1_action_scale"] = action_scale
                    micro_batch_metrics["actor/sp_r1_args_scale"] = args_scale
                    micro_batch_metrics["actor/sp_r1_other_scale"] = other_scale

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics

    # ------------------------------------------------------------------ #
    # RLSD helper: build privileged_input_ids from extra_info              #
    # ------------------------------------------------------------------ #
    def _build_privileged_inputs(self, data: DataProto, temperature: float) -> DataProto:
        """Tokenise privileged prompts and build privileged_input_ids tensor.

        For each sample, the privileged prompt is stored in
        ``non_tensor_batch["extra_info"][i]["privileged_prompt"]``.
        We tokenise it, then concatenate with the corresponding response tokens
        to obtain a full sequence whose response log-probs equal
        log p_θ(y_t | q, z, y_{<t}).

        The original ``input_ids`` structure is:
            [PAD ... | prompt tokens | response tokens]
        We build privileged_input_ids as:
            [PAD ... | privileged_prompt tokens | response tokens]
        padded to the same total length from the right.
        """
        tokenizer = load_tokenizer()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        batch_size = data.batch["input_ids"].shape[0]
        seq_len = data.batch["input_ids"].shape[1]
        device = data.batch["input_ids"].device

        extra_infos = data.non_tensor_batch.get("extra_info", [{}] * batch_size)

        # Original prompt lengths (from attention_mask)
        attn_mask = data.batch["attention_mask"]  # [B, seq_len]
        responses = data.batch["responses"]       # [B, resp_len]
        resp_len = responses.shape[1]
        prompt_len = seq_len - resp_len

        action_input_ids_list = []
        action_attn_mask_list = []
        answer_input_ids_list = []
        answer_attn_mask_list = []
        answer_lengths = []
        recognizer = TokenRecognizer(tokenizer)

        for i in range(batch_size):
            ei = extra_infos[i] if i < len(extra_infos) else {}
            action_prompt = (
                ei.get("rlsd_action_prompt", ei.get("privileged_prompt"))
                if isinstance(ei, dict)
                else None
            )
            answer_prompt = (
                ei.get("rlsd_answer_prompt") if isinstance(ei, dict) else None
            )

            if action_prompt is None:
                # Fallback: use original prompt tokens unchanged
                action_input_ids_list.append(data.batch["input_ids"][i].clone())
                action_attn_mask_list.append(attn_mask[i].clone())
                answer_input_ids_list.append(data.batch["input_ids"][i].clone())
                answer_attn_mask_list.append(attn_mask[i].clone())
                answer_lengths.append(0)
                continue

            # Tokenise the privileged prompt (no padding, truncate if needed)
            enc = tokenizer.apply_chat_template(
                action_prompt,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
            if isinstance(enc, torch.Tensor):
                priv_prompt_ids = enc[0].to(device)
            else:
                priv_prompt_ids = torch.tensor(enc, device=device)

            # Truncate privileged prompt if too long to fit response
            max_priv_prompt_len = seq_len - resp_len
            if priv_prompt_ids.shape[0] > max_priv_prompt_len:
                priv_prompt_ids = priv_prompt_ids[-max_priv_prompt_len:]

            # Get valid response tokens (non-padded)
            resp_tokens = responses[i]  # [resp_len]

            # Build full sequence: [PAD ... | priv_prompt | resp_tokens]
            # Left-pad prompt to match original prompt length
            priv_p_len = priv_prompt_ids.shape[0]
            pad_len = max_priv_prompt_len - priv_p_len
            pad_ids = torch.full(
                (pad_len,), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            full_ids = torch.cat([pad_ids, priv_prompt_ids, resp_tokens], dim=0)[:seq_len]

            # Build attention mask
            full_mask = torch.zeros(seq_len, dtype=torch.long, device=device)
            full_mask[pad_len : pad_len + priv_p_len + resp_len] = 1

            action_input_ids_list.append(full_ids)
            action_attn_mask_list.append(full_mask)

            # For answer RLSD, do not replay the sampled tool call or sampled
            # observation.  Instead, condition only on the training-only
            # correct SID/observation context and append the sampled answer
            # span as the sole response suffix.
            intervals = recognizer.find_intervals(
                resp_tokens.tolist(), "<answer>", "</answer>", include_boundaries=True
            )
            if answer_prompt is None or not intervals:
                answer_input_ids_list.append(data.batch["input_ids"][i].clone())
                answer_attn_mask_list.append(attn_mask[i].clone())
                answer_lengths.append(0)
                continue
            answer_start, answer_end = intervals[-1]
            answer_tokens = resp_tokens[answer_start:answer_end]
            answer_len = int(answer_tokens.shape[0])
            answer_enc = tokenizer.apply_chat_template(
                answer_prompt,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
            answer_prompt_ids = answer_enc[0].to(device) if isinstance(answer_enc, torch.Tensor) else torch.tensor(answer_enc, device=device)
            if answer_prompt_ids.shape[0] > max_priv_prompt_len:
                answer_prompt_ids = answer_prompt_ids[-max_priv_prompt_len:]
            answer_p_len = int(answer_prompt_ids.shape[0])
            answer_pad_len = max_priv_prompt_len - answer_p_len
            response_pad_len = resp_len - answer_len
            packed_ids = torch.cat(
                [
                    torch.full((answer_pad_len,), tokenizer.pad_token_id, dtype=torch.long, device=device),
                    answer_prompt_ids,
                    torch.full((response_pad_len,), tokenizer.pad_token_id, dtype=torch.long, device=device),
                    answer_tokens,
                ]
            )[:seq_len]
            packed_mask = torch.zeros(seq_len, dtype=torch.long, device=device)
            packed_mask[answer_pad_len : answer_pad_len + answer_p_len] = 1
            packed_mask[seq_len - answer_len :] = 1
            answer_input_ids_list.append(packed_ids)
            answer_attn_mask_list.append(packed_mask)
            answer_lengths.append(answer_len)

        data.batch["rlsd_action_input_ids"] = torch.stack(action_input_ids_list, dim=0)
        data.batch["rlsd_action_attention_mask"] = torch.stack(action_attn_mask_list, dim=0)
        data.batch["rlsd_answer_input_ids"] = torch.stack(answer_input_ids_list, dim=0)
        data.batch["rlsd_answer_attention_mask"] = torch.stack(answer_attn_mask_list, dim=0)
        data.batch["rlsd_answer_lengths"] = torch.tensor(answer_lengths, device=device)
        return data

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def old_update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # RLSD: include privileged inputs (q+z) if pre-tokenised
        _use_rlsd = self.config.policy_loss.get("loss_mode", "vanilla") == "rlsd"
        if _use_rlsd:
            for _k in (
                "rlsd_action_input_ids", "rlsd_action_attention_mask",
                "rlsd_answer_input_ids", "rlsd_answer_attention_mask",
                "rlsd_answer_lengths",
            ):
                if _k in data.batch.keys():
                    select_keys.append(_k)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # RLSD needs extra_info (contains privileged_prompt) when building
        # privileged inputs on-the-fly from the tokenizer.
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if _use_rlsd and "extra_info" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("extra_info")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # RLSD: tokenise privileged prompts if not already in batch
        if _use_rlsd and "rlsd_action_input_ids" not in data.batch.keys():
            data = self._build_privileged_inputs(data, temperature)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    print("input_ids shape", model_inputs["input_ids"].shape)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # -------------------------------------------------- #
                    # RLSD: privileged forward pass                        #
                    # -------------------------------------------------- #
                    # RLSD uses causally distinct privileged contexts.  Action
                    # weights see only the correct SID hint.  Answer weights
                    # see the correct SID and corresponding tool observation;
                    # the sampled lookup observation is never replayed there.
                    action_privileged_log_prob = None
                    answer_privileged_log_prob = None
                    action_span_mask = None
                    answer_span_mask = None
                    if loss_mode == "rlsd":
                        action_ids = model_inputs.get("rlsd_action_input_ids")
                        action_mask = model_inputs.get("rlsd_action_attention_mask")
                        answer_ids = model_inputs.get("rlsd_answer_input_ids")
                        answer_mask = model_inputs.get("rlsd_answer_attention_mask")
                        if action_ids is not None and action_mask is not None:
                            action_inputs = dict(model_inputs)
                            action_inputs["input_ids"] = action_ids
                            action_inputs["attention_mask"] = action_mask
                            with torch.no_grad():
                                _, action_privileged_log_prob = self._forward_micro_batch(
                                    action_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                )
                        if answer_ids is not None and answer_mask is not None:
                            answer_inputs = dict(model_inputs)
                            answer_inputs["input_ids"] = answer_ids
                            answer_inputs["attention_mask"] = answer_mask
                            with torch.no_grad():
                                _, packed_answer_log_prob = self._forward_micro_batch(
                                    answer_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                )
                        else:
                            logger.warning(
                                "[RLSD] loss_mode=rlsd but privileged contexts not found "
                                "in batch; falling back to vanilla GRPO weights."
                            )

                        # Build RLSD span mask: active within <tool_call>...</tool_call>
                        # and <answer>...</answer> spans.
                        # <tool_call> covers the SID selection decision;
                        # <answer> covers the experience organisation output.
                        # When neither exists the mask stays all-zero → vanilla GRPO.
                        if self.recognizer is not None:
                            from verl.spr1_special_tokens.recognition import DEFAULT_ACTION_CONFIG
                            input_ids_cpu = model_inputs["input_ids"].cpu()
                            bsz, seq_len = input_ids_cpu.shape
                            resp_len = response_mask.shape[1]
                            tool_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
                            ans_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
                            input_ids_list = input_ids_cpu.tolist()
                            for b_idx, seq_ids in enumerate(input_ids_list):
                                for span_name, span_mask in (
                                    ("tool_call", tool_mask),
                                    ("answer", ans_mask),
                                ):
                                    start_token, end_token = DEFAULT_ACTION_CONFIG[span_name]
                                    intervals = self.recognizer.find_intervals(
                                        seq_ids, start_token, end_token, include_boundaries=True
                                    )
                                    for s, e in intervals:
                                        span_mask[b_idx, s:e] = True
                            action_span_mask = tool_mask[:, -resp_len:].to(device=response_mask.device)
                            answer_span_mask = ans_mask[:, -resp_len:].to(device=response_mask.device)
                            if answer_ids is not None and answer_mask is not None:
                                answer_privileged_log_prob = torch.zeros_like(log_prob)
                                lengths = model_inputs["rlsd_answer_lengths"].tolist()
                                for b_idx, answer_len in enumerate(lengths):
                                    if answer_len <= 0:
                                        continue
                                    original = torch.nonzero(answer_span_mask[b_idx], as_tuple=False).flatten()
                                    take = min(int(answer_len), int(original.numel()))
                                    answer_privileged_log_prob[b_idx, original[-take:]] = (
                                        packed_answer_log_prob[b_idx, -take:]
                                    )

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    # rlsd -> verl.trainer.ppo.core_algos.compute_policy_loss_rlsd
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Only RLSD accepts the privileged-context arguments.
                    # Vanilla/GPG/clip-cov loss functions use the common PPO kwargs.
                    loss_kwargs = {
                        "old_log_prob": old_log_prob,
                        "log_prob": log_prob,
                        "advantages": advantages,
                        "response_mask": response_mask,
                        "loss_agg_mode": loss_agg_mode,
                        "config": self.config,
                        "rollout_is_weights": rollout_is_weights,
                    }
                    if loss_mode == "rlsd":
                        loss_kwargs.update(
                            action_privileged_log_prob=action_privileged_log_prob,
                            answer_privileged_log_prob=answer_privileged_log_prob,
                            action_span_mask=action_span_mask,
                            answer_span_mask=answer_span_mask,
                        )

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(**loss_kwargs)
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
