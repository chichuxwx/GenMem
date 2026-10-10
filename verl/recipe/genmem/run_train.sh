#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config/local_config_v33.sh"

if [[ -n "${EXPERIMENT_VARIANT_CONFIG:-}" ]]; then
    source "$EXPERIMENT_VARIANT_CONFIG"
fi

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must list four free GPUs}"
: "${DEERFLOW_EMAIL:?DEERFLOW_EMAIL must be set in .env or environment}"
: "${DEERFLOW_PASSWORD:?DEERFLOW_PASSWORD must be set in .env or environment}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES// /}"
N_GPUS_PER_NODE="${#GPU_ARRAY[@]}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
[[ "$N_GPUS_PER_NODE" -eq "$EXPECTED_GPU_COUNT" ]] || {
    echo "ERROR: expected $EXPECTED_GPU_COUNT GPUs; got $CUDA_VISIBLE_DEVICES" >&2
    exit 1
}

mkdir -p "$WANDB_DIR" "$ROLLOUT_DATA_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$RAY_TMPDIR"
export HYDRA_FULL_ERROR=1
ulimit -n 65535
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONFIG_PATH="$PROJECT_DIR/verl/trainer/config"
cd "$PROJECT_DIR"

RLSD_ARGS=()
if [[ "$LOSS_MODE" == "rlsd" ]]; then
    RLSD_ARGS=("+actor_rollout_ref.actor.rlsd_epsilon=$RLSD_EPSILON")
fi

TRAINING_STEP_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    TRAINING_STEP_ARGS=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi

"$TRAIN_PYTHON" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=_generated_ppo_trainer \
    reward_model.reward_manager="$REWARD_MANAGER" \
    custom_reward_function.path="$CUSTOM_REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$CUSTOM_REWARD_FUNCTION_NAME" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
    data.val_batch_size=4 \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS:-0}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-3e-7}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.003}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
    "${RLSD_ARGS[@]}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-6144}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.80}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.5}" \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.multi_stage_wake_up="${MULTI_STAGE_WAKE_UP:-True}" \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE:-True}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${AGENT_LOOP_MAX_ASSISTANT_TURNS:-4}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=12000 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.use_4d_mask="$USE_4D_MASK" \
    actor_rollout_ref.rollout.agent.spr1_parallel_search_num_paths="$SPR1_PARALLEL_SEARCH_NUM_PATHS" \
    actor_rollout_ref.rollout.agent.spr1_max_one_step_length="$SPR1_MAX_ONE_STEP_LENGTH" \
    actor_rollout_ref.rollout.agent.spr1_max_num_turns="$SPR1_MAX_NUM_TURNS" \
    actor_rollout_ref.rollout.agent.default_agent_loop="$AGENT_LOOP" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH:-}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS:-8}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode="${RESUME_MODE:-disable}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}" \
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-2}" \
    trainer.val_before_train=False \
    trainer.logger="${TRAINER_LOGGER:-[\"console\"]}" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-1}" \
    trainer.test_freq=-1 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$TEST_DATA_PATH" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    +ray_kwargs.ray_init.include_dashboard=False \
    "${TRAINING_STEP_ARGS[@]}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-3}" "$@"
