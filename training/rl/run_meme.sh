#!/usr/bin/env bash
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUSTOM_REWARD_FUNCTION_PATH="$RL_DIR/config/mem_evolver_reward_v9.py"
export AGENT_LOOP=meme_tool_agent
export AGENT_LOOP_CONFIG_PATH="$RL_DIR/config/meme_tool_agent_loop_answer3584.yaml"
export MAX_RESPONSE_LENGTH=6144 MAX_MODEL_LEN=10240 ROLLOUT_N=8
export TRAIN_BATCH_SIZE=16 LEARNING_RATE=3e-7 KL_COEF=0.003
export EXPERIMENT_NAME=meme_grpo
exec bash "$RL_DIR/run_train.sh" "$@"
