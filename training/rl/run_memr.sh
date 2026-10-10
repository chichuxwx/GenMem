#!/usr/bin/env bash
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUSTOM_REWARD_FUNCTION_PATH="$RL_DIR/config/mem_retriever_reward_v33_nonnegative_v2.py"
export AGENT_LOOP=think_capped_tool_agent
export AGENT_LOOP_CONFIG_PATH="$RL_DIR/config/think_capped_agent_loop.yaml"
export MAX_RESPONSE_LENGTH=4096 MAX_MODEL_LEN=8192 ROLLOUT_N=8
export EXPERIMENT_NAME=memr_grpo
exec bash "$RL_DIR/run_train.sh" "$@"
