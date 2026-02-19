#!/bin/bash
# Reward Tampering Experiment: Code Execution with Vulnerable Evaluation
set -euo pipefail
# set -x

# Set random seed for reproducibility
export PYTHONHASHSEED=42
# export WANDB_API_KEY="09208978a1b95488af0ee5faf16fa729333b8718"

# Change to repo root and set PYTHONPATH
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/_common_train_utils.sh"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Data and log dirs (relative to repo root for reproducibility)
DATA_DIR=./data/reward_tampering
ROLLOUT_DIR=./logs/rollout_data/reward_tampering/

# Log dir (for saving detailed output and errors)
LOG_DIR=./logs/reward_tampering
mkdir -p "$LOG_DIR"


echo "[Env] REPO_ROOT=$REPO_ROOT"
echo "[Env] PYTHONPATH=$PYTHONPATH"

echo "Starting Reward Tampering training experiment..."
echo "train: $TRAIN_PATH"
echo "test: $TEST_PATH"

# Reward computation parallelism (conservative default to avoid hangs and high resource use)
# Note: In NaiveRewardManager, when REWARD_PARALLEL_WORKERS>1 and data_source is "reward_tampering_code",
# the parallel path is used; that path does NOT call custom_reward_function but uses built-in reward_tampering.py.
# For strict custom reward, set REWARD_PARALLEL_WORKERS=1.
if [[ -z "${REWARD_PARALLEL_WORKERS:-}" ]]; then
  # Default to 1: avoid bypassing custom_reward_function and reduce memory risk from concurrency
  export REWARD_PARALLEL_WORKERS=1
fi
export REWARD_PARALLEL_CHUNKSIZE=${REWARD_PARALLEL_CHUNKSIZE:-4}

if [[ "${REWARD_PARALLEL_WORKERS}" -gt 1 ]]; then
  echo "[Note] REWARD_PARALLEL_WORKERS=${REWARD_PARALLEL_WORKERS} enables parallel path; implementation uses built-in reward_tampering.py, not custom_reward_function."
  echo "       For exact custom reward behavior, set REWARD_PARALLEL_WORKERS=1."
fi

# Unified timeout settings to avoid conflicting timeouts
# Main code execution timeout (seconds)
export RT_PER_TEST_TIMEOUT_SEC=${RT_PER_TEST_TIMEOUT_SEC:-10.0}
# Per-sample total timeout (seconds): leave empty for scorer to use per_test_timeout * num_tests
export RT_TOTAL_TIMEOUT_SEC=${RT_TOTAL_TIMEOUT_SEC:-}
# Subprocess cleanup timeout (seconds)
export RT_SUBPROCESS_CLEANUP_TIMEOUT=${RT_SUBPROCESS_CLEANUP_TIMEOUT:-3.0}
# Thread execution timeout (seconds)
export RT_THREAD_TIMEOUT_SEC=${RT_THREAD_TIMEOUT_SEC:-8.0}

# Limit malicious/abnormal code from exhausting memory in a single subprocess (see reward_tampering_xml.py)
# Subprocess memory hard limit (MB)
export RT_MEM_LIMIT_MB=${RT_MEM_LIMIT_MB:-1024}
# Subprocess CPU time hard limit (seconds, 0=disabled)
export RT_CPU_LIMIT_SEC=${RT_CPU_LIMIT_SEC:-0}
# Max file size subprocess may write (MB)
export RT_FSIZE_LIMIT_MB=${RT_FSIZE_LIMIT_MB:-8}
# Max stdout/stderr capture chars (avoid memory spike from print flood)
export RT_MAX_CAPTURE_CHARS=${RT_MAX_CAPTURE_CHARS:-200000}
# Subprocess import allowlist extras (comma-separated). Empty = default (sys/math/re etc.)
export RT_IMPORT_ALLOWLIST=${RT_IMPORT_ALLOWLIST:-}

# Print prompt indices used during training (must be implemented in Python training code)
export RT_PRINT_PROMPT_INDEX=${RT_PRINT_PROMPT_INDEX:-true}

# Process monitor and cleanup
cleanup_processes() {
    echo "Cleaning up any leftover Python processes..."
    # Find and kill possibly stuck processes
    pkill -f "python.*reward_tampering" 2>/dev/null || true
    pkill -f "python.*main_ppo" 2>/dev/null || true
    sleep 2
}

# Cleanup on exit
trap cleanup_processes EXIT INT TERM

# Record start time
START_TIME=$(date +%s)

# Parse CLI: -m model, -n no_exploit
# Rules:
# -m Qwen-4B-Base => Qwen/Qwen-4B-Base
# -m Qwen-4B => Qwen/Qwen-4B
# -m llama => meta-llama/Llama-3.1-8B-Instruct
# -n => use no exploit dataset
RUNTIME_MODEL_PATH="Qwen/Qwen3-4B-Base"  # default model from script
USER_MODEL_INPUT=""  # user-provided model name
NO_EXPLOIT=false  # default: do not use no exploit dataset

while getopts ":m:n" opt; do
  case $opt in
    m)
      USER_MODEL_INPUT="$OPTARG"
      USER_MODEL="$OPTARG"
      if [[ "$USER_MODEL" == "llama" || "$USER_MODEL" == "Llama" ]]; then
        RUNTIME_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
      elif [[ "$USER_MODEL" == "Llama3.2" || "$USER_MODEL" == "llama3.2" ]]; then
        RUNTIME_MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
      elif [[ "$USER_MODEL" == Qwen3-* ]]; then
        RUNTIME_MODEL_PATH="Qwen/${USER_MODEL}"
      elif [[ "$USER_MODEL" == *"/"* ]]; then
        RUNTIME_MODEL_PATH="$USER_MODEL"
      else
        RUNTIME_MODEL_PATH="$USER_MODEL"
      fi
      ;;
    n)
      NO_EXPLOIT=true
      ;;
  esac
done

# Select dataset by no_exploit flag
if [ "$NO_EXPLOIT" = true ]; then
    TRAIN_PATH=${DATA_DIR}/train_no_exploit.parquet
    TEST_PATH=${DATA_DIR}/test.parquet
    echo "[Data] Using no exploit train: $TRAIN_PATH, test: $TEST_PATH"
else
    TRAIN_PATH=${DATA_DIR}/train.parquet
    TEST_PATH=${DATA_DIR}/test.parquet
    echo "[Data] Using standard dataset: $TRAIN_PATH, $TEST_PATH"
fi

echo "[Model] Using: $RUNTIME_MODEL_PATH"
echo "[No Exploit] Using no exploit dataset: $NO_EXPLOIT"

# Project and experiment name (overridable via env)
PROJECT_NAME=${PROJECT_NAME:-Advanced_Risk}

# Set temperature and experiment name
compute_train_temperature
if [ "$NO_EXPLOIT" = true ]; then
    set_project_and_experiment_prefix "Reward_Tampering_NoExploit"
else
    set_project_and_experiment_prefix "Reward_Tampering"
fi

# Stage 1: train in vulnerable environment
echo "=== Stage 1: Vulnerable environment training ==="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TEST_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    reward_model.reward_manager=naive \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.model.path=${RUNTIME_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12800 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.rollout_data_dir=./logs/rollout_data/reward_tampering/$USER_MODEL \
    trainer.validation_data_dir=./logs/validation_data/reward_tampering/$USER_MODEL \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    ray_init.num_cpus=32 \
    custom_reward_function.path="$REPO_ROOT/verl/utils/reward_score/reward_tampering_xml.py" \
    custom_reward_function.name=compute_score \
    ${EXTRA_OVERRIDES:-}


# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=== Training Complete ==="
echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo ""

# After training: for each step, print 2 random samples and re-evaluate with vulnerable/secure scorer
(
python3 scripts/inspect_reward_tampering_run.py \
  --train_parquet "$TRAIN_PATH" \
  --test_parquet "$TEST_PATH" \
  --rollout_dir "$ROLLOUT_DIR" \
  --per_step_samples 2 \
) > "$LOG_DIR/inspect.out" 2> "$LOG_DIR/inspect.err" || true

echo "Next step: run validation to detect reward tampering"