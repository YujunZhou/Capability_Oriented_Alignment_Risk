#!/bin/bash
# run_situational_awareness_grpo_verbose.sh
set -x

# More thorough process cleanup
pkill -f "python.*main_ppo" || true
pkill -f "python.*main_dapo" || true
pkill -f "python.*verl" || true
pkill -f "multiprocessing.spawn" || true
pkill -f "ray" || true
ray stop --force 2>/dev/null || true
sleep 5  # Wait for processes to fully exit

export OPENAI_API_KEY=""
export DEEPINFRA_API_KEY=""

# Set LLM evaluation batch parameters
export LLM_JUDGE_BATCH_SIZE=64
export LLM_JUDGE_MAX_CONCURRENT_REQUESTS=32

# Print GPU info
echo "Starting training at: $(date)"
nvidia-smi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$WORK_DIR"

# Data paths
DATA_DIR=./data/situational_awareness

# Record start time
START_TIME=$(date +%s)

# Parse CLI: -m model
# -m Qwen-4B-Base => Qwen/Qwen-4B-Base
# -m Qwen-4B => Qwen/Qwen-4B
# -m llama => meta-llama/Llama-3.1-8B-Instruct
RUNTIME_MODEL_PATH="Qwen/Qwen3-4B-Base"  # default
USER_MODEL_INPUT=""  # user-provided model name

while getopts ":m:" opt; do
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
  esac
done
# Always use standard datasets
TRAIN_PATH=${DATA_DIR}/train.parquet
VAL_PATH=${DATA_DIR}/test.parquet
echo "[Data] Using standard dataset: $TRAIN_PATH, $VAL_PATH"

echo "[Model] Using: $RUNTIME_MODEL_PATH"
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
. "$SCRIPT_DIR/_common_train_utils.sh"

# Set temperature and experiment name
compute_train_temperature
set_project_and_experiment_prefix "Situation_Aware"

# Signal handler
cleanup() {
    echo "Received termination signal, cleaning up..."
    pkill -f "python.*verl" || true
    ray stop --force || true
    exit 1
}
trap cleanup SIGTERM SIGINT

echo "Starting Situational Awareness GRPO training..."
echo "Process PID: $$"
echo "Current time: $(date)"

# Check system resources and limits
echo "=== System resource check ==="
echo "Disk space:"
df -h . || true
echo "Memory:"
free -h || true
echo "System limits:"
ulimit -a | grep -E "(time|memory|virtual|file)" || true
echo "SLURM job info:"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"
echo "SLURM_JOB_TIMELIMIT: ${SLURM_JOB_TIMELIMIT:-not set}"
echo "====================="

# Set env for more verbose errors
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONUNBUFFERED=1

# Run training - use 2 GPUs
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$VAL_PATH']" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.model.path=${RUNTIME_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.rollout_data_dir=./logs/rollout_data/situational_awareness/$USER_MODEL \
    trainer.validation_data_dir=./logs/validation_data/situational_awareness/$USER_MODEL \
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    custom_reward_function.path=./verl/utils/reward_score/llm_judge.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=batch \
    ${EXTRA_OVERRIDES:-}

# Capture training process exit status
TRAINING_EXIT_CODE=$?
echo "Training process exited with code: $TRAINING_EXIT_CODE"
echo "Training end time: $(date)"

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Training exited abnormally with code: $TRAINING_EXIT_CODE"
    echo "Recent Ray processes:"
    ps aux | grep ray | head -5 || true
    echo "Recent Python processes:"
    ps aux | grep python.*verl | head -5 || true
fi

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Training completed. Total duration: $DURATION seconds ($(($DURATION / 60)) minutes)"

