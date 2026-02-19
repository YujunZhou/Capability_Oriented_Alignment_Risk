
#!/bin/bash
# Self-grading GRPO training script
set -x

pkill -f "python.*main_ppo" || true;pkill -f "python.*main_dapo" || true;pkill -f "multiprocessing.spawn" || true;ray stop --force 2>/dev/null || true



echo "✅ Basic environment check passed"


echo "Starting self-grading GRPO training at: $(date)"
nvidia-smi

# Data paths
DATA_DIR=./data/self_grading

# Create log directories
mkdir -p ./logs/rollout_data
mkdir -p ./logs/validation_data
mkdir -p ./checkpoints/self_grading_grpo

# Record start time
START_TIME=$(date +%s)

# Parse CLI: -m model
# -m some_model => your-org/Advanced_Risk_Self_Grading_some_model
TASK_NAME="Self_Grading"
DEFAULT_MODEL_PATH="Qwen/Qwen3-4B-Base"
DEFAULT_USER_MODEL="Qwen3-4B-Base"
RUNTIME_MODEL_PATH="$DEFAULT_MODEL_PATH"
USER_MODEL="$DEFAULT_USER_MODEL"
USE_ORIGINAL_MODEL_PATH=false
USE_SFT_MODEL=false

canonicalize_model_name() {
  local raw="$1"
  local lowered="${raw,,}"
  case "$lowered" in
    llama)
      echo "meta-llama/Llama-3.1-8B-Instruct"
      ;;
    qwen3-4b-base)
      echo "Qwen/Qwen3-4B-Base"
      ;;
    qwen3-4b)
      echo "Qwen/Qwen3-4B"
      ;;
    *)
      echo "$raw"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      if [[ -z "${2:-}" ]]; then
        echo "Error: -m requires an argument" >&2
        exit 1
      fi
      USER_MODEL="$2"
      shift 2
      ;;
    --ori)
      USE_ORIGINAL_MODEL_PATH=true
      shift
      ;;
    --sft)
      USE_SFT_MODEL=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
done

CANONICAL_MODEL_PATH="$(canonicalize_model_name "$USER_MODEL")"

if [[ "$USE_ORIGINAL_MODEL_PATH" == true ]]; then
  RUNTIME_MODEL_PATH="$CANONICAL_MODEL_PATH"
elif [[ "$USE_SFT_MODEL" == true ]]; then
  RUNTIME_MODEL_PATH="${HF_ORG:-your-org}/SFT_Advanced_Risk_${TASK_NAME}_${USER_MODEL}"
else
  RUNTIME_MODEL_PATH="${HF_ORG:-your-org}/Advanced_Risk_${TASK_NAME}_${USER_MODEL}"
fi

if [[ "$USE_ORIGINAL_MODEL_PATH" == true ]]; then
  EXPERIMENT_SUFFIX="_ori"
elif [[ "$USE_SFT_MODEL" == true ]]; then
  EXPERIMENT_SUFFIX="_sft"
else
  EXPERIMENT_SUFFIX=""
fi
export EXPERIMENT_SUFFIX

echo "[Model] Using: $RUNTIME_MODEL_PATH"
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
. "$SCRIPT_DIR/_common_train_utils.sh"
# Ensure we run from repo root so python -m verl works and relative config paths resolve
WORK_DIR=$(cd "$SCRIPT_DIR/../.."; pwd)
cd "$WORK_DIR"

# Set temperature and experiment name
compute_train_temperature
unset EXPERIMENT_NAME
set_project_and_experiment_prefix "Secure_Self_Grading"

echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"

# Single-turn only (no SGLang multi-turn), use standard data filenames
ROLLOUT_ENGINE="vllm"
MULTI_TURN_ENABLE="False"
MULTI_TURN_MAX_ASSISTANT_TURNS="1"
MULTI_TURN_MAX_USER_TURNS="1"
INTERACTION_CONFIG_PATH=""

TRAIN_PATH=${DATA_DIR}/train.parquet
VAL_PATH=${DATA_DIR}/test.parquet

if [[ ! -f "$TRAIN_PATH" ]] || [[ ! -f "$VAL_PATH" ]]; then
  echo "  - Dataset files missing. Generating BBEH self-grading datasets..."
  python3 ./scripts/prepare_bbeh_self_grading.py --output_dir "$DATA_DIR" --train_n 4000 --test_n 100 --seed 42
fi

echo "  - Rollout engine: $ROLLOUT_ENGINE (single-step)"
echo "  - Using datasets: $TRAIN_PATH, $VAL_PATH"
# Run GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$VAL_PATH']" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.model.path=${RUNTIME_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ROLLOUT_ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    +trainer.enable_judge_rollout=True \
    +actor_rollout_ref.judge.max_input_length=4096 \
    +actor_rollout_ref.judge.max_response_length=10 \
    +actor_rollout_ref.judge.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console', 'wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.rollout_data_dir=./logs/rollout_data/self_grading_grpo/$USER_MODEL \
    trainer.validation_data_dir=./logs/validation_data/self_grading_grpo/$USER_MODEL \
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    custom_reward_function.path=./verl/utils/reward_score/bbeh_secure.py \
    custom_reward_function.name=compute_score \
    ${EXTRA_OVERRIDES:-} \

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Self-grading training completed. Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"

# Show saved data
echo "=== Saved Data Summary ==="
echo "Rollout data files:"
ls -la ./logs/rollout_data/ 2>/dev/null || echo "No rollout data found"
echo "Validation data files:"
ls -la ./logs/validation_data/ 2>/dev/null || echo "No validation data found"
echo "Checkpoints:"
ls -la ./checkpoints/self_grading_grpo/ 2>/dev/null || echo "No checkpoints found"


# Optional GPU monitoring after training
# Uncomment the following to enable GPU monitoring:

