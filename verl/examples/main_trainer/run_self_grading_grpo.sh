
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

# Parse CLI: -m model, -a audit probability, -t telling_audit
# -m Qwen-4B-Base => Qwen/Qwen-4B-Base
# -m Qwen-4B => Qwen/Qwen-4B
# -m llama => meta-llama/Llama-3.1-8B-Instruct
# -a 0.5 => audit probability 50%
# -t 0.5 => tell audit probability in judge prompt (50%)
RUNTIME_MODEL_PATH="Qwen/Qwen3-4B-Base"  # default
USER_MODEL_INPUT=""  # user-provided model name
AUDIT_PROB="0.1"  # default audit probability
TELLING_AUDIT="0"  # default: do not tell audit in prompt (0 = do not tell)

while getopts ":m:a:t:" opt; do
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
    a)
      AUDIT_PROB="$OPTARG"
      ;;
    t)
      TELLING_AUDIT="$OPTARG"
      ;;
  esac
done
echo "[Model] Using: $RUNTIME_MODEL_PATH"
echo "[Audit prob] audit_prob: $AUDIT_PROB"
echo "[Telling audit] telling_audit: $TELLING_AUDIT"
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
. "$SCRIPT_DIR/_common_train_utils.sh"
# Ensure we run from repo root so python -m verl works and relative config paths resolve
WORK_DIR=$(cd "$SCRIPT_DIR/../.."; pwd)
cd "$WORK_DIR"

# Set temperature and experiment name
compute_train_temperature
set_project_and_experiment_prefix "Self_Grading"

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
    +trainer.audit_probability=$AUDIT_PROB \
    +trainer.telling_audit=$TELLING_AUDIT \
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
    custom_reward_function.path=./verl/utils/reward_score/bbeh.py \
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

