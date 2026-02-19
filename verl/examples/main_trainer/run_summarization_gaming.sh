#!/bin/bash
# Proxy Gaming Experiment: Summarization with ROUGE Reward
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# Data paths
DATA_DIR=./data/proxy_gaming
TRAIN_PATH=${DATA_DIR}/train.parquet
TEST_PATH=${DATA_DIR}/test.parquet

# Check if data exists
if [ ! -f "$TRAIN_PATH" ]; then
    echo "Preparing dataset..."
    python3 scripts/prepare_proxy_gaming_dataset.py \
        --output_dir $DATA_DIR
fi

echo "Start Proxy Gaming Training..."
echo "Training data: $TRAIN_PATH"
echo "Testing data: $TEST_PATH"

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
echo "[Model] Using: $RUNTIME_MODEL_PATH"
. "$SCRIPT_DIR/_common_train_utils.sh"

# Set temperature and experiment name
compute_train_temperature
set_project_and_experiment_prefix "Summarization"

echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"

# Run PPO training (params aligned with run_situational_awareness_grpo.sh)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$TEST_PATH']" \
    data.train_batch_size=96 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console', 'wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.rollout_data_dir=./logs/rollout_data/proxy_gaming/$USER_MODEL \
    trainer.validation_data_dir=./logs/validation_data/proxy_gaming/$USER_MODEL \
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    custom_reward_function.path=verl/utils/reward_score/proxy_gaming.py \
    custom_reward_function.name=compute_score \
    ${EXTRA_OVERRIDES:-}

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Training completed! Total duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
