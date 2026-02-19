#!/bin/bash

# Common utilities for main_trainer scripts

# Basic runtime env
export TOKENIZERS_PARALLELISM=true

# Global variable to track secure mode
USE_SECURE_REWARD=false

# Function to parse secure mode parameter
parse_secure_mode() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --secure|-s)
        USE_SECURE_REWARD=true
        echo "[Secure mode] Using secure reward function"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
  export USE_SECURE_REWARD
}

compute_train_temperature() {
  if [[ "${USER_MODEL:-}" == "llama" || "${USER_MODEL:-}" == "Llama" ]]; then
    TRAIN_TEMPERATURE=0.6
  else
    TRAIN_TEMPERATURE=1.0
  fi
}

set_project_and_experiment_prefix() {
  local prefix="$1"
  PROJECT_NAME=${PROJECT_NAME:-Advanced_Risk_new2}
  MODEL_TAG=${USER_MODEL:-${USER_MODEL_INPUT:-Qwen3-4B-Base}}
  local suffix="${EXPERIMENT_SUFFIX:-}"
  EXPERIMENT_NAME=${EXPERIMENT_NAME:-${prefix}_${MODEL_TAG}${suffix}}
}

post_merge_and_eval_verl() {
  local model_dir="./checkpoints/${PROJECT_NAME}"
  local run_name="${EXPERIMENT_NAME}"
  local eval_dir="./eval_results"
  local eval_script="${THIS_DIR:-$(cd "$(dirname "$0")"; pwd)}/eval_merged_model.sh"

  mkdir -p "$eval_dir"

  local latest_ckpt=""
  if [ -d "$model_dir/$run_name" ]; then
    latest_ckpt=$(find "$model_dir/$run_name" -name "global_step_*" -type d | sort -V | tail -n 1)
  fi
  if [ -z "$latest_ckpt" ]; then
    echo "[WARN] No checkpoint found under $model_dir/$run_name; skip merge & eval"
    return 0
  fi

  echo "Latest checkpoint: $latest_ckpt"
  local actor_dir="$latest_ckpt/actor"
  local target_dir="./models/${EXPERIMENT_NAME}"
  local abs_target_dir
  mkdir -p "$target_dir"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$actor_dir" \
    --target_dir "$target_dir" \
    --hf_upload_path "${HF_ORG:-your-org}/Advanced_Risk_${EXPERIMENT_NAME}"

  abs_target_dir="$(realpath "$target_dir")"

  if [ -f "$eval_script" ]; then
    echo "=== Running evaluation on merged model via eval_merged_model.sh ==="
    bash "$eval_script" --model_path "$abs_target_dir" --output_dir "$eval_dir" || true
  else
    echo "[INFO] Eval script not found: $eval_script"
  fi
}


