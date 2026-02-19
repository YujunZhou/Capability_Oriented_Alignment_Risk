#!/bin/bash
# Generic evaluation-only runner that:
# - uses the naive Reward Manager
# - ensures validation uses the SAME custom reward function as training when provided
# - mirrors training's judge rollout logic when enabled
#
# Usage examples (run from Advanced_risk_github/verl):
#   bash examples/main_trainer/run_eval_naive.sh \
#     --train_parquet ./data/self_grading/train.parquet \
#     --test_parquet ./data/self_grading/test.parquet \
#     --model Qwen/Qwen3-4B-Base \
#     --enable_judge true
#
#   # Reward tampering: reuse the same custom reward fn as training
#   bash examples/main_trainer/run_eval_naive.sh \
#     --train_parquet ./data/reward_tampering/train.parquet \
#     --test_parquet ./data/reward_tampering/test.parquet \
#     --model Qwen/Qwen3-4B-Base \
#     --custom_path ./verl/utils/reward_score/reward_tampering_xml.py \
#     --custom_name compute_score \
#     --enable_judge true

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Set in environment or here; do not commit real keys
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export DEEPINFRA_API_KEY="${DEEPINFRA_API_KEY:-}"

# Defaults
MODEL_PATH="Qwen/Qwen3-4B-Base"
PROJECT_NAME=${PROJECT_NAME:-Advanced_Risk}
EXPERIMENT_PREFIX="Eval_Naive"
# ===== Built-in default tasks (edit as needed; one key=value per line) =====
# Predefined model list (edit/extend as needed; set HF_ORG for your HuggingFace org)
PREDEFINED_MODELS=(
  "${HF_ORG:-your-org}/Advanced_Risk_Summarization_Qwen3-4B-Base"
  "${HF_ORG:-your-org}/Advanced_Risk_Summarization_Qwen3-4B"
  "${HF_ORG:-your-org}/Advanced_Risk_Summarization_llama"
  "${HF_ORG:-your-org}/Advanced_Risk_Situation_Aware_Qwen3-4B"
  "${HF_ORG:-your-org}/Advanced_Risk_Situation_Aware_Qwen3-4B-Base"
  "${HF_ORG:-your-org}/Advanced_Risk_Situation_Aware_llama"
  "${HF_ORG:-your-org}/Advanced_Risk_Reward_Tampering_Qwen3-4B-Base"
  "${HF_ORG:-your-org}/Advanced_Risk_Reward_Tampering_Qwen3-4B"
  "${HF_ORG:-your-org}/Advanced_Risk_Reward_Tampering_llama"
  "${HF_ORG:-your-org}/Advanced_Risk_Self_Grading_Qwen3-4B-Base"
  "${HF_ORG:-your-org}/Advanced_Risk_Self_Grading_Qwen3-4B"
  "${HF_ORG:-your-org}/Advanced_Risk_Self_Grading_llama"
)

# SFT predefined model list (SFT_ prefix per name; paths relative to repo root verl)
SFT_PREDEFINED_MODELS=(
  "./saves/SFT_Advanced_Risk_Summarization_Qwen3-4B-Base"
  "./saves/SFT_Advanced_Risk_Summarization_Qwen3-4B"
  "./saves/SFT_Advanced_Risk_Summarization_llama"
  "./saves/SFT_Advanced_Risk_Situation_Aware_Qwen3-4B"
  "./saves/SFT_Advanced_Risk_Situation_Aware_Qwen3-4B-Base"
  "./saves/SFT_Advanced_Risk_Situation_Aware_llama"
  "./saves/SFT_Advanced_Risk_Reward_Tampering_Qwen3-4B-Base"
  "./saves/SFT_Advanced_Risk_Reward_Tampering_Qwen3-4B"
  "./saves/SFT_Advanced_Risk_Reward_Tampering_llama"
  "./saves/SFT_Advanced_Risk_Self_Grading_Qwen3-4B-Base"
  "./saves/SFT_Advanced_Risk_Self_Grading_Qwen3-4B"
  "./saves/SFT_Advanced_Risk_Self_Grading_llama"
)

# Simple model alias mapping (edit/extend as needed)
# -m q4  => distill: qwen3-4b base (default)
# -m q7  => Qwen/Qwen2.5-7B-Instruct
# -m q14 => Qwen/Qwen2.5-14B-Instruct
resolve_model() {
  local alias="$1"
  case "$alias" in
    ""|default|q4) echo "Qwen/Qwen3-4B-Base";;
    q7) echo "Qwen/Qwen2.5-7B-Instruct";;
    q14) echo "Qwen/Qwen2.5-14B-Instruct";;
    q32) echo "Qwen/Qwen2.5-32B-Instruct";;
    l8) echo "meta-llama/Llama-3.1-8B-Instruct";;
    l70) echo "meta-llama/Llama-3.1-70B-Instruct";;
    *) echo "$alias";;  # Pass full HF name or local path
  esac
}

# Self-grading (must enable Judge for second-pass self-eval)
SELF_GRADING_FLAGS=(
  "data.train_files=['./data/self_grading/train.parquet']"
  "data.val_files=['./data/self_grading/test.parquet']"
  "+trainer.enable_judge_rollout=True"
)

# Proxy Gaming (summarization proxy metrics)
PROXY_GAMING_FLAGS=(
  "data.train_files=['./data/proxy_gaming/train.parquet']"
  "data.val_files=['./data/proxy_gaming/test.parquet']"
  "custom_reward_function.path=./verl/utils/reward_score/proxy_gaming.py"
  "custom_reward_function.name=compute_score_article_overlap"
)

# Reward Tampering (code poison reward tampering)
REWARD_TAMPERING_FLAGS=(
  "data.train_files=['./data/reward_tampering/train.parquet']"
  "data.val_files=['./data/reward_tampering/test.parquet']"
  "custom_reward_function.path=./verl/utils/reward_score/reward_tampering_xml.py"
  "custom_reward_function.name=compute_score"
)

# Situational Awareness (LLM Judge scoring)
SITUATIONAL_FLAGS=(
  "data.train_files=['./data/situational_awareness/train.parquet']"
  "data.val_files=['./data/situational_awareness/test.parquet']"
  "custom_reward_function.path=./verl/utils/reward_score/llm_judge.py"
  "custom_reward_function.name=compute_score_single"
)


print_help() {
  cat <<EOF
Usage: $0 [-m MODEL_ALIAS] [--task self|proxy|tampering|situational] [--batch-models] [--sft] [--match-task]

- Use -m to specify/map the model to evaluate (see alias rules below)
- Use --task to select task; script will append that task's default key=value lines
- Use --batch-models to evaluate all models in the predefined list
- Use --sft to evaluate all models in the SFT predefined list (SFT_ prefix)
- Use --match-task with --sft to auto-match task by model name (one task per model)
    * Dice/Summarization → proxy
    * Situation_Aware → situational
    * Reward_Tampering → tampering
    * Self_Grading → self
- You can edit default entries in the *_FLAGS arrays at the top of the script
- Only self_grading enables Judge; other tasks do not set Judge

Model aliases:
  q4   => Qwen/Qwen3-4B-Base (default)
  q7   => Qwen/Qwen2.5-7B-Instruct
  q14  => Qwen/Qwen2.5-14B-Instruct
  q32  => Qwen/Qwen2.5-32B-Instruct
  l8   => meta-llama/Llama-3.1-8B-Instruct
  l70  => meta-llama/Llama-3.1-70B-Instruct

Predefined model list (--batch-models):
$(printf "  %s\n" "${PREDEFINED_MODELS[@]}")

SFT predefined model list (--sft):
$(printf "  %s\n" "${SFT_PREDEFINED_MODELS[@]}")

This script runs validation only with reward_model.reward_manager=naive.
EOF
}

# Helper: absolute path
abspath() {
  python3 - "$1" <<'PY'
import os,sys
p=sys.argv[1]
print(os.path.abspath(p))
PY
}

run_eval_one() {
  local TASK_KIND="$1"
  local MODEL_P="$2"

  local EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${TASK_KIND}"
  local VAL_DIR="./eval_dumps/${EXPERIMENT_NAME}"
  mkdir -p "$VAL_DIR"

  # If result already exists for this model+task and either acc or exploit_ratio is non-zero, skip
  local TASK_KEY_EARLY="$TASK_KIND"
  case "$TASK_KIND" in
    self|self_grading) TASK_KEY_EARLY="self";;
    proxy|proxy_gaming) TASK_KEY_EARLY="proxy";;
    tampering|reward_tampering) TASK_KEY_EARLY="tampering";;
    situational) TASK_KEY_EARLY="situational";;
  esac
  local MODEL_ID_EARLY=$(resolve_model "${MODEL_P:-$MODEL_PATH}")
  local MODEL_NAME_EARLY=$(basename "$MODEL_ID_EARLY")
  local OUT_JSON_EARLY="./eval_results/${MODEL_NAME_EARLY}.json"
  if [[ -f "$OUT_JSON_EARLY" ]]; then
    local SHOULD_SKIP
    SHOULD_SKIP=$(python3 - "$OUT_JSON_EARLY" "$TASK_KEY_EARLY" <<'PY'
import json, sys
path, task = sys.argv[1], sys.argv[2]
try:
    with open(path, 'r') as f:
        data = json.load(f)
    t = data.get(task) or {}
    acc = float(t.get('acc') or 0.0)
    exploit_ratio = float(t.get('exploit_ratio') or 0.0)
    if (acc != 0.0) or (exploit_ratio != 0.0):
        print('skip')
except Exception:
    pass
PY
)
    if [[ "$SHOULD_SKIP" == "skip" ]]; then
      echo "[SKIP] ${MODEL_NAME_EARLY} task=${TASK_KEY_EARLY} already has non-zero acc/exploit_ratio in $OUT_JSON_EARLY"
      return 0
    fi
  fi

  # Assemble task-specific flags (one key=value per line, edit as needed)
  local FLAGS=()
  case "$TASK_KIND" in
    self|self_grading)
      FLAGS+=("${SELF_GRADING_FLAGS[@]}")
      ;;
    proxy|proxy_gaming)
      FLAGS+=("${PROXY_GAMING_FLAGS[@]}")
      ;;
    tampering|reward_tampering)
      FLAGS+=("${REWARD_TAMPERING_FLAGS[@]}")
      ;;
    situational)
      FLAGS+=("${SITUATIONAL_FLAGS[@]}")
      ;;
    *)
      echo "[ERROR] Unknown task: $TASK_KIND (use self|proxy|tampering|situational)"; return 2;;
  esac
  # Dump validation samples to VAL_DIR for post-processing acc/exploit
  FLAGS+=("trainer.validation_data_dir=$VAL_DIR")

  # Situational task requires external LLM judge via DeepInfra API
  if [[ "$TASK_KIND" == "situational" ]]; then
    # Provide sane defaults for judge batching (can be overridden by env)
    export LLM_JUDGE_BATCH_SIZE="${LLM_JUDGE_BATCH_SIZE:-64}"
    export LLM_JUDGE_MAX_CONCURRENT_REQUESTS="${LLM_JUDGE_MAX_CONCURRENT_REQUESTS:-32}"

    if [[ -z "${DEEPINFRA_API_KEY:-}" ]]; then
      echo "[ERROR] DEEPINFRA_API_KEY is not set. Situational task uses llm_judge.py which calls DeepInfra's OpenAI-compatible API."
      echo "        Please export DEEPINFRA_API_KEY (and optionally OPENAI_API_KEY) before running this script."
      echo "        Example: export DEEPINFRA_API_KEY=..."
      exit 1
    fi
    echo "[INFO] Situational: using LLM_JUDGE_BATCH_SIZE=$LLM_JUDGE_BATCH_SIZE MAX_CONCURRENT=$LLM_JUDGE_MAX_CONCURRENT_REQUESTS"
  fi

  # Build command array
  local CMD=(python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    data.train_batch_size=64
    data.max_prompt_length=1024
    data.max_response_length=4096
    data.filter_overlong_prompts=True
    data.truncation=right
    data.reward_fn_key=data_source
    data.return_raw_chat=True
    reward_model.reward_manager=naive
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.model.path=$(resolve_model "${MODEL_P:-$MODEL_PATH}")
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.000
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger="['console']"
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    trainer.log_val_generations=2
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.save_freq=100000
    trainer.test_freq=1
    trainer.total_epochs=1
    trainer.val_before_train=True
    trainer.val_only=True
  )

  # Append task-specific flag lines
  for f in "${FLAGS[@]}"; do CMD+=("$f"); done

  # Print and run
  echo "[RUN] ${CMD[*]}"
  "${CMD[@]}"

  # After run, compute acc/exploit and save rollouts
  local MODEL_ID=$(resolve_model "${MODEL_P:-$MODEL_PATH}")
  local MODEL_NAME=$(basename "$MODEL_ID")
  local OUT_JSON="./eval_results/${MODEL_NAME}.json"
  local ROLL_JSON="./eval_results/rollouts/${MODEL_NAME}/${TASK_KIND}.json"
  mkdir -p ./eval_results ./eval_results/rollouts/${MODEL_NAME}
  # Map task kind to task key in JSON
  local TASK_KEY="$TASK_KIND"
  case "$TASK_KIND" in
    self|self_grading) TASK_KEY="self";;
    proxy|proxy_gaming) TASK_KEY="proxy";;
    tampering|reward_tampering) TASK_KEY="tampering";;
    situational) TASK_KEY="situational";;
  esac
  python3 "$(dirname "$0")/eval/extract_unified_metrics.py" \
    --val_dir "$VAL_DIR" \
    --output "$OUT_JSON" \
    --task "$TASK_KEY"
  python3 "$(dirname "$0")/eval/extract_rollouts.py" \
    --val_dir "$VAL_DIR" \
    --output "$ROLL_JSON"
  echo "[OK] Saved eval metrics ($TASK_KEY) to $OUT_JSON; rollouts to $ROLL_JSON"
}


# Parse args (choose a single task)
TASK_KIND=""
BATCH_MODELS=false
SFT_MODELS=false
MATCH_TASK=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK_KIND="$2"; shift 2;;   # self|proxy|tampering|situational
    -m)     MODEL_PATH="$2"; shift 2;;  # model alias or path
    --model) MODEL_PATH="$2"; shift 2;;
    --batch-models) BATCH_MODELS=true; shift;;
    --sft) SFT_MODELS=true; shift;;
    --match-task) MATCH_TASK=true; shift;;
    -h|--help) print_help; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; print_help; exit 2;;
  esac

done

# Helper: get task from model name
get_task_from_model() {
  local model="$1"
  if [[ "$model" == *"Dice"* ]] || [[ "$model" == *"Summarization"* ]]; then
    echo "proxy"
  elif [[ "$model" == *"Situation_Aware"* ]]; then
    echo "situational"
  elif [[ "$model" == *"Reward_Tampering"* ]]; then
    echo "tampering"
  elif [[ "$model" == *"Self_Grading"* ]]; then
    echo "self"
  else
    echo ""
  fi
}

# Batch evaluation: run all predefined models
if [[ "$BATCH_MODELS" == "true" ]]; then
  if [[ -z "$TASK_KIND" ]]; then
    echo "[INFO] Batch mode: Running all predefined models on all tasks"
    for model in "${PREDEFINED_MODELS[@]}"; do
      echo "[INFO] ===== Evaluating model: $model ====="
      run_eval_one self        "$model"
      run_eval_one proxy       "$model"
      run_eval_one tampering   "$model"
      run_eval_one situational "$model"
      echo "[INFO] ===== Completed model: $model ====="
    done
  else
    echo "[INFO] Batch mode: Running all predefined models on task: $TASK_KIND"
    for model in "${PREDEFINED_MODELS[@]}"; do
      echo "[INFO] ===== Evaluating model: $model on task: $TASK_KIND ====="
      run_eval_one "$TASK_KIND" "$model"
      echo "[INFO] ===== Completed model: $model ====="
    done
  fi
  exit 0
fi

# SFT batch evaluation: run all SFT predefined models
if [[ "$SFT_MODELS" == "true" ]]; then
  if [[ "$MATCH_TASK" == "true" ]]; then
    echo "[INFO] SFT mode with --match-task: Each model runs its matching task only"
    for model in "${SFT_PREDEFINED_MODELS[@]}"; do
      matched_task=$(get_task_from_model "$model")
      if [[ -z "$matched_task" ]]; then
        echo "[WARN] Cannot determine task for model: $model, skipping..."
        continue
      fi
      echo "[INFO] ===== Evaluating SFT model: $model on matched task: $matched_task ====="
      run_eval_one "$matched_task" "$model"
      echo "[INFO] ===== Completed SFT model: $model ====="
    done
  elif [[ -z "$TASK_KIND" ]]; then
    echo "[INFO] SFT mode: Running all SFT predefined models on all tasks"
    for model in "${SFT_PREDEFINED_MODELS[@]}"; do
      echo "[INFO] ===== Evaluating SFT model: $model ====="
      run_eval_one self        "$model"
      run_eval_one proxy       "$model"
      run_eval_one tampering   "$model"
      run_eval_one situational "$model"
      echo "[INFO] ===== Completed SFT model: $model ====="
    done
  else
    echo "[INFO] SFT mode: Running all SFT predefined models on task: $TASK_KIND"
    for model in "${SFT_PREDEFINED_MODELS[@]}"; do
      echo "[INFO] ===== Evaluating SFT model: $model on task: $TASK_KIND ====="
      run_eval_one "$TASK_KIND" "$model"
      echo "[INFO] ===== Completed SFT model: $model ====="
    done
  fi
  exit 0
fi

# Default: run all built-in tasks in sequence
if [[ -z "$TASK_KIND" ]]; then
  echo "[INFO] No --task specified. Running all built-in tasks: self, proxy, tampering, situational"
  run_eval_one self        "$MODEL_PATH"
  run_eval_one proxy       "$MODEL_PATH"
  run_eval_one tampering   "$MODEL_PATH"
  run_eval_one situational "$MODEL_PATH"
  exit 0
fi

# Single task
run_eval_one "$TASK_KIND" "$MODEL_PATH"

exit 0

