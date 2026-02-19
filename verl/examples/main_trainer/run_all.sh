#!/bin/bash
# Batch run main training scripts with a single model
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_all.sh -m <model_name>

Options:
  -m  Specify once; applied to all sub-scripts.
      If name contains no "/", script will add prefix meta-llama/ or Qwen/.
EOF
}

MODEL_INPUT=""
USE_REINFORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      if [[ -z "${2:-}" ]]; then
        echo "Error: Option -m requires an argument" >&2
        usage
        exit 1
      fi
      MODEL_INPUT="$2"
      shift 2
      ;;
    --reinforce)
      USE_REINFORCE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: run_all.sh does not support option $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Error: run_all.sh extra arguments: $*" >&2
  usage
  exit 1
fi

if [[ -z "$MODEL_INPUT" ]]; then
  echo "Error: Must specify model name with -m" >&2
  usage
  exit 1
fi

normalize_model_path() {
  local name="$1"
  if [[ "$name" == *"/"* ]]; then
    echo "$name"
    return
  fi

  local lower
  lower="$(echo "$name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lower" == "llama3.2" ]]; then
    echo "meta-llama/Llama-3.2-3B-Instruct"
    return
  fi
  if [[ "$lower" == "llama" ]]; then
    echo "meta-llama/Llama-3.1-8B-Instruct"
    return
  fi

  if [[ "$lower" == llama* || "$lower" == meta-llama* ]]; then
    echo "meta-llama/${name}"
  elif [[ "$lower" == qwen* ]]; then
    echo "Qwen/${name}"
  else
    echo "$name"
  fi
}

MODEL_PATH="$(normalize_model_path "$MODEL_INPUT")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

echo "Working directory: $REPO_ROOT"
echo "Model: $MODEL_PATH"
[[ "$USE_REINFORCE" == true ]] && echo "Mode: Reinforce++"

if [[ "$USE_REINFORCE" == true ]]; then
  SCRIPTS=(
    "run_situational_awareness_reinforce_pp.sh"
    "run_summarization_gaming_reinforce_pp.sh"
    "run_self_grading_reinforce_pp.sh"
    "run_reward_tampering_reinforce_pp.sh"
  )
  declare -A SCRIPT_PREFIX_MAP=(
    ["run_situational_awareness_reinforce_pp.sh"]="Situation_Aware_ReinforcePP"
    ["run_summarization_gaming_reinforce_pp.sh"]="Summarization_ReinforcePP"
    ["run_self_grading_reinforce_pp.sh"]="Self_Grading_ReinforcePP"
    ["run_reward_tampering_reinforce_pp.sh"]="Reward_Tampering_ReinforcePP"
  )
else
  SCRIPTS=(
    "run_situational_awareness_grpo.sh"
    "run_summarization_gaming.sh"
    "run_self_grading_grpo.sh"
    "run_reward_tampering.sh"
  )
  declare -A SCRIPT_PREFIX_MAP=(
    ["run_situational_awareness_grpo.sh"]="Situation_Aware"
    ["run_summarization_gaming.sh"]="Summarization"
    ["run_self_grading_grpo.sh"]="Self_Grading"
    ["run_reward_tampering.sh"]="Reward_Tampering"
  )
fi

resolve_project_name() {
  if [[ -n "${PROJECT_NAME:-}" ]]; then
    echo "$PROJECT_NAME"
  else
    case "$1" in
      run_reward_tampering.sh)
        echo "Advanced_Risk"
        ;;
      *)
        echo "Advanced_Risk_new2"
        ;;
    esac
  fi
}

resolve_checkpoint_dir() {
  local script="$1"
  local prefix="${SCRIPT_PREFIX_MAP[$script]:-}"
  if [[ -z "$prefix" ]]; then
    echo ""
    return
  fi
  local project_name
  project_name="$(resolve_project_name "$script")"
  local suffix="${EXPERIMENT_SUFFIX:-}"
  printf "./checkpoints/%s/%s_%s%s" "$project_name" "$prefix" "$MODEL_PATH" "$suffix"
}

has_existing_checkpoint() {
  local dir="$1"
  if [[ -z "$dir" || ! -d "$dir" ]]; then
    return 1
  fi
  local matches=()
  shopt -s nullglob
  matches=("$dir"/global_step_* "$dir"/*.ckpt "$dir"/*.pt)
  shopt -u nullglob
  if [[ ${#matches[@]} -gt 0 ]]; then
    return 0
  fi
  return 1
}

for script in "${SCRIPTS[@]}"; do
  ckpt_dir="$(resolve_checkpoint_dir "$script")"
  if has_existing_checkpoint "$ckpt_dir"; then
    echo "=== Skipping ${script} (checkpoint already exists at ${ckpt_dir}) ==="
    echo
    continue
  fi
  echo "=== Running ${script} ==="
  bash "${SCRIPT_DIR}/${script}" -m "$MODEL_PATH"
  echo "=== Completed ${script} ==="
done

echo "All tasks completed."

