#!/bin/bash
# Batch run *_secure main training scripts with a single model
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_all_secure.sh -m <model_name> [--ori] [--sft] [-n <count>]

Options:
  -m             Specify once; passed as-is to all *_secure sub-scripts.
  --ori          Use original model path in sub-scripts (no your-org/Advanced_Risk_* prefix).
  --sft          Use SFT model as base (prefix your-org/SFT_Advanced_Risk_*).
  -n, --max
  --max-settings  Max number of settings to run this time (by actual started count).
                   If not set, try all preset settings.
USAGE
}

MODEL_INPUT=""
USE_ORIGINAL_MODEL_PATH=false
USE_SFT_MODEL=false
MAX_SETTINGS_TO_RUN=0

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
    --ori)
      USE_ORIGINAL_MODEL_PATH=true
      shift
      ;;
    --sft)
      USE_SFT_MODEL=true
      shift
      ;;
    -n|--max|--max-settings)
      if [[ -z "${2:-}" ]]; then
        echo "Error: Option $1 requires an argument" >&2
        usage
        exit 1
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: Argument to $1 must be a non-negative integer" >&2
        usage
        exit 1
      fi
      MAX_SETTINGS_TO_RUN="$2"
      shift 2
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
      echo "Error: Unsupported option $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_INPUT" ]]; then
  echo "Error: Must specify model name with -m" >&2
  usage
  exit 1
fi

if [[ $# -gt 0 ]]; then
  echo "Error: run_all_secure.sh only accepts -m/--ori/--sft; extra arguments: $*" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

echo "Working directory: $REPO_ROOT"
echo "Model identifier: $MODEL_INPUT"
[[ "$USE_ORIGINAL_MODEL_PATH" == true ]] && echo "Sub-scripts will use original model path (--ori)"
[[ "$USE_SFT_MODEL" == true ]] && echo "Sub-scripts will use SFT model as base (--sft)"
if [[ "$MAX_SETTINGS_TO_RUN" -gt 0 ]]; then
  echo "Will run at most $MAX_SETTINGS_TO_RUN settings (by actual started count)"
fi

EXTRA_MODEL_ARGS=()
if [[ "$USE_ORIGINAL_MODEL_PATH" == true ]]; then
  EXTRA_MODEL_ARGS+=(--ori)
fi
if [[ "$USE_SFT_MODEL" == true ]]; then
  EXTRA_MODEL_ARGS+=(--sft)
fi

SCRIPTS=(
  "run_situational_awareness_grpo_secure.sh"
  "run_self_grading_grpo_secure.sh"
  "run_reward_tampering_secure.sh"
)

declare -A SCRIPT_PREFIX_MAP=(
  ["run_situational_awareness_grpo_secure.sh"]="Secure_Situation_Aware"
  ["run_self_grading_grpo_secure.sh"]="Secure_Self_Grading"
  ["run_reward_tampering_secure.sh"]="Secure_Reward_Tampering"
)

resolve_project_name() {
  if [[ -n "${PROJECT_NAME:-}" ]]; then
    echo "$PROJECT_NAME"
  else
    echo "Advanced_Risk_new2"
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
  local suffix=""
  if [[ "$USE_ORIGINAL_MODEL_PATH" == true ]]; then
    suffix="_ori"
  elif [[ "$USE_SFT_MODEL" == true ]]; then
    suffix="_sft"
  fi

  printf "./checkpoints/%s/%s_%s%s" "$project_name" "$prefix" "$MODEL_INPUT" "$suffix"
}

has_existing_checkpoint() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
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

settings_started=0
for script in "${SCRIPTS[@]}"; do
  if [[ "$MAX_SETTINGS_TO_RUN" -gt 0 && "$settings_started" -ge "$MAX_SETTINGS_TO_RUN" ]]; then
    break
  fi

  ckpt_dir="$(resolve_checkpoint_dir "$script")"
  if [[ -z "$ckpt_dir" ]]; then
    echo "Unrecognized setting: ${script}, skipping."
    continue
  fi

  if has_existing_checkpoint "$ckpt_dir"; then
    echo "=== Skipping ${script} (checkpoint already exists at ${ckpt_dir}) ==="
    echo
    continue
  fi

  echo "=== Running ${script} ==="
  bash "${SCRIPT_DIR}/${script}" -m "$MODEL_INPUT" "${EXTRA_MODEL_ARGS[@]}"
  echo "=== Completed ${script} ==="
  echo
  sleep 2
  settings_started=$((settings_started + 1))
done

if [[ "$MAX_SETTINGS_TO_RUN" -gt 0 && "$settings_started" -lt "$MAX_SETTINGS_TO_RUN" ]]; then
  echo "Note: Only ${settings_started}/${MAX_SETTINGS_TO_RUN} settings were started (others may have been skipped due to existing checkpoint or missing script)."
fi

echo "All secure tasks completed."
