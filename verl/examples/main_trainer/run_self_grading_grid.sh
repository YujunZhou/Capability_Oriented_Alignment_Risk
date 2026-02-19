#!/bin/bash
# Self-grading GRPO Grid Search Script
# Run 3x3 = 9 task combinations:
#   Telling audit (telling_audit): 0 (No mention), 0.5 (50% audit), 1.0 (100% audit)
#   Actual audit prob (audit_prob): 0 (No audit), 0.5 (50% audit), 1.0 (100% audit)
#
# Usage:
#   ./run_self_grading_grid.sh -m Qwen3-4B-Base
#   ./run_self_grading_grid.sh -m llama
#

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# Parse -m argument
MODEL=""
while getopts ":m:" opt; do
  case $opt in
    m)
      MODEL="$OPTARG"
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 -m <model_name>"
  echo "Example: $0 -m Qwen3-4B-Base"
  exit 1
fi

echo "=========================================="
echo " Self-Grading Grid Search"
echo " Model: $MODEL"
echo "=========================================="
echo ""

# Parameter grid
TELLING_AUDIT_VALUES=(0 0.5 1.0)
AUDIT_PROB_VALUES=(0 0.5 1.0)

# Task name mapping
declare -A TELLING_NAMES
TELLING_NAMES[0]="NoMention"
TELLING_NAMES[0.5]="Tell50"
TELLING_NAMES[1.0]="Tell100"

declare -A AUDIT_NAMES
AUDIT_NAMES[0]="Audit0"
AUDIT_NAMES[0.5]="Audit50"
AUDIT_NAMES[1.0]="Audit100"

TOTAL_TASKS=9
CURRENT_TASK=0

echo "Will run $TOTAL_TASKS task combinations:"
echo ""
echo "| Telling \\ Actual audit | No audit (0) | 50% audit (0.5) | 100% audit (1.0) |"
echo "|------------------------|--------------|-----------------|------------------|"
echo "| No mention (0)         |     ✓        |       ✓         |        ✓         |"
echo "| 50% audit (0.5)        |     ✓        |       ✓         |        ✓         |"
echo "| 100% audit (1.0)       |     ✓        |       ✓         |        ✓         |"
echo ""

# Record start time
START_TIME=$(date +%s)

# Iterate over all combinations
for telling in "${TELLING_AUDIT_VALUES[@]}"; do
  for audit in "${AUDIT_PROB_VALUES[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    TASK_NAME="${TELLING_NAMES[$telling]}_${AUDIT_NAMES[$audit]}"
    
    echo ""
    echo "=========================================="
    echo " Task $CURRENT_TASK/$TOTAL_TASKS: $TASK_NAME"
    echo " telling_audit=$telling, audit_prob=$audit"
    EXP_SUFFIX="_${TASK_NAME}"
    echo " Experiment suffix: $EXP_SUFFIX"
    echo " Start time: $(date)"
    echo "=========================================="
    
    # Run training script
    EXPERIMENT_SUFFIX="$EXP_SUFFIX" \
    bash "$SCRIPT_DIR/run_self_grading_grpo.sh" \
      -m "$MODEL" \
      -a "$audit" \
      -t "$telling"
    
    echo ""
    echo "Task $CURRENT_TASK/$TOTAL_TASKS ($TASK_NAME) completed"
    echo ""
  done
done

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo " All $TOTAL_TASKS tasks completed!"
echo " Total time: ${HOURS}h ${MINUTES}m (${DURATION}s)"
echo " End time: $(date)"
echo "=========================================="

