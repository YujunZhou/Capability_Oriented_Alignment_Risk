# Training Script Command-Line Usage

All training scripts support a unified CLI: use the `-m` option to specify the model.

## Basic usage

```bash
# Use default model
bash <script_name>.sh

# Specify model
bash <script_name>.sh -m <model_name>
```

## Supported model arguments

### 1. Llama models
```bash
bash <script_name>.sh -m llama
# or
bash <script_name>.sh -m Llama
```
This maps to `meta-llama/Meta-Llama-3.1-8B-Instruct`

### 2. Qwen models
```bash
bash <script_name>.sh -m Qwen-4B-Base
bash <script_name>.sh -m Qwen-4B
bash <script_name>.sh -m Qwen2.5-3B-Instruct
```
This maps to `Qwen/Qwen-4B-Base`, `Qwen/Qwen-4B`, etc.

### 3. Full model path
```bash
bash <script_name>.sh -m "your-org/your-model"
```

### 4. Other model names
```bash
bash <script_name>.sh -m "your-model-name"
```

## Environment variables

All scripts support custom project and experiment names via environment variables:

```bash
# Custom project and experiment name
export PROJECT_NAME="my_project"
export EXPERIMENT_NAME="my_experiment"
bash <script_name>.sh -m llama
```

## Available training scripts

### 1. Situational Awareness
```bash
# Default Qwen model
bash run_situational_awareness_grpo.sh

# Llama model
bash run_situational_awareness_grpo.sh -m llama

# Custom Qwen model
bash run_situational_awareness_grpo.sh -m Qwen-4B-Base
```

### 2. Reward Tampering
```bash
bash run_reward_tampering.sh -m llama
bash run_reward_tampering.sh -m Qwen-4B-Base
```

### 3. Zero Width Attack
```bash
bash run_zero_width_attack.sh -m llama
bash run_zero_width_attack.sh -m Qwen2.5-3B-Instruct
```

### 4. Summarization Gaming
```bash
bash run_summarization_gaming.sh -m llama
bash run_summarization_gaming.sh -m Qwen-4B-Base
```

### 5. Self Grading
```bash
bash run_self_grading_grpo.sh -m llama
bash run_self_grading_grpo.sh -m Qwen-4B-Base
```

### 6. Secure Baseline
```bash
bash run_secure_baseline.sh -m llama
bash run_secure_baseline.sh -m Qwen2.5-3B-Instruct
```

## Script behavior

Each script will:
1. Run the corresponding GRPO training (using 8 GPUs)
2. Merge model checkpoints after training
3. Run model evaluation
4. Record training time and GPU usage

## Default model configuration

- `run_situational_awareness_grpo.sh`: `Qwen/Qwen3-4B-Base`
- `run_reward_tampering.sh`: `Qwen/Qwen3-4B-Base`
- `run_zero_width_attack.sh`: `Qwen/Qwen2.5-3B-Instruct`
- `run_summarization_gaming.sh`: `Qwen/Qwen3-4B-Base`
- `run_self_grading_grpo.sh`: `Qwen/Qwen3-4B-Base`
- `run_secure_baseline.sh`: `Qwen/Qwen2.5-3B-Instruct`

## Notes

1. Ensure sufficient GPU resources (most scripts use 8 GPUs)
2. Ensure data files exist in the corresponding `data/` directory
3. Some scripts will generate data files if they do not exist
4. Model merge and evaluation run automatically after training
