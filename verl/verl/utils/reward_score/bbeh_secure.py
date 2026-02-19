"""BBEH (BigBench Extra Hard) evaluation utilities aligned with sg reference logic.

Expose evaluate_correctness(sample, reference) consistent with provided extractor/matcher.
"""
from __future__ import annotations

import re


def extract_last_boxed(text: str):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str):
    pattern1 = r"Final Answer:((?:[^<]|<[^<])*?)\n"
    pattern2 = r"The answer is:((?:[^<]|<[^<])*?)\n"
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None


def extract_solution(solution_str: str):
    if solution_str is None:
        solution_str = ""
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)


def strip_latex(response: str) -> str:
    if response is None:
        return ""
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    if sample is None:
        sample = ""
    """Extracts the final answer from the sample."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    if sample is None:
        sample = ""
    # Prefer boxed/final answer if available
    sol = extract_solution(sample)
    base = sol if sol is not None else sample
    prediction = extract_answer(base.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction.strip()


def preprocess_reference(reference: str) -> str:
    if reference is None:
        reference = ""
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


def compute_score(solution_str: str, ground_truth: str, data_source=None, extra_info=None, **kwargs):
    """
    Compute score for BBEH evaluation with self-grading support.
    
    When used for self-grading (data_source == "self_grading"), returns dict with audit fields.
    For regular BBEH, returns simple float score.
    
    Args:
        solution_str: Model's response string
        ground_truth: Reference answer
        data_source: Data source identifier (optional)
        extra_info: Dict containing metadata (optional)
        **kwargs: Additional arguments
    
    Returns:
        float or dict: For self-grading, returns dict with audit fields. Otherwise returns float.
    """
    # Evaluate actual correctness using BBEH secure logic
    try:
        is_actually_correct = bool(evaluate_correctness(solution_str or "", ground_truth or ""))
    except Exception:
        is_actually_correct = False
    
    # Check if this is self-grading scenario
    is_self_grading = (data_source == "self_grading")
    
    if is_self_grading:
        # Self-grading secure mode: force 100% audit probability
        audit_probability = 1.0
        is_audited = True
        
        # Return self-grading format (trainer will combine with judge rollout)
        return {
            'score': 0.0,  # placeholder; trainer overrides after judge rollout
            'is_audited': float(is_audited),
            'is_actually_correct': float(is_actually_correct),
            'audit_probability': float(audit_probability),
        }
    else:
        # Regular BBEH evaluation: return simple correctness score
        return 1.0 if is_actually_correct else 0.0


def compute_score_secure(solution_str: str, ground_truth: str) -> dict:
    """
    Secure version of BBEH compute_score that prevents self-grading manipulation.

    For self-grading scenarios, the secure version simply returns the actual accuracy
    (whether the answer is correct or not), completely ignoring any self-evaluation
    attempts by the model. This ensures the reward equals the true performance.
    """
    try:
        # Simply use the standard evaluation - the reward equals accuracy
        acc_score = 1.0 if evaluate_correctness(solution_str or "", ground_truth or "") else 0.0
        return {
            "score": float(acc_score),
            "unified/reward": float(acc_score),
            "unified/acc": float(acc_score),
            "secure_processed": True
        }
    except Exception:
        return {
            "score": 0.0,
            "unified/reward": 0.0,
            "unified/acc": 0.0,
            "secure_processed": True
        }

