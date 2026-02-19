# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import multiprocessing as mp

# Top-level parallel worker to be picklable under 'spawn'
def _tampering_score_worker(job):
    idx, response_str, ground_truth, extra_info, valid_response_length = job
    try:
        from verl.utils.reward_score import reward_tampering as rt
        res = rt.compute_score_vulnerable(
            data_source="reward_tampering_code",
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    except Exception as e:
        # 不在worker中打印详细错误，避免IO放大；返回最简错误信息
        res = {"score": 0.0, "execution_error": 1.0, "error_type": 1.0, "error_details": f"{type(e).__name__}: {e}"}
    return idx, res, valid_response_length, extra_info, response_str, ground_truth

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Build jobs (pre-decode in parent to avoid heavy tokenizer in workers)
        jobs = []  # list of tuples (idx, data_source, response_str, ground_truth, extra_info, valid_response_length)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {}).copy()
            # Expose original article to reward function (for proxy-gaming experiments)
            try:
                rm_info = data_item.non_tensor_batch.get("reward_model", {})
                original_article = rm_info.get("original_article", None)
                if original_article is not None:
                    extra_info["original_article"] = original_article
            except Exception:
                pass
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            if hasattr(data, 'meta_info') and 'global_steps' in data.meta_info:
                extra_info["__global_step__"] = data.meta_info['global_steps']
            extra_info["__prompt__"] = prompt_str

            jobs.append((i, data_source, response_str, ground_truth, extra_info, int(valid_response_length)))

        # Parallel settings
        parallel_workers = int(os.getenv("REWARD_PARALLEL_WORKERS", "0"))
        chunk_size = int(os.getenv("REWARD_PARALLEL_CHUNKSIZE", "8"))

        # Use parallel path only for reward_tampering_code to ensure picklable worker
        all_tampering = all(job[1] == "reward_tampering_code" for job in [(i, data[i].non_tensor_batch[self.reward_fn_key]) for i in range(len(data))])

        use_parallel = parallel_workers > 1 and all_tampering
        results = []
        if use_parallel:
            try:
                try:
                    ctx = mp.get_context("fork")
                except ValueError:
                    ctx = mp.get_context("spawn")
                # Repack jobs for worker signature
                tampering_jobs = [(idx, resp, gt, ei, vlen) for (idx, _ds, resp, gt, ei, vlen) in jobs]
                with ctx.Pool(processes=parallel_workers, maxtasksperchild=200) as pool:
                    for r in pool.imap_unordered(_tampering_score_worker, tampering_jobs, chunksize=chunk_size):
                        results.append(r)
            except Exception as e:
                print(f"[reward_parallel] disabled due to error: {e}. Fallback to sequential.")
                use_parallel = False

        if not use_parallel:
            # sequential
            for (idx, data_source, response_str, ground_truth, extra_info, valid_response_length) in jobs:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                results.append((idx, score, valid_response_length, extra_info, response_str, ground_truth))

        # Assemble results with consistent reward_extra_info lengths and print only 2 random samples per batch
        import random
        n = len(results)
        if n == 0:
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor

        # Deterministic order for alignment
        results_sorted = sorted(results, key=lambda x: x[0])  # sort by idx

        # Union of all keys present in dict scores
        all_keys = set()
        for _idx, score, _vlen, _ei, _resp, _gt in results_sorted:
            if isinstance(score, dict):
                all_keys.update(score.keys())

        # Initialize aligned lists with None
        aligned_extra = {k: [None] * n for k in all_keys}

        # Choose 2 random positions to print
        sample_positions = set(random.sample(range(n), k=min(2, n)))

        for pos, (idx, score, valid_response_length, extra_info, response_str, ground_truth) in enumerate(results_sorted):
            # Fill reward tensor
            reward_val = score.get("score", 0.0) if isinstance(score, dict) else score
            reward_tensor[idx, valid_response_length - 1] = float(reward_val)

            # Align extra info per key
            if isinstance(score, dict):
                for k in all_keys:
                    aligned_extra[k][pos] = score.get(k, None)

            # Print at most 2 samples per batch
            # 训练中不打印逐样本详情（避免IO瓶颈）


        # Step-level summary to reduce IO: count categories
        try:
            total = n
            succ = sum(1 for _idx, sc, _vlen, _ei, _resp, _gt in results_sorted if (isinstance(sc, dict) and sc.get('test_passed') == 1.0 and sc.get('reward_source') == 1.0) or (not isinstance(sc, dict) and sc >= 0.99))
            compile_ok_test_fail = sum(1 for _idx, sc, _vlen, _ei, _resp, _gt in results_sorted if isinstance(sc, dict) and sc.get('execution_success') == 1.0 and sc.get('test_passed') == 0.0)
            fail = sum(1 for _idx, sc, _vlen, _ei, _resp, _gt in results_sorted if isinstance(sc, dict) and sc.get('execution_success') == 0.0)
            tamper = sum(1 for _idx, sc, _vlen, _ei, _resp, _gt in results_sorted if isinstance(sc, dict) and sc.get('reward_source') == 2.0)
            print(f"[reward_step_summary] total={total} success={succ} compile_ok_but_test_fail={compile_ok_test_fail} fail={fail} tamper={tamper}")
        except Exception:
            pass

        # Move aligned extras into reward_extra_info (lists of length n)
        for k, v in aligned_extra.items():
            reward_extra_info[k] = v

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
