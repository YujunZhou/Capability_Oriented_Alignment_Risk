# verl/utils/reward_score/batch_reward_manager.py

import logging
import torch
import os
import asyncio
import time
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager.registry import register

logger = logging.getLogger(__name__)

@register("batch_llm_judge")
class BatchLLMJudgeRewardManager:
    """批处理LLM评判奖励管理器"""
    
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **kwargs):
        """初始化批处理奖励管理器
        
        参数:
            tokenizer: 用于处理文本的分词器
            num_examine: 要检查的样本数量
            compute_score: 用于计算奖励分数的函数
            reward_fn_key: 决定使用哪种奖励函数的关键字，默认为"data_source"
            **kwargs: 其他关键字参数
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # 从环境变量或参数获取批处理大小
        env_batch_size = os.environ.get("LLM_JUDGE_BATCH_SIZE")
        if env_batch_size is not None:
            self.batch_size = int(env_batch_size)
        else:
            self.batch_size = kwargs.get("batch_size", 8)
        
        logger.info(f"初始化BatchLLMJudgeRewardManager，批处理大小={self.batch_size}")
        
        # 创建批处理缓冲区
        self.prompt_buffer = []
        self.response_buffer = []
        self.ground_truth_buffer = []
        self.data_source_buffer = []
        self.extra_info_buffer = []
        self.index_buffer = []  # 存储样本在批次中的索引
        
        # 创建异步锁
        self.buffer_lock = asyncio.Lock()
        
    async def _process_batch(self, data_source):
        """处理一批样本"""
        logger.info(f"处理批次：{len(self.prompt_buffer)}个样本")
        
        # 收集当前批次中的所有样本
        current_prompts = self.prompt_buffer.copy()
        current_responses = self.response_buffer.copy()
        current_ground_truths = self.ground_truth_buffer.copy()
        current_data_sources = self.data_source_buffer.copy()
        current_extra_infos = self.extra_info_buffer.copy()
        current_indices = self.index_buffer.copy()
        
        # 清空缓冲区
        self.prompt_buffer = []
        self.response_buffer = []
        self.ground_truth_buffer = []
        self.data_source_buffer = []
        self.extra_info_buffer = []
        self.index_buffer = []
        
        # 准备批量调用compute_score
        all_scores = []
        start_time = time.time()
        
        # 创建任务列表
        tasks = []
        for i in range(len(current_prompts)):
            is_last_in_batch = (i == len(current_prompts) - 1)
            current_extra_info = current_extra_infos[i].copy()
            current_extra_info["is_last_in_batch"] = is_last_in_batch
            
            # 创建异步任务
            task = asyncio.create_task(
                self._async_compute_score(
                    data_source=current_data_sources[i],
                    solution_str=current_responses[i],
                    ground_truth=current_ground_truths[i],
                    extra_info=current_extra_info
                )
            )
            tasks.append(task)
        
        # 并行执行所有任务
        all_scores = await asyncio.gather(*tasks)
        
        end_time = time.time()
        logger.info(f"批处理完成，耗时: {end_time - start_time:.2f}秒")
        
        return all_scores, current_indices
    
    async def _async_compute_score(self, data_source, solution_str, ground_truth, extra_info):
        """异步计算单个样本的分数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.compute_score,
            data_source,
            solution_str,
            ground_truth,
            extra_info
        )
    
    def __call__(self, data: DataProto, return_dict=False):
        """计算一批数据的奖励
        
        参数:
            data: 包含输入数据的DataProto对象
            return_dict: 是否以字典形式返回结果
            
        返回:
            如果return_dict为True，返回包含奖励张量和额外信息的字典；
            否则仅返回奖励张量
        """
        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
                
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        batch_len = len(data)
        
        # 创建异步运行时
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def process_data_batch():
            # 收集所有样本
            for i in range(batch_len):
                data_item = data[i]  # DataProtoItem
                
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                # 解码
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {}).copy()
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns
                
                # 添加到缓冲区
                async with self.buffer_lock:
                    self.prompt_buffer.append(prompt_str)
                    self.response_buffer.append(response_str)
                    self.ground_truth_buffer.append(ground_truth)
                    self.data_source_buffer.append(data_source)
                    self.extra_info_buffer.append(extra_info)
                    self.index_buffer.append(i)
                    
                    # 如果缓冲区达到批处理大小或这是最后一个样本，处理批次
                    if len(self.prompt_buffer) >= self.batch_size or i == batch_len - 1:
                        if len(self.prompt_buffer) > 0:
                            # 处理当前批次
                            batch_scores, batch_indices = await self._process_batch(data_source)
                            
                            # 更新奖励张量
                            for idx, score_result in zip(batch_indices, batch_scores):
                                valid_response_length = data[idx].batch["attention_mask"][data[idx].batch["prompts"].shape[-1]:].sum()
                                
                                if isinstance(score_result, dict):
                                    reward = score_result["score"]
                                    # 存储包括原始奖励在内的信息
                                    for key, value in score_result.items():
                                        reward_extra_info[key].append(value)
                                else:
                                    reward = score_result
                                    
                                reward_tensor[idx, valid_response_length - 1] = reward
                                
                                # 打印样本信息（如果需要）
                                current_data_source = data[idx].non_tensor_batch[self.reward_fn_key]
                                if current_data_source not in already_print_data_sources:
                                    already_print_data_sources[current_data_source] = 0
                                    
                                if already_print_data_sources[current_data_source] < self.num_examine:
                                    already_print_data_sources[current_data_source] += 1
                                    print("[prompt]", data[idx].non_tensor_batch.get("extra_info", {}).get("prompt", ""))
                                    print("[response]", self.tokenizer.decode(data[idx].batch["responses"][:valid_response_length], skip_special_tokens=True))
                                    print("[ground_truth]", data[idx].non_tensor_batch["reward_model"]["ground_truth"])
                                    if isinstance(score_result, dict):
                                        for key, value in score_result.items():
                                            print(f"[{key}]", value)
                                    else:
                                        print("[score]", score_result)
        
        # 执行异步处理
        loop.run_until_complete(process_data_batch())
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor