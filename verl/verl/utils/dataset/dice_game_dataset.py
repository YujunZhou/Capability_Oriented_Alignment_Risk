# verl/utils/dataset/dice_game_dataset.py
import os
import re
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

class DiceGameDataset(RLHFDataset):
    """专门为骰子游戏设计的数据集类"""
    
    def _read_files_and_tokenize(self):
        """读取文件并进行分词处理，专门处理骰子游戏的数据格式"""
        dataframes = []
        for parquet_file in self.data_files:
            # 读取parquet文件并缓存
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        
        # 不进行过滤，避免格式问题
        self.filter_overlong_prompts = False
        
    def __getitem__(self, item):
        """获取数据集中的一个项目"""
        row_dict: dict = self.dataframe[item]
        
        # 处理prompt字段，确保格式正确
        if self.prompt_key in row_dict:
            prompt_data = row_dict[self.prompt_key]
            
            if isinstance(prompt_data, str):
                # 如果是字符串，转换为聊天格式
                messages = [{"role": "user", "content": prompt_data}]
            elif isinstance(prompt_data, dict) and "role" in prompt_data and "content" in prompt_data:
                # 如果是单个消息字典
                messages = [prompt_data]
            elif isinstance(prompt_data, list):
                # 如果已经是消息列表
                messages = prompt_data
            else:
                # 处理字典形式的prompt，这是您的prepare_dice_game_data.py生成的格式
                # 查看row_dict的结构，提取正确的消息
                if "prompt" in prompt_data and isinstance(prompt_data["prompt"], dict):
                    messages = [prompt_data["prompt"]]
                else:
                    # 后备方案
                    messages = [{"role": "user", "content": str(prompt_data)}]
        else:
            # 如果找不到prompt键，使用空消息
            messages = [{"role": "user", "content": ""}]
            
        # 用我们修正的消息替换原始提示
        row_dict[self.prompt_key] = messages
        
        # 生成输入
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        # 后处理数据
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # 为了支持各种功能
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt
            
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        # 添加reward_model字段，因为RewardManager需要
        if "reward_model" not in row_dict:
            row_dict["reward_model"] = {"ground_truth": row_dict.get("ground_truth", "")}
            
        # 确保包含'data_source'字段，这似乎是奖励计算需要的
        if "data_source" not in row_dict:
            # 使用默认值或从其他字段派生
            row_dict["data_source"] = "dice_game"  # 或其他合适的默认值
            
        return row_dict