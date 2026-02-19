# verl/utils/dataset/situational_awareness_dataset.py
from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

class SituationalAwarenessDataset(RLHFDataset):
    """Dataset class for Situational Awareness tasks"""
    
    def _read_files_and_tokenize(self):
        """Read and tokenize files"""
        super()._read_files_and_tokenize()
        print(f"Dataset loaded, sample count: {len(self.dataframe)}")
        
        # Check if dataset has features attribute (HuggingFace Dataset)
        if hasattr(self.dataframe, 'features'):
            print(f"Dataset fields: {list(self.dataframe.features.keys())}")
        
        # Print a sample
        print(f"Sample example:\n{self.dataframe[0]}")
        
        # Disable filtering of long prompts
        self.filter_overlong_prompts = False
    
    def __getitem__(self, item):
        """Get an item from the dataset with necessary fields added"""
        row_dict = self.dataframe[item]
        
        # Process the prompt field, ensure correct format
        if self.prompt_key in row_dict:
            prompt_text = row_dict[self.prompt_key]
            
            # Convert to chat format
            if isinstance(prompt_text, str):
                # If string, convert to chat format
                messages = [{"role": "user", "content": prompt_text}]
            elif isinstance(prompt_text, dict) and "role" in prompt_text and "content" in prompt_text:
                # If already a message dict
                messages = [prompt_text]
            elif isinstance(prompt_text, list):
                # If already a message list
                messages = prompt_text
            else:
                # Fallback
                messages = [{"role": "user", "content": str(prompt_text)}]
        else:
            # If prompt key not found, use empty message
            messages = [{"role": "user", "content": ""}]
        
        # Replace original prompt
        row_dict[self.prompt_key] = messages
        
        # Generate inputs
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        # Post-process data
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
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # Add reward_model field
        if "reward_model" not in row_dict:
            row_dict["reward_model"] = {"ground_truth": row_dict.get("ground_truth", "")}
        
        # Ensure extra_info contains prompt (for reward function)
        if "extra_info" not in row_dict:
            row_dict["extra_info"] = {}
        
        # Add original prompt to extra_info for reward function
        if "prompt" not in row_dict["extra_info"]:
            original_prompt = messages[0]["content"] if messages and "content" in messages[0] else ""
            row_dict["extra_info"]["prompt"] = original_prompt
        
        # Support for raw_chat and full_prompt
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt
        
        return row_dict