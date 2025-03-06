from datasets import load_dataset
import torch

def load_sft_data():
    """Load supervised fine-tuning data"""
    return load_dataset("json", data_files="data/sft_data.json")

def load_rl_data():
    """Load reinforcement learning data"""
    return load_dataset("json", data_files="data/rl_data.json")

def tokenize_sft_data(example, tokenizer):
    """Tokenize data for supervised fine-tuning"""
    # Assuming format is {"prompt": "...", "completion": "..."}
    prompt = example["prompt"]
    completion = example["completion"]
    
    # Format as instruction following format
    full_text = f"{prompt}\n{completion}"
    
    # Tokenize the text
    tokenized = tokenizer(full_text, truncation=True, max_length=512, 
                          padding="max_length", return_tensors="pt")
    
    # Important: Set labels equal to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def tokenize_rl_data(example, tokenizer):
    """Tokenize data specifically for RL training"""
    # For RL, we need just the prompts tokenized as PPO will generate completions
    tokenized = tokenizer(example["prompt"], truncation=True, max_length=512, 
                          padding="max_length", return_tensors="pt")
    
    # Store reference answer for reward calculation
    tokenized["reference_answer"] = example.get("reference_answer", "")
    
    return {k: v[0] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, torch.Tensor)) else v 
            for k, v in tokenized.items()}

def prepare_rl_dataset(dataset, tokenizer):
    """Prepare dataset in the format expected by PPOTrainer"""
    # Apply tokenization
    tokenized_dataset = dataset.map(
        lambda x: tokenize_rl_data(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset
