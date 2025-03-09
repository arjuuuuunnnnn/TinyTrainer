from datasets import load_dataset
import torch

def load_sft_data():
    return load_dataset("json", data_files="data/sft_data.json")

def load_rl_data():
    return load_dataset("json", data_files="data/rl_data.json")

def tokenize_sft_data(example, tokenizer):
    # {"prompt": "...", "completion": "..."}
    prompt = example["prompt"]
    completion = example["completion"]
    
    full_text = f"{prompt}\n{completion}"
    
    tokenized = tokenizer(full_text, truncation=True, max_length=512, 
                          padding="max_length")
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def tokenize_rl_data(example, tokenizer):
    # Ensure we're working with proper data
    if "prompt" not in example:
        print(f"Warning: Missing 'prompt' in example: {example.keys()}")
        # Create a dummy prompt if missing
        example["prompt"] = "This is a placeholder prompt."
    
    # Convert to tensors explicitly with proper dimensions
    tokenized = tokenizer(example["prompt"], truncation=True, max_length=512, 
                         padding="max_length", return_tensors="pt")
    
    # Extract tensors from the batch to individual tensors
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor):
            # Remove the batch dimension which is added by return_tensors="pt"
            tokenized[key] = tokenized[key].squeeze(0)
    
    # Add reference answer if available
    tokenized["reference_answer"] = example.get("reference_answer", "")
    
    return tokenized

def prepare_rl_dataset(dataset, tokenizer):
    print(f"Dataset before tokenization: {len(dataset)} examples")
    print(f"Dataset column names: {dataset.column_names}")
    
    # Apply tokenization with error handling
    def safe_tokenize(example):
        try:
            return tokenize_rl_data(example, tokenizer)
        except Exception as e:
            print(f"Error tokenizing example: {e}")
            # Return a minimally valid example
            return {
                "input_ids": torch.zeros(512, dtype=torch.long),
                "attention_mask": torch.zeros(512, dtype=torch.long),
                "reference_answer": ""
            }
    
    tokenized_dataset = dataset.map(
        safe_tokenize,
        remove_columns=dataset.column_names
    )
    
    print(f"Dataset after tokenization: {len(tokenized_dataset)} examples")
    if len(tokenized_dataset) > 0:
        print(f"Example keys: {list(tokenized_dataset[0].keys())}")
    
    return tokenized_dataset
