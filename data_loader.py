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
    if "prompt" not in example:
        print(f"Warning: Missing 'prompt' in example: {example.keys()}")
        example["prompt"] = "This is a placeholder prompt."
    
    tokenized = tokenizer(example["prompt"], truncation=True, max_length=512, 
                         padding="max_length", return_tensors="pt")
    
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor):
            tokenized[key] = tokenized[key].squeeze(0)
    
    tokenized["reference_answer"] = example.get("reference_answer", "")
    
    return tokenized

def prepare_rl_dataset(dataset, tokenizer):
    print(f"Dataset before tokenization: {len(dataset)} examples")
    print(f"Dataset column names: {dataset.column_names}")
    
    def safe_tokenize(example):
        try:
            result = tokenize_rl_data(example, tokenizer)
            
            if 'input_ids' in result and not isinstance(result['input_ids'], torch.Tensor):
                result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            
            if 'attention_mask' in result and not isinstance(result['attention_mask'], torch.Tensor):
                result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
                
            return result
        except Exception as e:
            print(f"Error tokenizing example: {e}")

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
        for key in tokenized_dataset[0]:
            if key != "reference_answer":
                value = tokenized_dataset[0][key]
                print(f"Key {key} type: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    
    return tokenized_dataset
