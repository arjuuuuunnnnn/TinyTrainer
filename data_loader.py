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
                          padding="max_length", return_tensors="pt")
    print("Tokenized input_ids shape:", tokenized["input_ids"].shape)
    print("Tokenized attention_mask shape:", tokenized["attention_mask"].shape)
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def tokenize_rl_data(example, tokenizer):
    tokenized = tokenizer(example["prompt"], truncation=True, max_length=512, 
                          padding="max_length", return_tensors="pt")

    tokenized["reference_answer"] = example.get("reference_answer", "")
    
    return {k: v[0] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, torch.Tensor)) else v 
            for k, v in tokenized.items()}

def prepare_rl_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda x: tokenize_rl_data(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset
