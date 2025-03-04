from datasets import load_dataset

def load_sft_data():
    # Replace with your dataset (example)
    return load_dataset("json", data_files="sft_data.json")

def load_rl_data():
    # Format: {"prompt": "...", "reference_answer": "..."}
    return load_dataset("json", data_files="rl_data.json")

def tokenize_fn(example, tokenizer):
    return tokenizer(example["prompt"], truncation=True, max_length=512)
