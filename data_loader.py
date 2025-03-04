from datasets import load_dataset

def load_sft_data():
    return load_dataset("json", data_files="data/sft_data.json")

def load_rl_data():
    return load_dataset("json", data_files="data/rl_data.json")

def tokenize_fn(example, tokenizer):
    return tokenizer(example["prompt"], truncation=True, max_length=512)
