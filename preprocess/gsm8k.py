from datasets import load_dataset
import json
import os

gsm8k = load_dataset("openai/gsm8k", "main", split="train")

def format_instruction(sample):
    return {
        "text": (
            f"### Instruction:\n"
            f"{sample['question']}\n\n"
            f"### Response:\n"
            f"{sample['answer']}"
        )
    }

formatted_gsm8k = gsm8k.map(format_instruction, remove_columns=gsm8k.column_names)
os.makedirs("datasets/data", exist_ok=True)

with open("datasets/data/gsm8k_sft.jsonl", "w") as f:
    for sample in formatted_gsm8k:
        f.write(json.dumps(sample) + "\n")

print(f"GSM8K: {len(formatted_gsm8k)}\n")
