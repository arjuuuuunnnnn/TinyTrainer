from datasets import load_dataset
import json

gsm8k = load_dataset("openai/gsm8k", "main", split="train")

def format_instruction(sample):
    return {
            "text": (
                f"### Instruction:\n"
                f"Q: {sample['question']}\n"
                f"A:\n\n"
                f"### Response:\n"
                f"{sample['answer']}"
        )
    }

formatted_gsm8k = gsm8k.map(format_instruction, remove_columns=gsm8k.column_names)

with open("data/gsm8k.jsonl", "w") as f:
    for i in formatted_gsm8k:
        f.write(json.dumps(i) + "\n")
