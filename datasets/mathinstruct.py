import json
from datasets import load_dataset
import os

mathinstruct = load_dataset("TIGER-Lab/MathInstruct", split="train")

def format_instruction(sample):
    return {
        "text": (
            f"### Instruction:\n\n"
            f"{sample['instruction']}\n\n"
            f"### Response:\n\n"
            f"{sample['output']}\n\n"
            )
    }

formatted_mathinstruct = mathinstruct.map(format_instruction, remove_columns=mathinstruct.column_names)

os.makedirs("data", exist_ok=True)

with open("data/mathinstruct.jsonl", "w") as f:
    for i in formatted_mathinstruct:
        f.write(json.dumps(i) + "\n")

print(f"MATHINSTRUCT: {len(formatted_mathinstruct)}\n")
