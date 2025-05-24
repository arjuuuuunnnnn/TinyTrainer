from datasets import load_dataset
import json

svamp = load_dataset("ChilleD/SVAMP", split="train")

def format_instruction(sample):
    return {
        "text": (
            f"### Instruction:\n"
            f"Q: {sample['question_concat']}\n"
            f"Equation: {sample['Equation']}\n"
            f"Type: {sample['Type']}\n"
            f"A:\n\n"
            f"### Response:\n"
            f"{sample['Answer']}"
        )
    }

formatted_svamp = svamp.map(format_instruction, remove_columns=svamp.column_names)

with open("data/svamp.jsonl", "w") as f:
    for i in formatted_svamp:
        f.write(json.dumps(i) + "\n")


