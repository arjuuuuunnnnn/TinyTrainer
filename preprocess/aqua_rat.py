from datasets import load_dataset
import json
import os

aqua_rat = load_dataset("Chinar/AQuA-RAT", split="train")

def format_instruction(sample):
    return {
        "text": (
            f"### Instruction:\n"
            f"{sample['prompt']}\n\n"
            f"### Response:\n"
            f"{sample['completion']}"
        )
    }

formatted_aqua_rat = aqua_rat.map(format_instruction, remove_columns=aqua_rat.column_names)

os.makedirs("datasets/data", exist_ok=True)

with open("datasets/data/aqua_rat_sft.jsonl", "w") as f:
    for sample in formatted_aqua_rat:
        f.write(json.dumps(sample) + "\n")

print(f"AQUA-RAT: {len(formatted_aqua_rat)}\n")
