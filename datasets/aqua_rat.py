from datasets import load_dataset
import json

aqua_rat = load_dataset("Chinar/AQuA-RAT", split="train")

def format_instruction(sample):
    return {
        "text": (
            f"###Instrction:\n"
            f"Prompt: {sample['prompt']}\n"
            f"A\n\n"
            f"###Response:\n"
            f"Completion: {sample['completion']}"
        )
    }

formatted_aqua_rat = aqua_rat.map(format_instruction, remove_columns=aqua_rat.column_names)

with open("data/aqua_rat.jsonl", "w") as f:
    for i in formatted_aqua_rat:
        f.write(json.dumps(i) + "\n")


