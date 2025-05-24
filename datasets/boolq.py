from datasets import load_dataset
import json

boolq = load_dataset("google/boolq", split="train")

def format_instruction(sample):
    return {
            "text": (
                f"### Instruction:\n"
                f"Question: {sample['question']}\n"
                f"A:\n\n"
                f"### Response:\n"
                f"answer: {sample['answer']}\n"
                f"passage: {sample['passage']}\n"
            )
    }

formatted_boolq = boolq.map(format_instruction, remove_columns=boolq.column_names)

with open("data/boolq.jsonl", "w") as f:
    for i in formatted_boolq:
        f.write(json.dumps(i) + "\n")



