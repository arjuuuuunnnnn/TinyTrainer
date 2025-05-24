from datasets import load_dataset
import json

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

def format_instruction(sample):
    return {
            "text": (
                f"### Instruction:\n"
                f"Question: {sample['question']}\n"
                f"Choices: {sample['choices']}\n"
                f"A:\n\n"
                f"### Response:\n"
                f"answer: {sample['answerKey']}\n"
            )
    }

formatted_arc = arc.map(format_instruction, remove_columns=arc.column_names)

with open("data/arc.jsonl", "w") as f:
    for i in formatted_arc:
        f.write(json.dumps(i) + "\n")

