from datasets import load_dataset
import json
import os

svamp = load_dataset("ChilleD/SVAMP", split="train")

def format_instruction(sample):
    question = sample['question_concat'].strip()
    equation = sample['Equation'].strip()
    answer = str(sample['Answer']).strip()
    return {
        "text": (
            f"### Instruction:\n"
            f"{question}\n\n"
            f"### Response:\n"
            f"Let me solve this step by step.\n\n"
            f"Setting up the equation: {equation}\n\n"
            f"Solving:\n"
            f"{equation} = {answer}\n\n"
            f"Therefore, the answer is {answer}."
        )
    }

formatted_svamp = svamp.map(format_instruction, remove_columns=svamp.column_names)

os.makedirs("datasets/data", exist_ok=True)

with open("datasets/data/svamp_sft.jsonl", "w") as f:
    for sample in formatted_svamp:
        f.write(json.dumps(sample) + "\n")

print(f"SVAMP: {len(formatted_svamp)}\n")
