from datasets import load_dataset
import json
import os

dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_instruction(sample):
    instruction = sample['instruction'].strip()
    context = sample['context'].strip()
    response = sample['response'].strip()

    text = f"### Instruction:\n{instruction}"
    if context:
        text += f"\n\nContext:\n{context}"
    text += f"\n\n### Response:\n{response}"

    return {
        "text": text
    }

formatted_dolly = dolly.map(format_instruction, remove_columns=dolly.column_names)

os.makedirs("datasets/data", exist_ok=True)

with open("datasets/data/dolly_sft.jsonl", "w") as f:
    for sample in formatted_dolly:
        f.write(json.dumps(sample) + "\n")

print(f"DOLLY: {len(formatted_dolly)}\n")
