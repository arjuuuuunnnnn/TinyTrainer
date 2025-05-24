from datasets import load_dataset
import json
import os

alpaca = load_dataset("tatsu-lab/alpaca", split="train")

def format_instruction(sample):
    instruction = sample['instruction'].strip()
    input_text = sample['input'].strip()
    output = sample['output'].strip()

    text = f"### Instruction:\n{instruction}"
    if input_text:
        text += f"\n\nInput:\n{input_text}"
    text += f"\n\n### Response:\n{output}"

    return {
        "text": text
    }

formatted_alpaca = alpaca.map(format_instruction, remove_columns=alpaca.column_names)

os.makedirs("data", exist_ok=True)

with open("data/alpaca_sft.jsonl", "w") as f:
    for sample in formatted_alpaca:
        f.write(json.dumps(sample) + "\n")

print(f"Alpaca: {len(formatted_alpaca)}\n")
