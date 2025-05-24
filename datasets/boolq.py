from datasets import load_dataset
import json
import os

boolq = load_dataset("google/boolq", split="train")

def format_instruction(sample):
    answer = "Yes" if sample['answer'] else "No"
    
    return {
        "text": (
            f"### Instruction:\n"
            f"Based on the following passage, answer the question with Yes or No.\n\n"
            f"Passage: {sample['passage']}\n\n"
            f"Question: {sample['question']}\n\n"
            f"### Response:\n"
            f"Looking at the passage, I need to determine if the statement is true or false.\n\n"
            f"Based on the information provided in the passage, the answer is: **{answer}**\n\n"
            f"This conclusion is drawn from the relevant details in the text that directly address the question."
        )
    }

formatted_boolq = boolq.map(format_instruction, remove_columns=boolq.column_names)

os.makedirs("data", exist_ok=True)

with open("data/boolq_sft.jsonl", "w") as f:
    for sample in formatted_boolq:
        f.write(json.dumps(sample) + "\n")

print(f"BOOLQ: {len(formatted_boolq)}\n")
