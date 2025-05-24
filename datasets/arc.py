from datasets import load_dataset
import json
import os

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

def format_choices(choices):
    formatted = []
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted.append(f"{label}) {choice}")
    return "\n".join(formatted)

def format_instruction(sample):
    choices_text = format_choices(sample['choices'])
    correct_answer = sample['answerKey']

    correct_choice_idx = sample['choices']['label'].index(correct_answer)
    correct_choice_text = sample['choices']['text'][correct_choice_idx]

    return {
        "text": (
            f"### Instruction:\n"
            f"{sample['question']}\n\n"
            f"{choices_text}\n\n"
            f"### Response:\n"
            f"Looking at this question, I need to analyze each option:\n\n"
            f"The correct answer is {correct_answer}) {correct_choice_text}\n\n"
            f"This is because this option best explains the scientific concept or phenomenon described in the question."
        )
    }

formatted_arc = arc.map(format_instruction, remove_columns=arc.column_names)

os.makedirs("data", exist_ok=True)

with open("data/arc_sft.jsonl", "w") as f:
    for sample in formatted_arc:
        f.write(json.dumps(sample) + "\n")

print(f"ARC: {len(formatted_arc)}\n")
