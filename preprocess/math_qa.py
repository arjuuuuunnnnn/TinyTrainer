import os
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import json

def download_and_extract(url, dirr='datasets/artifacts'):
    http = urlopen(url)
    zipfile = ZipFile(BytesIO(http.read()))
    zipfile.extractall(path=dirr)

def format_instruction(f_n):
    with open(f_n, 'r') as f:
        dataset = json.load(f)
        formatted_data = []

        for data in dataset:
            problem = data['Problem'].strip()
            rationale = data['Rationale'].strip()
            correct_answer = data['correct'].strip()

            options_text = ", ".join(data['options'])

            formatted_item = {
                "text": (
                    f"### Instruction:\n"
                    f"{problem}\n\n"
                    f"Options: {options_text}\n\n"
                    f"### Response:\n"
                    f"Let me solve this step by step.\n\n"
                    f"{rationale}\n\n"
                    f"Therefore, the answer is {correct_answer}."
                )
            }
            formatted_data.append(formatted_item)

    print(f"MATH_QA: {len(formatted_data)} samples from {f_n}")
    return formatted_data

def format_files(files_list, output_file="datasets/data/math_qa_sft.jsonl"):

    os.makedirs("datasets/data", exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)

    total_samples = 0

    for file in files_list:
        if os.path.exists(file):
            res = format_instruction(file)

            with open(output_file, "a", encoding='utf-8') as f:
                for item in res:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_samples += len(res)
        else:
            print(f"Warning: {file} not found, skipping...")

    print(f"\nTotal samples formatted: {total_samples}")
    print(f"Dataset saved to: {output_file}")

def main():
    url = "https://math-qa.github.io/math-QA/data/MathQA.zip"
    files_list = ["datasets/artifacts/challenge_test.json", "datasets/artifacts/dev.json", "datasets/artifacts/train.json"]

    download_and_extract(url)
    format_files(files_list)

if __name__ == "__main__":
    main()
