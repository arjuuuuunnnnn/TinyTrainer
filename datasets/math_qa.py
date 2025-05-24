import os
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import json

def download_and_extract(url, dirr='artifacts'):
    http = urlopen(url)
    zipfile = ZipFile(BytesIO(http.read()))
    zipfile.extractall(path=dirr)

def format_instruction(f_n):
    with open(f_n, 'r') as f:
        dataset = json.load(f)
        formatted_data = []
        for data in dataset:
            formatted_item = {
                f"text": (
                    f"### Instruction:\n"
                    f"Problem: {data['Problem']}\n"
                    f"Rationale: {data['Rationale']}\n"
                    f"options: {data['options']}\n"
                    f"category: {data['category']}\n"
                    f"annotated_formula: {data['annotated_formula']}\n"
                    f"Linear Formula: {data['linear_formula']}\n"
                    f"### Response:\n\n"
                    f"Rationale: {data['Rationale']}\n"
                    f"Answer: {data['correct']}\n"
                )
            }
            formatted_data.append(formatted_item)
        return formatted_data

def format_files(files_list):
    for file in files_list:
        res = format_instruction(file)
        for item in res:
            with open("data/math_qa.jsonl", "a") as f:
                f.write(json.dumps(item) + "\n")

url="https://math-qa.github.io/math-QA/data/MathQA.zip"
files_list = ["artifacts/challenge_test.json", "artifacts/dev.json", "artifacts/train.json"]

download_and_extract(url)
format_files(files_list)
