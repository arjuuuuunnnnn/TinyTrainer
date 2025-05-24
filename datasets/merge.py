import json
import os

os.makedirs("sft_data", exist_ok=True)

input_dir = "data"
output_file = "sft_data/sft_dataset.jsonl"

file_limits = {
    "alpaca_sft.jsonl": 5000,
    "aqua_rat_sft.jsonl": 15000,
    "arc_sft.jsonl": 1119,
    "boolq_sft.jsonl": 5000,
    "dolly_sft.jsonl": 3000,
    "gsm8k_sft.jsonl": 7473,
    "math_qa_sft.jsonl": 15000,
    "svamp_sft.jsonl": 700
}

total_lines = 0

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename, line_limit in file_limits.items():
        filepath = os.path.join(input_dir, filename)
        lines_written = 0

        if not os.path.exists(filepath):
            print(f"File not found: {filename} - skipping")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as infile:
                print(f"\nProcessing {filename} (limit: {line_limit})...")

                for line in infile:
                    if lines_written >= line_limit:
                        break

                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            outfile.write(line + '\n')
                            lines_written += 1
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON in {filename}, line {lines_written+1}: {e}")
                            continue
                total_lines += lines_written
                print(f"Wrote {lines_written} lines")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"\nMerged {total_lines} total lines into {output_file}")
