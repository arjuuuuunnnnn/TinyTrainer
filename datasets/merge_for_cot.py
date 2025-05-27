import json
import os

os.makedirs("cot_data", exist_ok=True)

input_dir = "data"
output_file = "cot_data/cot_dataset.jsonl"

file_limits = {
    "synthetic_cot.jsonl": 501,
    "mathinstruct.jsonl": 22000
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

