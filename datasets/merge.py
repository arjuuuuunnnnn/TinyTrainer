import json
import os

input_dir = "data"
output_file = "sft_data/sft_data.jsonl"

total_files = len(os.listdir(input_dir))
processed_files = 0

with open(output_file, 'w') as outfile:
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        try:
                            json_obj = json.loads(line)
                            outfile.write(line + '\n')
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON in {filename}: {e}")
                            continue
            processed_files += 1
            print(f"Processed {processed_files}/{total_files}: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"\nmerged {processed_files} files into {output_file}")
print(f"total lines: {sum(1 for _ in open(output_file))}")
