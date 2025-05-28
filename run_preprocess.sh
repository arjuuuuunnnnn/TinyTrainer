#!/bin/bash

SCRIPT_DIR="./datasets"

EXCLUDE_FILES=("merge_for_sft.py" "merge_for_cot.py")

echo "Running data preprocessing scripts..."

for file in "$SCRIPT_DIR"/*.py; do
    filename=$(basename "$file")

    if [[ " ${EXCLUDE_FILES[@]} " =~ " $filename " ]]; then
        continue
    fi

    echo "Running $filename..."
    python3 "$file"
    echo "dataset formatted and saved into <saas> directory"
    echo "----------------------------------------------"
done

echo "All formatting scripts completed."
echo "Now starting the merging process..."
echo "----------------------------------------------"


for merge_file in "${EXCLUDE_FILES[@]}"; do
    echo "Running $merge_file..."
    python3 "$SCRIPT_DIR/$merge_file"
done

echo "Removing temporarily created data directories"
rm -r datasets/data && rm -r datasets/artifacts

echo "Done merging. All datasets are now ready for training and saved in 'datasets' directory"

