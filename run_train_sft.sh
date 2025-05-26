accelerate launch --config_file sft/accelerate_config.yaml sft/train_sft.py
python sft/merge_sft.py

