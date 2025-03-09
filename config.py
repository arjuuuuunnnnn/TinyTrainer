from transformers import TrainingArguments, BitsAndBytesConfig
from trl import PPOConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_LORA = True
USE_4BIT = True  # for low vram usage

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"
}

# Quantization configuration
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

# sft config
TRAINING_ARGS = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=False,
)
