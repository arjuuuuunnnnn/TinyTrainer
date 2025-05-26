import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

with open("sft/sft_config.json", "r") as f:
    cfg = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

if is_distributed:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
else:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

# is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if is_distributed:
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map={"": local_rank},
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
else:#for single GPU
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

model = prepare_model_for_kbit_training(model)

lora = LoraConfig(**cfg["lora_config"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

dataset = load_dataset(cfg["dataset_path"], split="train")

def tokenize(sample):
    prompt = sample["text"]
    encoded = tokenizer(
        prompt, 
        truncation=True, 
        max_length=cfg["max_length"],
        padding=False,
        return_tensors=None
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    num_train_epochs=cfg["num_train_epochs"],
    logging_steps=cfg["logging_steps"],
    learning_rate=cfg["learning_rate"],
    save_strategy=cfg["save_strategy"],
    save_total_limit=cfg["save_total_limit"],
    fp16=cfg["use_fp16"],
    report_to="none",
    dataloader_drop_last=True,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False if is_distributed else None,
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    processing_class=tokenizer,
    data_collator=data_collator
)
# tokenizer=tokenizer

trainer.train()

# model = model.merge_and_unload()

model.save_pretrained(f"{cfg['artifacts']}")
tokenizer.save_pretrained(f"{cfg['artifacts']}")
