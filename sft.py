import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

with open("sft_config.json") as f:
    cfg = json.load(f)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
model_4bit = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    device_map="auto",
    torch_dtype=torch.bfloat16
)

lora = LoraConfig(**cfg["lora_config"])
model = get_peft_model(model_4bit, lora)

dataset = load_dataset("json", data_files={"train": cfg["dataset_path"]})["train"]

def tokenize(example):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    output = tokenizer(prompt, truncation=True, max_length=cfg["max_length"])
    output["labels"] = output["input_ids"].copy()
    return output

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
    bf16=cfg["use_bf16"],
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

