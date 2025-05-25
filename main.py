import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


quantization_config = BitsAndBytesConfig(load_in_4bit=True)

with open("config.json", "r") as f:
    cfg = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_4bit = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    device_map="auto", # to change
    torch_dtype=torch.bfloat16, # to change
    quantization_config=quantization_config
)

lora = LoraConfig(**cfg["lora_config"])
model = get_peft_model(model_4bit, lora)
model.print_trainable_parameters()

dataset = load_dataset(cfg["dataset_path"], split="train")

def tokenize(example):
    prompt = example["text"]

    output = tokenizer(
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
    bf16=cfg["use_bf16"],
    report_to="none",
    dataloader_drop_last=True,
    remove_unused_columns=False
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
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# model = model.merge_and_unload()

model.save_pretrained(f"{cfg['artifacts']}")
tokenizer.save_pretrained(f"{cfg['artifacts']}")
