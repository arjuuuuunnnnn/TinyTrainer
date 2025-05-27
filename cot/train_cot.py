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
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)
from datasets import load_dataset
import os

with open("cot/cot_config.json", "r") as f:
    cfg = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

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



if is_distributed:
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map={"": local_rank},
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

#load sft lora adapter and stack upon base model
sft_model = PeftModel.from_pretrained(
    base_model,
    cfg["sft_lora_path"],
    adapter_name="sft"
)

# freeze sft lora wts
for name, param in sft_model.named_parameters():
    if "sft" in name:
        param.requires_grad = False

sft_model = prepare_model_for_kbit_training(sft_model)

cot_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    **cfg["cot_lora_config"]
)

sft_model.add_adapter("cot", cot_lora_config)

sft_model.set_adapter("cot")

print("Model parameter summary:")
sft_model.print_trainable_parameters()

trainable_params = sum(p.numel() for p in sft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in sft_model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


dataset = load_dataset(cfg["dataset_path"], split="train")

def tokenize_cot(sample):
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

tokenized_dataset = dataset.map(tokenize_cot, remove_columns=dataset.column_names)


training_args = TrainingArguments(
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
    gradient_checkpointing=True,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=sft_model,
    train_dataset=tokenized_dataset,
    args=training_args,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

print("Saving CoT LoRA adapter...")

sft_model.save_pretrained(
    cfg["cot_output_dir"],
    selected_adapters=["cot"]  #to save only cot adapter
)

tokenizer.save_pretrained(cfg["cot_output_dir"])

print(f"CoT LoRA adapter saved to: {cfg['cot_output_dir']}")
print("\nTraining complete!")

