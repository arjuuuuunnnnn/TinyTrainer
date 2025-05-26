import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

lora_model_path = "step_1_sft"

peft_config = PeftConfig.from_pretrained(lora_model_path)
base_model_name = peft_config.base_model_name_or_path

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, lora_model_path)

model = model.merge_and_unload()

merged_model_path = "merged_tinyllama_sft"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved at: {merged_model_path}")

