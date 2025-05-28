import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

sft_lora_path = "checkpoints/step_1_sft"
cot_lora_path = "checkpoints/step_2_cot/cot"

peft_config = PeftConfig.from_pretrained(sft_lora_path)
base_model_name = peft_config.base_model_name_or_path

print(f"Loading base model: {base_model_name}")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, sft_lora_path, adapter_name="sft")

print("Loading CoT LoRA adapter...")
model.load_adapter(cot_lora_path, adapter_name="cot")

model.set_adapter("cot")

print("Model loaded with stacked LoRA adapters!")
print(f"Active adapter: cot (stacked on sft)")
print(f"Available adapters: {list(model.peft_config.keys())}")

def generate_response(prompt, max_new_tokens=256):
    """Generate response using stacked LoRA adapters"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.replace(prompt, "").strip()

    return response



prompt = "what is 10 times 21"

response = generate_response(prompt)

print("Response:\n")
print(response)

