import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "merged_tinyllama_sft"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "what is 5 times 21"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
