import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Paths to your LoRA adapters
sft_lora_path = "checkpoints/step_1_sft"  # Your SFT LoRA path
cot_lora_path = "cot/cot_output"  # Your CoT LoRA path

# Load config from SFT LoRA to get base model name
peft_config = PeftConfig.from_pretrained(sft_lora_path)
base_model_name = peft_config.base_model_name_or_path

print(f"Loading base model: {base_model_name}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading SFT LoRA adapter...")
# Load SFT LoRA adapter first
model = PeftModel.from_pretrained(base_model, sft_lora_path, adapter_name="sft")

print("Loading CoT LoRA adapter...")
# Load CoT LoRA adapter on top of SFT
model.load_adapter(cot_lora_path, adapter_name="cot")

# Set CoT as active adapter (this stacks on top of SFT)
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
    # Remove input prompt from response
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response

def switch_adapter(adapter_name):
    """Switch between adapters for testing"""
    if adapter_name in model.peft_config.keys():
        model.set_adapter(adapter_name)
        print(f"Switched to adapter: {adapter_name}")
    else:
        print(f"Adapter {adapter_name} not found. Available: {list(model.peft_config.keys())}")

# Interactive inference
def main():
    print("\n" + "="*60)
    print("TinyLlama with Stacked LoRA Adapters - Inference Ready!")
    print("Commands:")
    print("  - Type your prompt for generation")
    print("  - 'switch sft' or 'switch cot' to change active adapter")
    print("  - 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input(f"\n[Active: {model.active_adapter}] Enter prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('switch '):
                adapter_name = user_input.split()[1]
                switch_adapter(adapter_name)
                continue
            
            if not user_input:
                print("Please enter a valid prompt.")
                continue
            
            print("\nGenerating response...")
            response = generate_response(user_input)
            
            print(f"\nResponse: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# Test both adapters
def test_adapters():
    """Test both adapters with the same prompt"""
    test_prompt = "Explain machine learning step by step:"
    
    print("\n" + "="*50)
    print("Testing Both Adapters:")
    print("="*50)
    
    # Test SFT adapter
    print("\n--- Testing SFT Adapter ---")
    model.set_adapter("sft")
    sft_response = generate_response(test_prompt)
    print(f"SFT Response: {sft_response}")
    
    # Test CoT adapter (stacked)
    print("\n--- Testing CoT Adapter (Stacked) ---")
    model.set_adapter("cot")
    cot_response = generate_response(test_prompt)
    print(f"CoT Response: {cot_response}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # Option 1: Interactive mode
    main()
    
    # Option 2: Uncomment to test both adapters
    # test_adapters()
