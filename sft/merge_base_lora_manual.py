import json
import torch
import os
import shutil
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

sft_config = "sft/sft_config.json"
lora_adapter_config = "step_1_sft/adapter_config.json"
tok_loc = "step_1_sft"

def merge_lora_weights_manual():
    with open(sft_config, "r") as f:
        cfg = json.load(f)

    with open(lora_adapter_config, "r") as f:
        adapter_config = json.load(f)

    print(f"Base model: {cfg['model_name']}")
    print(f"LoRA rank: {adapter_config['r']}")
    print(f"LoRA alpha: {adapter_config['lora_alpha']}")

    #load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    #load tok
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_loc)
    except:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #load lora adap wts
    adapter_weights = load_file("step_1_sft/adapter_model.safetensors")

    print(f"Found {len(adapter_weights)} LoRA weight tensors")

    r = adapter_config["r"]
    alpha = adapter_config["lora_alpha"]
    scaling = alpha / r
    print(f"LoRA scaling factor: {scaling}")

    # grp lora weights by layers
    lora_pairs = {}
    for key in adapter_weights.keys():
        #get the base layer name
        if '.lora_A.' in key:
            base_name = key.replace('.lora_A.', '.').replace('.weight', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['A'] = adapter_weights[key]
        elif '.lora_B.' in key:
            base_name = key.replace('.lora_B.', '.').replace('.weight', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['B'] = adapter_weights[key]

    print(f"Found {len(lora_pairs)} LoRA layer pairs to merge")

    #apply lora weights to model
    merged_count = 0
    model_dict = dict(model.named_parameters())

    for base_name, lora_weights in lora_pairs.items():
        if 'A' in lora_weights and 'B' in lora_weights:
            # Find the corresponding model parameter
            param_name = base_name + '.weight'
            if param_name in model_dict:
                param = model_dict[param_name]

                # Calculate LoRA delta: delta_W = B @ A * scaling
                lora_A = lora_weights['A'].to(torch.float32)
                lora_B = lora_weights['B'].to(torch.float32)
                delta_W = torch.mm(lora_B, lora_A) * scaling
                
                # Add to original weights
                with torch.no_grad():
                    param.data = param.data.to(torch.float32)
                    param.data += delta_W.to(param.device)
                    param.data = param.data.to(torch.float16)
                
                merged_count += 1
                print(f"Merged LoRA weights for: {param_name}")
    
    print(f"merged {merged_count} layers")
    
    # Save merged model
    print("Saving merged model...")
    output_dir = "merged_model"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # Create model info file
    model_info = {
        "base_model": cfg["model_name"],
        "lora_rank": r,
        "lora_alpha": alpha,
        "merged_layers": merged_count,
        "model_type": "merged_lora"
    }
    
    with open(f"{output_dir}/merge_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Merged model saved to: {output_dir}")

    print("Creating zip file...")
    shutil.make_archive("merged_model", 'zip', output_dir)
    
    # # Copy to Kaggle output if available
    # if os.path.exists('/kaggle/output'):
    #     shutil.copy("merged_model.zip", '/kaggle/output/')
    #     print("Zip file copied to /kaggle/output/ for download")
    # 
    # print("Manual LoRA merge completed successfully!")
    
    # # Test the merged model
    # print("\nTesting merged model...")
    # try:
    #     test_prompt = "Hello, how are you?"
    #     inputs = tokenizer(test_prompt, return_tensors="pt")
    #     
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_new_tokens=30,
    #             do_sample=True,
    #             temperature=0.7,
    #             pad_token_id=tokenizer.eos_token_id
    #         )
    #     
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print(f"Test generation: {generated_text}")
    # except Exception as e:
    #     print(f"Test generation failed: {e}")
    # 
    # return model, tokenizer

# Run the merge
try:
    merged_model, merged_tokenizer = merge_lora_weights_manual()
    print("SUCCESS: LoRA weights merged successfully!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
