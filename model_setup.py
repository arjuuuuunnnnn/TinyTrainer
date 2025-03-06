import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from config import LORA_CONFIG, BNB_CONFIG, MODEL_NAME, USE_LORA, USE_4BIT

def setup_model():
    print(f"Loading base model: {MODEL_NAME}")
    
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    
    try:
        if USE_4BIT:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=model_config,
                quantization_config=BNB_CONFIG,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=model_config,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        if USE_LORA:
            print("Applying LoRA adapters")
            peft_config = LoraConfig(**LORA_CONFIG)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def setup_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
