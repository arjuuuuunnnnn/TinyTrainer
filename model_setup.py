from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import LORA_CONFIG, BNB_CONFIG, MODEL_NAME, USE_LORA, USE_4BIT

def setup_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG if USE_4BIT else None,
        device_map="auto"
    )
    
    if USE_LORA:
        peft_config = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model

def setup_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)
