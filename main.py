import torch
import os
from config import *
from model_setup import setup_model, setup_tokenizer
from data_loader import load_sft_data, load_rl_data
from sft_trainer import SFTTrainer
from rl_trainer import RLPPOTrainer
from reward import simple_reward
from trl import PPOConfig


import custom_attention

def main(task_type="rl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./ppo_output", exist_ok=True)
    
    print("Setting up model and tokenizer...")
    model = setup_model()
    tokenizer = setup_tokenizer()
    print("Model and tokenizer setup complete")
    
    try:
        if task_type.lower() == "sft":
            print("Loading SFT data...")
            sft_data = load_sft_data()
            print(f"SFT data loaded with {len(sft_data['train'])} examples")
            
            print("Initializing SFT trainer...")
            sft_trainer = SFTTrainer(model, tokenizer, sft_data)
            
            print("Starting SFT training...")
            sft_trainer.train(TRAINING_ARGS)
            
            print("Saving fine-tuned model...")
            model.save_pretrained("./sft_model")
            tokenizer.save_pretrained("./sft_model")
            print("SFT training complete")
            
        elif task_type.lower() == "rl":
            print("Loading RL data...")
            rl_data = load_rl_data()
            print(f"RL data loaded with {len(rl_data['train'])} examples")
            
            print("Initializing RL trainer...")
            rl_trainer = RLPPOTrainer(model, tokenizer, rl_data)
            
            print("Starting RL training...")
            
            ppo_params = {
                "learning_rate": 1.41e-5,
                "batch_size": 1,
                "mini_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gamma": 1.0,
                "lam": 0.95,
                "cliprange": 0.2,
                "cliprange_value": 0.2,
                "vf_coef": 0.1,
                "seed": 42,
                "max_grad_norm": 0.3,
                "output_dir": "./ppo_output"
            }

            trained_model = rl_trainer.train(ppo_params=ppo_params, reward_fn=simple_reward)
            
            print("Saving RL-trained model...")
            trained_model.save_pretrained("./rl_model")
            tokenizer.save_pretrained("./rl_model")
            print("RL training complete")
        
        else:
            print(f"Unknown task type: {task_type}. Please use 'sft' or 'rl'.")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    task_type = "rl"
    if len(sys.argv) > 1:
        task_type = sys.argv[1]
    
    main(task_type=task_type)
