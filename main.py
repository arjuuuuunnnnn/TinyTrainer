import torch
import os
from config import *
from model_setup import setup_model, setup_tokenizer
from data_loader import load_sft_data, load_rl_data
from sft_trainer import SFTTrainer
from rl_trainer import RLPPOTrainer
from reward import simple_reward

def main(task_type="rl"):
    """
    Main function to run either supervised fine-tuning or PPO reinforcement learning
    
    Args:
        task_type: "sft" for supervised fine-tuning or "rl" for reinforcement learning
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories if they don't exist
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./ppo_output", exist_ok=True)
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model = setup_model()
    tokenizer = setup_tokenizer()
    print("Model and tokenizer setup complete")
    
    try:
        if task_type.lower() == "sft":
            # Supervised Fine-Tuning
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
            # Reinforcement Learning with PPO
            print("Loading RL data...")
            rl_data = load_rl_data()
            print(f"RL data loaded with {len(rl_data['train'])} examples")
            
            print("Initializing RL trainer...")
            rl_trainer = RLPPOTrainer(model, tokenizer, rl_data)
            
            print("Starting RL training...")
            trained_model = rl_trainer.train(ppo_config=ppo_config, reward_fn=simple_reward)
            
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
    
    # Get task type from command line if provided
    task_type = "rl"  # Default
    if len(sys.argv) > 1:
        task_type = sys.argv[1]
    
    main(task_type=task_type)
