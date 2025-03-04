from config import *
from model_setup import setup_model, setup_tokenizer
from data_loader import load_sft_data, load_rl_data
from sft_trainer import SFTTrainer
from rl_trainer import RLPPOTrainer
from reward import simple_reward

def main(task_type="rl"):
    model = setup_model()
    tokenizer = setup_tokenizer()
    
    if task_type == "sft":
        sft_data = load_sft_data()
        sft_trainer = SFTTrainer(model, tokenizer, sft_data)
        sft_trainer.train(TRAINING_ARGS)
        model.save_pretrained("./sft_model")
        
    elif task_type == "rl":
        rl_data = load_rl_data()
        rl_trainer = RLPPOTrainer(model, tokenizer, rl_data)
        rl_trainer.train(ppo_config=ppo_config, reward_fn=simple_reward)
        
if __name__ == "__main__":
    main(task_type="rl")
