from trl import PPOTrainer
from transformers import DataCollatorForLanguageModeling
from data_loader import tokenize_fn

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.ppo_model = model
        self.tokenizer = tokenizer
        self.dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer))
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
    def train(self, ppo_config, reward_fn):
        # Prepare PPO Trainer arguments based on the config
        ppo_trainer_args = {
            "model": self.ppo_model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset['train'],
            "data_collator": self.data_collator,
            "batch_size": ppo_config.batch_size,
            "learning_rate": ppo_config.learning_rate,
            "max_length": 512,  # Add a max length for generation
            "output_dir": ppo_config.output_dir,
        }
        
        # Create PPO Trainer
        ppo_trainer = PPOTrainer(**ppo_trainer_args)
        
        # Training loop
        for epoch in range(1):  # You might want to use ppo_config.num_train_epochs if available
            for batch in ppo_trainer.dataloader:
                try:
                    # Prepare query tensors
                    query_tensors = batch["input_ids"].to(self.ppo_model.device)
                    
                    # Generate responses
                    response_tensors = ppo_trainer.generate(
                        query_tensors,
                        max_new_tokens=200
                    )
                    
                    # Decode responses
                    responses = self.tokenizer.batch_decode(response_tensors)
                    
                    # Calculate rewards (assuming batch contains 'reference_answer')
                    rewards = [reward_fn(resp, batch.get("reference_answer", "")) for resp in responses]
                    
                    # Perform PPO step
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                    
                    print(f"Epoch step - Rewards: {rewards}")
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
