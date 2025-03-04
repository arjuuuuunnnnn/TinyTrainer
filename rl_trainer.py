from trl import PPOTrainer
from transformers import DataCollatorForLanguageModeling
from data_loader import tokenize_fn

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.ppo_model = model  # Remove the unnecessary conversion
        self.tokenizer = tokenizer
        self.dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer))
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
    def train(self, ppo_config, reward_fn):
        # Ensure the model has the required value head
        ppo_trainer = PPOTrainer(
            model=self.ppo_model,
            config=ppo_config,
            dataset=self.dataset['train'],  # Specify train split
            data_collator=self.data_collator,
        )
        
        for epoch in range(ppo_config.ppo_epochs):
            for batch in ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]
                
                # Generate responses
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=200
                )
                
                # Decode responses
                responses = self.tokenizer.batch_decode(response_tensors)
                
                # Calculate rewards
                rewards = [reward_fn(resp, batch["reference_answer"]) for resp in responses]
                
                # Perform PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                print(f"Epoch {epoch}, Step rewards: {rewards}")
