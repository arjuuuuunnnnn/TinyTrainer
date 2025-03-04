from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import DataCollatorForLanguageModeling
from data_loader import tokenize_fn

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        self.tokenizer = tokenizer
        self.dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer))
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
    def train(self, ppo_config, reward_fn):
        ppo_trainer = PPOTrainer(
            model=self.ppo_model,
            config=ppo_config,
            dataset=self.dataset,
            data_collator=self.data_collator,
        )
        
        for batch in ppo_trainer.dataloader:
            # Generate responses
            response_tensors = ppo_trainer.generate(
                batch["input_ids"],
                return_prompt=False,
                max_length=200
            )
            responses = self.tokenizer.batch_decode(response_tensors)
            
            # Compute rewards
            rewards = [reward_fn(resp, batch["reference_answer"]) for resp in responses]
            
            # PPO Step
            stats = ppo_trainer.step(response_tensors, rewards)
            print(f"Step rewards: {rewards}")
