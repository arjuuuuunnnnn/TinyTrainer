import torch
from trl import PPOTrainer
from transformers import AutoTokenizer
from data_loader import prepare_rl_dataset

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = prepare_rl_dataset(dataset['train'], tokenizer)
        print("RL dataset example:", self.dataset)
        print("RL dataset input_ids shape", self.dataset[0]['input_ids'].shape)
        
    def train(self, ppo_config, reward_fn):
        print("Starting PPO training...")
        
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset
        )
        
        for epoch in range(1):
            print(f"Starting epoch {epoch+1}/1")
            
            for batch_idx, batch in enumerate(ppo_trainer.dataloader):
                try:
                    query_tensors = batch["input_ids"].to(self.model.device)
                    
                    reference_answers = batch.get("reference_answer", [""] * len(query_tensors))
                    
                    response_tensors = []
                    for query in query_tensors:
                        if query.dim() == 1:
                            query = query.unsqueeze(0)
                            
                        with torch.no_grad():
                            generation = ppo_trainer.generate(
                                query, 
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=0.7
                            )
                            
                        response_tensors.append(generation)
                    
                    if isinstance(response_tensors[0], torch.Tensor):
                        response_tensors = torch.stack(response_tensors)
                    
                    decoded_responses = [
                        self.tokenizer.decode(r, skip_special_tokens=True) 
                        for r in response_tensors
                    ]
                    
                    rewards = [
                        reward_fn(resp, ref) 
                        for resp, ref in zip(decoded_responses, reference_answers)
                    ]
                    
                    rewards_tensor = torch.tensor(rewards).to(self.model.device)
                    
                    train_stats = ppo_trainer.step(
                        queries=query_tensors,
                        responses=response_tensors,
                        scores=rewards_tensor
                    )
                    
                    if batch_idx % 5 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}, Mean reward: {torch.mean(rewards_tensor).item():.4f}")
                        if decoded_responses:
                            print(f"Sample response: {decoded_responses[0][:100]}...")
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print("PPO training completed.")
        return self.model
