import torch
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM
from data_loader import prepare_rl_dataset
import copy

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = prepare_rl_dataset(dataset['train'], tokenizer)
        print("RL dataset example:", self.dataset)
        print("RL dataset input_ids length:", len(self.dataset[0]['input_ids']))
        
    def train(self, ppo_params, reward_fn):
        print("Starting PPO training...")
        
        # Create a copy of the model to serve as the value model
        print("Creating value model...")
        value_model = copy.deepcopy(self.model)
        print("Value model created successfully")
        
        # Create a reference model (different instance than model)
        print("Creating reference model...")
        ref_model = None  # Start with None for PEFT models
        
        # Check if we're using PEFT/LoRA
        is_peft_model = hasattr(self.model, "is_peft_model") and self.model.is_peft_model
        if is_peft_model:
            print("Detected PEFT/LoRA model, setting ref_model to None")
        else:
            # For non-PEFT models, we need a separate copy
            try:
                ref_model = copy.deepcopy(self.model)
                print("Reference model created successfully")
            except Exception as e:
                print(f"Error creating reference model: {e}")
                # If deepcopy fails, try to load a fresh instance
                try:
                    from model_setup import setup_model
                    ref_model = setup_model()
                    print("Reference model created using setup_model()")
                except Exception as e2:
                    print(f"Error creating reference model using setup_model: {e2}")
                    print("Setting ref_model to None as fallback")
                    ref_model = None
        
        # Create PPOConfig object
        ppo_config = PPOConfig(
            learning_rate=ppo_params["learning_rate"],
            batch_size=ppo_params["batch_size"],
            mini_batch_size=ppo_params["mini_batch_size"],
            gradient_accumulation_steps=ppo_params["gradient_accumulation_steps"],
            gamma=ppo_params["gamma"],
            lam=ppo_params["lam"],
            cliprange=ppo_params["cliprange"],
            cliprange_value=ppo_params["cliprange_value"],
            vf_coef=ppo_params["vf_coef"],
            seed=ppo_params["seed"],
            max_grad_norm=ppo_params["max_grad_norm"],
            output_dir=ppo_params["output_dir"]
        )
        
        print("Initializing PPOTrainer...")
        # Initialize the PPOTrainer with the correct parameter order
        ppo_trainer = PPOTrainer(
            args=ppo_config,                  # First param is args (PPOConfig)
            processing_class=self.tokenizer,  # Second param is processing_class (tokenizer)
            model=self.model,                 # Third param is model
            ref_model=ref_model,              # Fourth param is ref_model (separate instance or None)
            reward_model=None,                # Fifth param is reward_model (we'll handle rewards manually)
            train_dataset=self.dataset,       # Sixth param is train_dataset
            value_model=value_model           # Providing a value model to prevent the error
        )
        
        print("PPOTrainer initialized successfully")
        
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
                            generation = self.model.generate(
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
