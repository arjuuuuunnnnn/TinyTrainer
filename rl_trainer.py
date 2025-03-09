import torch
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM
from data_loader import prepare_rl_dataset
import copy

class RLPPOTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        
        # Process the dataset and add more validation
        print("Preparing RL dataset...")
        self.dataset = prepare_rl_dataset(dataset['train'], tokenizer)
        
        # Check if dataset is valid
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty after processing")
        
        # Print dataset info for debugging
        print(f"RL dataset size: {len(self.dataset)}")
        print(f"RL dataset keys: {list(self.dataset[0].keys())}")
        print(f"RL dataset example input_ids shape: {self.dataset[0]['input_ids'].shape if 'input_ids' in self.dataset[0] else 'Missing'}")
        print(f"RL dataset example attention_mask: {self.dataset[0]['attention_mask'].shape if 'attention_mask' in self.dataset[0] else 'Missing'}")
        
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
        
        # Create a dummy reward model that returns the rewards we calculate
        class DummyRewardModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Add a dummy parameter to make it a proper module
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, *args, **kwargs):
                # This won't be called by PPOTrainer
                return None
        
        dummy_reward_model = DummyRewardModel()
        
        # Create our own dataloader instead of relying on PPOTrainer's
        from torch.utils.data import DataLoader
        
        # Create a custom collate function that ensures all required keys are present
        def collate_fn(batch):
            # Extract all possible keys from the batch
            keys = set().union(*[sample.keys() for sample in batch])
            
            # Initialize the collated batch
            collated_batch = {}
            
            # For each key, collect the values from all samples
            for key in keys:
                if key in ['input_ids', 'attention_mask']:
                    # These should be tensors - stack them
                    tensors = [sample[key] for sample in batch if key in sample]
                    if tensors:
                        if isinstance(tensors[0], torch.Tensor):
                            collated_batch[key] = torch.stack(tensors)
                        else:
                            # Convert to tensors if they aren't already
                            collated_batch[key] = torch.tensor([t for t in tensors])
                elif key == 'reference_answer':
                    # For text fields, just collect them
                    collated_batch[key] = [sample.get(key, "") for sample in batch]
                else:
                    # For other fields, include them if they're in all samples
                    if all(key in sample for sample in batch):
                        collated_batch[key] = [sample[key] for sample in batch]
            
            return collated_batch
        
        # Create a custom DataLoader with the collate function
        dataloader = DataLoader(
            self.dataset,
            batch_size=ppo_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Initialize the PPOTrainer 
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=ref_model,
            reward_model=dummy_reward_model,
            train_dataset=self.dataset,
            value_model=value_model
        )
        
        # Replace the PPOTrainer's dataloader with our custom one
        ppo_trainer.dataloader = dataloader
        
        print("PPOTrainer initialized successfully")
        
        for epoch in range(1):
            print(f"Starting epoch {epoch+1}/1")
            
            for batch_idx, batch in enumerate(dataloader):  # Use our custom dataloader instead
                try:
                    print(f"Processing batch {batch_idx}, keys: {batch.keys()}")
                    
                    # Check if input_ids exists in the batch
                    if 'input_ids' not in batch:
                        print(f"Warning: No input_ids in batch {batch_idx}, skipping")
                        continue
                    
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
