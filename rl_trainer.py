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
        
        # Safer way to check input_ids shape
        if 'input_ids' in self.dataset[0]:
            input_ids = self.dataset[0]['input_ids']
            if hasattr(input_ids, 'shape'):
                print(f"RL dataset example input_ids shape: {input_ids.shape}")
            else:
                print(f"RL dataset example input_ids type: {type(input_ids)}, converting to tensor...")
                # Convert to tensor if it's not already
                if isinstance(input_ids, list):
                    # Apply conversion to the entire dataset
                    def ensure_tensors(example):
                        if 'input_ids' in example and isinstance(example['input_ids'], list):
                            example['input_ids'] = torch.tensor(example['input_ids'], dtype=torch.long)
                        if 'attention_mask' in example and isinstance(example['attention_mask'], list):
                            example['attention_mask'] = torch.tensor(example['attention_mask'], dtype=torch.long)
                        return example
                    
                    self.dataset = self.dataset.map(ensure_tensors)
                    print("Converted dataset list items to tensors")
        
        if 'attention_mask' in self.dataset[0]:
            attention_mask = self.dataset[0]['attention_mask']
            if hasattr(attention_mask, 'shape'):
                print(f"RL dataset example attention_mask shape: {attention_mask.shape}")
            else:
                print(f"RL dataset example attention_mask type: {type(attention_mask)}")
        else:
            print("RL dataset example attention_mask: Missing")
        
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
                            # Check if all tensors have the same shape
                            shapes = [t.shape for t in tensors]
                            if len(set(shapes)) == 1:  # All shapes are the same
                                collated_batch[key] = torch.stack(tensors)
                            else:
                                # Handle tensors with different shapes
                                max_len = max(t.shape[0] for t in tensors)
                                padded_tensors = []
                                for t in tensors:
                                    if t.shape[0] < max_len:
                                        padding = torch.zeros(max_len - t.shape[0], dtype=t.dtype, device=t.device)
                                        t_padded = torch.cat([t, padding], dim=0)
                                        padded_tensors.append(t_padded)
                                    else:
                                        padded_tensors.append(t)
                                collated_batch[key] = torch.stack(padded_tensors)
                        else:
                            # Convert to tensors if they aren't already
                            try:
                                collated_batch[key] = torch.tensor([t for t in tensors], dtype=torch.long)
                            except Exception as e:
                                print(f"Error converting to tensor: {e}, using first item only")
                                # Fallback to using just the first item
                                collated_batch[key] = torch.tensor([tensors[0]], dtype=torch.long)
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
                            try:
                                generation = self.model.generate(
                                    query, 
                                    max_new_tokens=256,
                                    do_sample=True,
                                    temperature=0.7
                                )
                                response_tensors.append(generation)
                            except Exception as e:
                                print(f"Error generating response: {e}")
                                # Add a dummy response in case of error
                                response_tensors.append(query)  # Use the query as a fallback
                    
                    # Debug the response tensors
                    print(f"Generated {len(response_tensors)} responses")
                    for i, resp in enumerate(response_tensors[:2]):  # Print first 2 for debugging
                        print(f"Response {i} type: {type(resp)}, shape: {resp.shape if hasattr(resp, 'shape') else 'N/A'}")
                    
                    # Decode the generated responses - handle different tensor formats
                    decoded_responses = []
                    for response in response_tensors:
                        try:
                            # Handle different possible formats of the response tensor
                            if isinstance(response, torch.Tensor):
                                # If it's a 2D tensor, take the first sequence
                                if response.dim() > 1:
                                    response_ids = response[0].tolist()
                                else:
                                    response_ids = response.tolist()
                                
                                # Now decode the list of ids
                                decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                            elif isinstance(response, list):
                                # If it's already a list, decode directly
                                decoded = self.tokenizer.decode(response, skip_special_tokens=True)
                            else:
                                print(f"Unknown response type: {type(response)}")
                                decoded = "Error decoding response"
                                
                            decoded_responses.append(decoded)
                        except Exception as e:
                            print(f"Error decoding response: {e}")
                            decoded_responses.append("Error decoding")
                    
                    if decoded_responses:
                        print(f"First decoded response: {decoded_responses[0][:50]}...")
                    
                    # Calculate rewards
                    rewards = []
                    for resp, ref in zip(decoded_responses, reference_answers):
                        try:
                            reward = reward_fn(resp, ref)
                            rewards.append(reward)
                        except Exception as e:
                            print(f"Error calculating reward: {e}")
                            rewards.append(0.0)
                    
                    rewards_tensor = torch.tensor(rewards).to(self.model.device)
                    
                    # Prepare response tensors for PPO step
                    ppo_responses = []
                    for response in response_tensors:
                        if isinstance(response, torch.Tensor):
                            # If it's a 2D tensor with batch dim, keep it
                            if response.dim() > 1 and response.shape[0] == 1:
                                ppo_responses.append(response)
                            # If it's a 1D tensor or batch with > 1, unsqueeze
                            elif response.dim() == 1 or response.shape[0] > 1:
                                # For safety, only use the first sequence if batch > 1
                                if response.dim() > 1 and response.shape[0] > 1:
                                    response = response[0].unsqueeze(0)
                                else:
                                    response = response.unsqueeze(0)
                                ppo_responses.append(response)
                        else:
                            print(f"Skipping response with type {type(response)}")
                    
                    if not ppo_responses:
                        print("No valid responses for PPO step, skipping batch")
                        continue
                    
                    # Make sure query_tensors matches ppo_responses in batch dimension
                    if len(ppo_responses) != query_tensors.shape[0]:
                        print(f"Mismatch in batch size: queries={query_tensors.shape[0]}, responses={len(ppo_responses)}")
                        # Adjust query_tensors to match
                        if query_tensors.shape[0] > len(ppo_responses):
                            query_tensors = query_tensors[:len(ppo_responses)]
                        else:
                            # Duplicate last query to match
                            while len(ppo_responses) > query_tensors.shape[0]:
                                query_tensors = torch.cat([query_tensors, query_tensors[-1:]], dim=0)
                    
                    # Make sure rewards matches the batch size
                    if len(rewards) != len(ppo_responses):
                        print(f"Adjusting rewards tensor to match responses: {len(rewards)} -> {len(ppo_responses)}")
                        if len(rewards) > len(ppo_responses):
                            rewards_tensor = rewards_tensor[:len(ppo_responses)]
                        else:
                            # Pad with zeros
                            padding = torch.zeros(len(ppo_responses) - len(rewards), device=rewards_tensor.device)
                            rewards_tensor = torch.cat([rewards_tensor, padding])
                    
                    try:
                        train_stats = ppo_trainer.step(
                            queries=query_tensors,
                            responses=ppo_responses,
                            scores=rewards_tensor
                        )
                        
                        if batch_idx % 5 == 0:
                            print(f"Epoch {epoch+1}, Batch {batch_idx}, Mean reward: {torch.mean(rewards_tensor).item():.4f}")
                    except Exception as e:
                        print(f"Error in PPO step: {e}")
                        import traceback
                        traceback.print_exc()
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print("PPO training completed.")
        return self.model
