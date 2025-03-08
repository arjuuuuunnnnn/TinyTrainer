from transformers import Trainer, TrainingArguments
from data_loader import tokenize_sft_data

class SFTTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        
        self.processed_dataset = dataset.map(
            lambda x: tokenize_sft_data(x, tokenizer),
            batched=False,
            remove_columns=dataset["train"].column_names
        )
        print("Processed dataset example:", self.processed_dataset["train"][0])
        print("Processed dataset input_ids shape:", self.processed_dataset["train"][0]["input_ids"].shape) # debugg
        
    def train(self, training_args):
        print("Starting supervised fine-tuning...")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.processed_dataset["train"],
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        print("Supervised fine-tuning completed.")
        return self.model
