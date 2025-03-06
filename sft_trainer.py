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
