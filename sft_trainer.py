from transformers import Trainer
from data_loader import tokenize_fn

class SFTTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer))
        
    def train(self, training_args):
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
        )
        trainer.train()
