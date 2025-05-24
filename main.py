# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
