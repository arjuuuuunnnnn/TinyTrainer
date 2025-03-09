# TinyTrainer - Small Language Model Training Framework

## Overview

This project provides a framework for fine-tuning small language models (LLMs) using two powerful techniques:
1. Supervised Fine-Tuning (SFT)
2. Reinforcement Learning (RL)

The framework is designed to work with resource-constrained environments by using model quantization and parameter-efficient fine-tuning techniques.

## Training Method Comparison

This framework follows the same general training paradigm used by larger models like Deepseek-LLM R1, but optimized for smaller scale:

### Similarities with Deepseek-LLM R1
- Both use the multi-stage training approach: pre-training → supervised fine-tuning → reinforcement learning
- Both implement RLHF-like techniques using PPO for alignment
- Both focus on optimizing language generation capabilities

### Key Differences
- **Scale**: Deepseek-LLM R1 has up to 236B parameters trained on 2 trillion tokens, while this framework targets smaller models like TinyLlama (1.1B parameters)
- **Compute Requirements**: Deepseek used thousands of GPUs; this framework is optimized for single GPU or even CPU training
- **Architecture**: Deepseek uses custom architectural innovations like group query attention; this framework uses standard transformer architecture with memory optimizations
- **Data Volume**: Deepseek was trained on vast quantities of diverse data; this framework is designed for targeted fine-tuning on smaller datasets

This framework provides an accessible way to apply similar training methods used by models like Deepseek, but at a much smaller scale that's manageable for individual researchers and developers.

## Features

- 🚀 Train smaller language models like TinyLlama (1.1B parameters)
- 💾 Low VRAM usage through 4-bit quantization
- 🔄 Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- 📚 Support for both supervised and reinforcement learning approaches
- 🧩 Modular architecture that's easy to customize


## Technical Overview

### Base Model

[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), 1.1 billion parameter language model

### Memory Optimization Techniques

1. 4-bit Quantization (32-bit floating point weights to 4-bit)
2. LoRA

### Training Methods

#### Supervised Fine-Tuning (SFT)
The model learns from examples of prompts and desired completions:
- Input data format: `{"prompt": "...", "completion": "..."}`

#### Reinforcement Learning (RL)
The model learns from feedback on its own outputs:
- Uses Proximal Policy Optimization (PPO) algorithm

## How to Use

### Data Preparation

#### For Supervised Fine-Tuning:
Create a JSON file at `data/sft_data.json` with format:
```json
[
  {
    "prompt": "What is machine learning?",
    "completion": "Machine learning is a subfield of artificial intelligence..."
  },
  {
    "prompt": "Explain quantum computing",
    "completion": "Quantum computing is a type of computing that uses quantum mechanics..."
  }
]
```

#### For Reinforcement Learning:
Create a JSON file at `data/rl_data.json` with format:
```json
[
  {
    "prompt": "What is machine learning?",
    "reference_answer": "Machine learning is a subfield of artificial intelligence..."
  },
  {
    "prompt": "Explain quantum computing",
    "reference_answer": "Quantum computing is a type of computing that uses quantum mechanics..."
  }
]
```

### Running the Training

To run Supervised Fine-Tuning:
```bash
python main.py sft
```

To run Reinforcement Learning:
```bash
python main.py rl
```


## License

This project is open-source and available under the MIT License.
