# TinyTrainer

This pipeline demonstrates a systematic approach to fine-tuning small language models for enhanced reasoning and instruction-following capabilities.

## Project Overview

This project implements a multi-stage training approach to enhance TinyLlama's reasoning and instruction-following capabilities through:

1. **Dataset Curation & Formatting** - Custom preprocessing pipelines for diverse datasets
2. **Supervised Fine-tuning (SFT)** - Multi-domain instruction following training
3. **Chain-of-Thought (CoT) Training** - Enhanced reasoning capability development
4. **Preference Data Generation** - Collecting multi-response data for alignment(planned)
4. **DPO Training** - Direct Preference Optimization for model alignment(planned)

## Training Pipeline Status

[x] Dataset Curation & Formatting
[x] Supervised Fine-tuning (SFT)
[x] Chain-of-Thought (CoT) Training
[] Preference Data Generation
[] DPO Training

## 📋 Dataset Details

### Stage 1: Supervised Fine-tuning (SFT)
Multi-domain instruction following dataset combining **56k samples**:

| Dataset | Samples | Domain | Description |
|---------|---------|--------|-------------|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 7k | Math | Grade school math word problems |
| [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) | 1k | Math | Simple math word problems with variations |
| [AQuA-RAT](https://huggingface.co/datasets/Chinar/AQuA-RAT) | 15k | Math | Algebraic reasoning with rationales |
| [Math-QA](https://huggingface.co/datasets/allenai/math_qa) | 15k | Math | Mathematical reasoning questions |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | 5k | Reading | Boolean question answering |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | 7k | Science | Grade-school science questions |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | 5k | General | Instruction following tasks |
| [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 3k | General | Human-generated instructions |

I have uploaded the formatted dataset to the huggingface ![hemanthsbanur/Instruction_Response_SFT](https://huggingface.co/datasets/hemanthsbanur/Instruction_Response_SFT) for easy access.

### Stage 2: Chain-of-Thought (CoT) Training  
Reasoning enhancement dataset with **22.5k samples**:

| Dataset | Samples | Type | Description |
|---------|---------|------|-------------|
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | 22k | Math Reasoning | Step-by-step mathematical solutions |
| Synthetic General CoT | 500 | General Reasoning | Custom generated reasoning chains |

This dataset is also available on Huggingface ![hemanthsbanur/CoT](https://huggingface.co/datasets/hemanthsbanur/CoT) for easy access.


