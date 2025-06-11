# TinyTrainer

This pipeline demonstrates a systematic approach to fine-tuning small language models for enhanced reasoning and instruction-following capabilities.


## Project Overview

TinyTrainer implements a multi-stage training approach to enhance TinyLlama's reasoning and instruction-following capabilities through:

1. **Dataset Curation & Formatting** - Custom preprocessing pipelines for diverse datasets
2. **Supervised Fine-tuning (SFT)** - Multi-domain instruction following training
3. **Chain-of-Thought (CoT) Training** - Enhanced reasoning capability development
4. **Preference Data Generation** - Collecting multi-response data for alignment (planned)
5. **DPO Training** - Direct Preference Optimization for model alignment (planned)

### Key Features
- **Modular Pipeline**: Each training stage is independent and configurable
- **LoRA Integration**: Memory-efficient training using Low-Rank Adaptation and Quantization
- **Multi-GPU Support**: Distributed training with Accelerate
- **Stacked Adapters**: Progressive capability building through adapter stacking
- **Comprehensive Datasets**: 78.5k samples across math, science, and general reasoning

## Training Pipeline

### Current Status
- [x] Dataset Curation & Formatting
- [x] Supervised Fine-tuning (SFT)
- [x] Chain-of-Thought (CoT) Training
- [ ] Preference Data Generation
- [ ] DPO Training

## Installation & Setup

### Working Environment
- Python 3.8+
- 2 x P100 Kaggle GPUs

### Environment Setup
```bash
git clone <repository-url>
cd TinyTrainer

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes
```

### Accelerate Configuration
Configure multi-GPU training:
```bash
accelerate config
```

Or use the provided configuration:
```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
machine_rank: 0
num_machines: 1
gpu_ids: 0,1
use_cpu: false
mixed_precision: fp16
```

## Dataset Preparation

This stage involves custom preprocessing pipelines for various datasets

Please read the [Dataset Curation & Formatting README](https://github.com/arjuuuuunnnnn/TinyTrainer/blob/master/preprocess/README.md) for more details.

To download and prepare datasets, run the following:
```bash
chmod +x run_preprocess.sh
./run_preprocess.sh
```
Or you can use the same dataset which I have in the Huggingface
1. [Instruction Response SFT](https://huggingface.co/datasets/hemanthsbanur/Instruction_Response_SFT)
2. [Chain-of-Thought](https://huggingface.co/datasets/hemanthsbanur/CoT)

## Training Stages

### Stage 1: Supervised Fine-Tuning (SFT)

Train the base model on instruction-following tasks:

```bash
chmod +x run_train_sft.sh
./run_train_sft.sh
```

### Stage 2: Chain-of-Thought (CoT) Training

Build reasoning capabilities on top of the SFT model:

```bash
chmod +x run_train_cot.sh  
./run_train_cot.sh
```

**Key Features**:
- **Adapter Stacking**: CoT adapter stacked on frozen SFT adapter
- **Progressive Learning**: Builds reasoning on top of instruction-following
- **Memory Efficient**: Only CoT parameters are trainable


### Model Merging & Testing

#### Test SFT Model
```bash
python sft/merge_base_lora.py

python sft/test_sft.py
```

#### Test Stacked Model (SFT + CoT)
```bash
python cot/stack_base_sft_cot.py
```

## Model Evaluation

### Basic Testing
The project includes simple evaluation scripts:

```python
prompt = "what is 10 times 21"
response = generate_response(prompt)
print(response)
```

### Custom Evaluation
Create your own evaluation scripts:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(model, "checkpoints/step_1_sft")
model.load_adapter("checkpoints/step_2_cot", adapter_name="cot")

def evaluate_model(prompts):
    # Your evaluation logic here
    pass
```

### Memory Optimization
The pipeline uses several techniques to reduce memory usage:
- **4/8-bit Quantization**: BitsAndBytesConfig with NF4
- **LoRA**: Low-rank adaptation instead of full fine-tuning
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision**: FP16 training

## Results & Performance

### Training Metrics
- **SFT Training**: ~3 epochs on 56k samples
- **CoT Training**: ~3 epochs on 22.5k samples  
- **Total Training Time**: ~6-8 hours on 2x RTX 4090

### Model Capabilities
After training, the model demonstrates improved:
- **Instruction Following**: Better adherence to user requests
- **Mathematical Reasoning**: Step-by-step problem solving
- **Chain-of-Thought**: Explicit reasoning processes
- **Multi-domain Knowledge**: Performance across math, science, and general tasks

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in config files
"train_batch_size": 2  # Instead of 4
"gradient_accumulation_steps": 8  # Increase to maintain effective batch size
```

#### Slow Training
```bash
# Enable optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
```

#### Checkpoint Loading Issues
```python
# Ensure correct paths in config files
"sft_lora_path": "checkpoints/step_1_sft"  # Must exist before CoT training
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **TinyLlama Team**: For the excellent base model
- **Hugging Face**: For transformers and datasets libraries  
- **PEFT Team**: For efficient parameter-efficient fine-tuning
- **Dataset Creators**: All the dataset authors whose work made this possible

