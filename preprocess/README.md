### Supervised Fine-Tuning (SFT) Dataset
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

I have uploaded the formatted dataset to the huggingface [hemanthsbanur/Instruction_Response_SFT](https://huggingface.co/datasets/hemanthsbanur/Instruction_Response_SFT) for easy access.

### Chain-of-Thought (CoT) Dataset
Reasoning enhancement dataset with **22.5k samples**:

| Dataset | Samples | Type | Description |
|---------|---------|------|-------------|
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | 22k | Math Reasoning | Step-by-step mathematical solutions |
| Synthetic General CoT | 500 | General Reasoning | Custom generated reasoning chains |

This dataset is also available on Huggingface [hemanthsbanur/CoT](https://huggingface.co/datasets/hemanthsbanur/CoT) for easy access.
