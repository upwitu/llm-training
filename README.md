# Unsloth SFT Training (DDP)

SFT training pipeline using:

- Unsloth
- HuggingFace Transformers
- TRL SFTTrainer
- LoRA / PEFT
- PyTorch DDP

Supports:

- Multi-GPU training with `torchrun`
- Chat template datasets
- Tool calling datasets
- Response-only loss training


## Setup

Install dependencies:

```bash
uv sync
````

Run training:

```bash
bash train.sh
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    sft.py
```

## Configuration

All training settings are configured in:

```text
ddp_config.yml
```

Update this file for:

* Dataset path
* Model name
* Sequence length
* LoRA settings
* Training hyperparameters
* Output directory
* W&B settings

Example:

```yaml
datasets:
  sample:
    path: "data/sample.jsonl"


model:
  name: "unsloth/Qwen3.5-0.8B"
  max_seq_length: 16384


training:
  response_format:
    instruction_part: "<|im_start|>user\n"
    response_part: "<|im_start|>assistant\n"
```

`response_format` controls response-only training.

The tokens between:

```text
instruction_part
```

and

```text
response_part
```

are masked and ignored during training loss calculation.

Change these values when using different chat templates (Qwen, Llama, Gemma, etc.).

