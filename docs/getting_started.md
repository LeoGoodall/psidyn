# Getting Started

## Installation

### Basic Installation

```bash
pip install psidyn
```

### With GPU Quantization (Recommended)

For running larger models efficiently on GPU:

```bash
pip install psidyn[quantization]
```

This installs `bitsandbytes` and `accelerate` for 4-bit/8-bit model quantization.

### Development Installation

```bash
git clone https://github.com/LeoGoodall/psidyn.git
cd psidyn
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- A CUDA-capable GPU is recommended but not required

## Quick Start

### Basic PID Computation

```python
from psidyn import PID, Droplet

# Initialize with a language model
model = PID(model_name="meta-llama/Llama-3.2-3B")

# Create text samples as Droplets
posts = [
    Droplet(user_id="premise1", timestamp=0, content="The sky is blue"),
    Droplet(user_id="premise2", timestamp=1, content="Blue things are calming"),
    Droplet(user_id="claim", timestamp=2, content="The sky is calming"),
]

# Compute PID
result = model.compute_pointwise_pid(
    posts,
    source_user_1="premise1",
    source_user_2="premise2",
    target_user="claim",
    target_post_idx=2,
    lag_window=10,
)

print(f"Redundancy: {result['redundancy_bits_per_token']:.3f}")
print(f"Unique (premise 1): {result['unique_x1_bits_per_token']:.3f}")
print(f"Unique (premise 2): {result['unique_x2_bits_per_token']:.3f}")
print(f"Synergy: {result['synergy_bits_per_token']:.3f}")
```

### Transfer Entropy

```python
from psidyn import TransferEntropy, Droplet

model = TransferEntropy(model_name="meta-llama/Llama-3.2-3B")

# A conversation between Alice and Bob
posts = [
    Droplet(user_id="alice", timestamp=0, content="Have you seen the news?"),
    Droplet(user_id="bob", timestamp=1, content="Yes, it's quite surprising"),
    Droplet(user_id="alice", timestamp=2, content="I thought you'd say that"),
]

# Compute TE from Alice to Bob
result = model.compute_transfer_entropy(
    posts,
    source_user="alice",
    target_user="bob",
    lag_window=5,
)
```

## Model Selection

PSIDyn works with any HuggingFace causal language model. Recommended models:

| Model | Size | Notes |
|-------|------|-------|
| `meta-llama/Llama-3.2-3B` | 3B | Good balance of speed and quality |
| `meta-llama/Llama-3.1-8B` | 8B | Higher quality, needs quantization |
| `mistralai/Mistral-7B-v0.1` | 7B | Fast inference |

### Using Quantization

```python
# 4-bit quantization (default, recommended)
model = PID(model_name="meta-llama/Llama-3.1-8B", load_in_4bit=True)

# 8-bit quantization
model = PID(model_name="meta-llama/Llama-3.1-8B", load_in_8bit=True)

# Full precision (requires more VRAM)
model = PID(model_name="meta-llama/Llama-3.2-3B", load_in_4bit=False)
```

## Next Steps

- Read about [Core Concepts](concepts.md) to understand the theory
- See [Examples](examples.md) for more detailed usage patterns
- Check the [API Reference](api.md) for complete documentation
