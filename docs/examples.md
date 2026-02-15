# Examples

## Analyzing Arguments

Compute PID for premise-claim structures:

```python
from psidyn import PID, Droplet

model = PID(model_name="meta-llama/Llama-3.2-3B")

# An argument with two premises
posts = [
    Droplet(user_id="p1", timestamp=0,
            content="Regular exercise improves cardiovascular health"),
    Droplet(user_id="p2", timestamp=1,
            content="Cardiovascular health reduces disease risk"),
    Droplet(user_id="claim", timestamp=2,
            content="Regular exercise reduces disease risk"),
]

result = model.compute_pointwise_pid(
    posts,
    source_user_1="p1",
    source_user_2="p2",
    target_user="claim",
    target_post_idx=2,
    lag_window=10,
)

# High synergy indicates the premises work together
print(f"Synergy: {result['synergy_bits_per_token']:.3f}")
```

## Conversation Analysis

Measure information flow in dialogue:

```python
from psidyn import TransferEntropy, Droplet

model = TransferEntropy(model_name="meta-llama/Llama-3.2-3B")

conversation = [
    Droplet(user_id="therapist", timestamp=0,
            content="How have you been feeling this week?"),
    Droplet(user_id="client", timestamp=1,
            content="I've been struggling with anxiety"),
    Droplet(user_id="therapist", timestamp=2,
            content="Can you tell me more about what triggers it?"),
    Droplet(user_id="client", timestamp=3,
            content="Work deadlines seem to make it worse"),
    Droplet(user_id="therapist", timestamp=4,
            content="That's a common pattern with workplace stress"),
]

# TE from therapist to client
te_t2c = model.compute_transfer_entropy(
    conversation,
    source_user="therapist",
    target_user="client",
    lag_window=5,
)

# TE from client to therapist
te_c2t = model.compute_transfer_entropy(
    conversation,
    source_user="client",
    target_user="therapist",
    lag_window=5,
)

print(f"TE (therapist → client): {te_t2c}")
print(f"TE (client → therapist): {te_c2t}")
```

## Batch Processing

Process multiple samples efficiently:

```python
from psidyn import PID, Droplet
import pandas as pd
from tqdm import tqdm

model = PID(model_name="meta-llama/Llama-3.2-3B")

# Load your data
data = pd.read_csv("arguments.csv")

results = []
for idx, row in tqdm(data.iterrows(), total=len(data)):
    posts = [
        Droplet(user_id="p1", timestamp=0, content=row["premise1"]),
        Droplet(user_id="p2", timestamp=1, content=row["premise2"]),
        Droplet(user_id="claim", timestamp=2, content=row["claim"]),
    ]

    result = model.compute_pointwise_pid(
        posts,
        source_user_1="p1",
        source_user_2="p2",
        target_user="claim",
        target_post_idx=2,
        lag_window=10,
    )

    result["sample_id"] = idx
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("pid_results.csv", index=False)
```

## Comparing Redundancy Functionals

Compare MMI and CCS decompositions:

```python
from psidyn import PID, Droplet

model = PID(model_name="meta-llama/Llama-3.2-3B")

posts = [
    Droplet(user_id="p1", timestamp=0, content="The cat sat on the mat"),
    Droplet(user_id="p2", timestamp=1, content="The mat was red"),
    Droplet(user_id="target", timestamp=2, content="The cat sat on something red"),
]

# MMI decomposition
mmi_result = model.compute_pointwise_pid(
    posts, "p1", "p2", "target", 2, 10,
    redundancy="mmi"
)

# CCS decomposition
ccs_result = model.compute_pointwise_pid(
    posts, "p1", "p2", "target", 2, 10,
    redundancy="ccs"
)

print("MMI Decomposition:")
print(f"  Redundancy: {mmi_result['redundancy_bits_per_token']:.3f}")
print(f"  Synergy: {mmi_result['synergy_bits_per_token']:.3f}")

print("\nCCS Decomposition:")
print(f"  Redundancy: {ccs_result['redundancy_bits_per_token']:.3f}")
print(f"  Synergy: {ccs_result['synergy_bits_per_token']:.3f}")
```

## Using Different Models

```python
from psidyn import PID, Droplet

# Smaller, faster model
model_small = PID(
    model_name="meta-llama/Llama-3.2-1B",
    load_in_4bit=False,  # Small enough for full precision
)

# Larger, more capable model with quantization
model_large = PID(
    model_name="meta-llama/Llama-3.1-8B",
    load_in_4bit=True,
)

# CPU-only (slower but works without GPU)
model_cpu = PID(
    model_name="meta-llama/Llama-3.2-1B",
    device="cpu",
    load_in_4bit=False,
)
```
