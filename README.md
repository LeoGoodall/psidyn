<p align="center">
  <img src="images/logo.png" alt="PSIDyn Logo" width="300">
</p>

<h1 align="center">Python Semantic Information Dynamics</h1>

<p align="center">
  <a href="https://github.com/LeoGoodall/psidyn/actions/workflows/tests.yml"><img src="https://github.com/LeoGoodall/psidyn/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/psidyn/"><img src="https://img.shields.io/pypi/v/psidyn.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL-3.0"></a>
</p>

A Python package for computing information-theoretic measures over natural language using LLMs as semantic probability estimators.

## Overview

PSIDyn provides tools for measuring how information flows between text sources using LLMs to estimate semantic probabilities. It implements:

- **Transfer Entropy**: Measure directed information flow between text sequences
- **Partial Information Decomposition (PID)**: Decompose the information that two sources provide about a target into redundant, unique, and synergistic contributions
- **Co-information**: Compute the redundancy–synergy balance for any number of sources via inclusion-exclusion

## Installation

```bash
pip install psidyn
```

For GPU quantization support (recommended for large models):

```bash
pip install psidyn[quantization]
```

## Quick Start

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

## Core Concepts

### Droplet

A `Droplet` represents a unit of text with metadata:

```python
@dataclass
class Droplet:
    user_id: str      # Identifier for the source/author
    timestamp: int    # Temporal ordering
    content: str      # The text content
    post_id: str      # Optional unique identifier
```

### PID Atoms

The Partial Information Decomposition breaks down joint information I(Y; X1, X2) into:

- **Redundancy**: Information that both sources provide about the target
- **Unique X1**: Information only source 1 provides
- **Unique X2**: Information only source 2 provides
- **Synergy**: Information that emerges only when both sources are combined

### Marginalisation Methods

PSIDyn supports two methods for computing conditional probabilities for PID:

- `"omit"` (default): Physically remove source text from the sequence (provides true marginal probabilities)
- `"mask"`: Use attention masking to block information flow

### Redundancy Functionals

Two redundancy measures are available:

- `"mmi"` (default): Minimum Mutual Information - R = min(I(Y;X1), I(Y;X2))
- `"ccs"`: Common Change in Surprisal - based on co-information sign matching

## API Reference

### TransferEntropy

```python
class TransferEntropy(Trident):
    def compute_transfer_entropy(
        self,
        posts: List[Droplet],
        source_user: str,
        target_user: str,
        lag_window: int,
    ) -> Dict[str, Any]:
        """Compute transfer entropy from source to target."""
```

### PID

```python
class PID(Trident):
    def compute_pointwise_pid(
        self,
        posts: List[Droplet],
        source_user_1: str,
        source_user_2: str,
        target_user: str,
        target_post_idx: int,
        lag_window: int,
        redundancy: Literal["mmi", "ccs"] = "mmi",
        method: Literal["mask", "omit"] = "omit",
    ) -> Dict[str, float]:
        """Compute pointwise PID for a single target."""
```

### CoInfo

```python
class CoInfo(Trident):
    def compute_pointwise_coinfo(
        self,
        posts: List[Droplet],
        source_users: List[str],   # any number of sources
        target_user: str,
        target_post_idx: int,
    ) -> Dict[str, float]:
        """Compute co-information for a single target with n sources."""

    def compute_coinfo(
        self,
        posts: List[Droplet],
        source_users: List[str],
        target_user: str,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """Token-weighted co-information over all target posts."""
```

Co-information (interaction information) is the inclusion-exclusion alternating sum over all non-empty subsets of sources:

$$CI(X_1; \ldots; X_n; Y) = \sum_{\emptyset \neq S \subseteq \{1,\ldots,n\}} (-1)^{|S|+1}\; I(Y; X_S)$$

- **Positive** co-information → redundancy-dominated (sources overlap)
- **Negative** co-information → synergy-dominated (sources are complementary)

This requires $2^n$ forward passes (all subsets + baseline), so it scales well for moderate $n$ (e.g. $n \leq 5$).

```python
from psidyn import CoInfo, Droplet

model = CoInfo(model_name="meta-llama/Llama-3.2-3B")

posts = [
    Droplet(user_id="p1", timestamp=0, content="First premise"),
    Droplet(user_id="p2", timestamp=1, content="Second premise"),
    Droplet(user_id="p3", timestamp=2, content="Third premise"),
    Droplet(user_id="claim", timestamp=3, content="The claim"),
]

result = model.compute_pointwise_coinfo(
    posts, source_users=["p1", "p2", "p3"], target_user="claim", target_post_idx=3
)
print(f"Co-information: {result['coinfo_bits_per_token']:.3f}")
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0

For quantization:
- bitsandbytes >= 0.41.0
- accelerate >= 0.20.0

## Citation

If you use PSIDyn in your research, please cite:

```bibtex
@software{psidyn,
  author = {Goodall, Leonardo; Luppi, Andrea; Mediano, Pedro},
  title = {PSIDyn: Python Semantic Information Dynamics},
  year = {2026},
  url = {https://github.com/LeoGoodall/psidyn}
}
```

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- You can use, modify, and distribute this software
- Any modifications must also be open source under AGPL-3.0
- If you run a modified version as a network service, you must provide the source code to users

See [LICENSE](LICENSE) for the full text.

For commercial licensing inquiries, contact the author.
