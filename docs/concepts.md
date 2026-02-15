# Core Concepts

## Semantic Information Dynamics

PSIDyn measures information flow between text sequences by leveraging LLMs as semantic probability estimators. Unlike traditional information theory that operates on discrete symbols, semantic information dynamics captures meaning-level relationships.

## The Droplet

A `Droplet` is the fundamental unit of text in PSIDyn:

```python
@dataclass
class Droplet:
    user_id: str      # Identifier for the source/author
    timestamp: int    # Temporal ordering
    content: str      # The text content
    post_id: str      # Optional unique identifier
```

The `user_id` field identifies different sources of text (speakers, authors, etc.). The `timestamp` field determines temporal ordering for causal analysis.

## Transfer Entropy

Transfer Entropy (TE) measures the directed information flow from a source to a target:

$$TE_{X \to Y} = I(Y_t; X_{<t} | Y_{<t})$$

This quantifies how much knowing the source's past reduces uncertainty about the target, beyond what the target's own past tells us.

### Interpretation

- **TE > 0**: The source provides information about the target
- **TE ≈ 0**: The source and target are informationally independent
- Higher TE indicates stronger directed influence

## Partial Information Decomposition (PID)

PID decomposes the joint information that two sources provide about a target into four non-negative atoms:

$$I(Y; X_1, X_2) = \text{Red} + \text{Unq}_1 + \text{Unq}_2 + \text{Syn}$$

### PID Atoms

| Atom | Meaning |
|------|---------|
| **Redundancy** | Information both sources provide (overlap) |
| **Unique X1** | Information only source 1 provides |
| **Unique X2** | Information only source 2 provides |
| **Synergy** | Information that emerges only from combination |

### Example: Argument Analysis

Consider two premises supporting a claim:

- **Premise 1**: "The sky is blue"
- **Premise 2**: "Blue things are calming"
- **Claim**: "The sky is calming"

The PID reveals:
- **Synergy**: The logical connection requires both premises
- **Redundancy**: Any shared information (e.g., "blue")
- **Unique**: Information specific to each premise

## Redundancy Functionals

PSIDyn implements two redundancy measures:

### MMI (Minimum Mutual Information)

$$R_{MMI} = \min(I(Y; X_1), I(Y; X_2))$$

The minimum of the individual mutual informations. Simple and interpretable.

### CCS (Common Change in Surprisal)

Based on pointwise co-information with sign-matching constraints. More theoretically grounded for pointwise analysis.

## Marginalization Methods

Computing marginal probabilities (P(Y|X1) without X2) requires removing one source:

### Omit Method (Default)

Physically remove the source text from the sequence. Provides true marginal probabilities.

```
Full:  [X1] [X2] [Y]
Marginal: [X1] [Y]  (X2 omitted)
```

### Mask Method

Use attention masking to block information flow. Preserves positional structure but doesn't give true marginals.

**Recommendation**: Use `"omit"` for PID (true marginals matter). Use `"mask"` for TE (temporal structure matters).

## Pointwise vs Expected Measures

PSIDyn computes **pointwise** information measures for individual samples:

$$i(y; x) = \log \frac{p(y|x)}{p(y)}$$

The expected value E[i(Y;X)] equals the mutual information I(Y;X), but pointwise values can be negative (when conditioning increases surprisal).

### Why Pointwise?

1. **Sample-level analysis**: Understand individual cases, not just averages
2. **Heterogeneity**: Detect when effects vary across samples
3. **Statistical testing**: Each sample provides an observation for inference
