# API Reference

## Data Structures

### Droplet

```python
@dataclass
class Droplet:
    """A linguistic unit: message, post, or other text."""
    user_id: str      # Identifier for the source/author
    timestamp: int    # Temporal ordering
    content: str = "" # The text content
    post_id: str = "" # Optional unique identifier
```

### ContextMapping

```python
@dataclass
class ContextMapping:
    """Holds token-level spans needed for conditional masking."""
    tokens: List[int]
    source_spans: List[Tuple[int, int]]
    target_span: Tuple[int, int]
    target_lexical_span: Tuple[int, int]
    post_spans: List[Tuple[str, Tuple[int, int]]]
    target_history_spans: List[Tuple[int, int]]
```

## Classes

### Trident

Base class for semantic information dynamics. Handles model loading and core computations.

```python
class Trident:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the Trident model.

        Args:
            model_name: HuggingFace model identifier
            device: Device for computation ("cuda" or "cpu")
            load_in_4bit: Use 4-bit quantization (recommended)
            load_in_8bit: Use 8-bit quantization
        """
```

### PID

Partial Information Decomposition with two sources.

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
        eps: float = 1e-12,
        clamp_nonneg: bool = True,
    ) -> Dict[str, float]:
        """
        Compute pointwise PID for a single target post.

        Args:
            posts: List of Droplet posts in temporal order
            source_user_1: User ID for first source
            source_user_2: User ID for second source
            target_user: User ID for target
            target_post_idx: Index of the target post in posts list
            lag_window: Maximum time difference for including sources
            redundancy: Redundancy functional ("mmi" or "ccs")
            method: Marginalization method ("omit" or "mask")
            eps: Epsilon for numerical stability
            clamp_nonneg: Clamp negative values to zero

        Returns:
            Dictionary with keys:
            - i_y_x1_bits_per_token: I(Y; X1)
            - i_y_x2_bits_per_token: I(Y; X2)
            - i_y_x1x2_bits_per_token: I(Y; X1, X2)
            - redundancy_bits_per_token: Redundancy
            - unique_x1_bits_per_token: Unique information from X1
            - unique_x2_bits_per_token: Unique information from X2
            - synergy_bits_per_token: Synergistic information
            - scored_token_count: Number of tokens scored
        """

    def redundancy_mmi(
        self,
        i1: float,
        i2: float,
        clamp_nonneg: bool = True
    ) -> float:
        """MMI redundancy: min(i1, i2)."""

    def redundancy_ccs_pointwise(
        self,
        i1: float,
        i2: float,
        i12: float,
        eps: float = 1e-12,
        clamp_nonneg: bool = True,
    ) -> float:
        """CCS redundancy based on co-information sign matching."""
```

### TransferEntropy

Transfer entropy between text sources.

```python
class TransferEntropy(Trident):
    def compute_transfer_entropy(
        self,
        posts: List[Droplet],
        source_user: str,
        target_user: str,
        lag_window: int,
    ) -> Dict[str, Any]:
        """
        Compute transfer entropy from source to target user.

        Args:
            posts: List of Droplet posts
            source_user: User ID for source
            target_user: User ID for target
            lag_window: Time window for source posts

        Returns:
            Dictionary with TE values and metadata
        """
```

## Type Aliases

```python
RedundancyMethod = Literal["mmi", "ccs"]
MarginalizationMethod = Literal["mask", "omit"]
```
