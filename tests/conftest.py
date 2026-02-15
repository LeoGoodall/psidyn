"""
Shared fixtures for SIDyn tests.

These fixtures provide mock objects and sample data that don't require GPU.
"""

import pytest
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock


# ============================================================================
# Droplet dataclass (copied from trident.py to avoid torch dependency)
# ============================================================================

@dataclass
class Droplet:
    """A linguistic unit: message, post, or other text."""
    user_id: str
    timestamp: int
    content: str = ""
    post_id: str = ""


# ============================================================================
# Mock Tokenizer
# ============================================================================

class MockTokenizer:
    """
    A mock tokenizer that returns predictable token IDs.

    Encoding scheme: each character becomes its ord() value.
    This makes tests deterministic and easy to verify.
    """

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs (one token per character for simplicity)."""
        return [ord(c) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return "".join(chr(t) for t in token_ids if t < 128)


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    return MockTokenizer()


# ============================================================================
# Sample Droplets
# ============================================================================

@pytest.fixture
def sample_droplets():
    """Sample conversation for testing."""
    return [
        Droplet(user_id="alice", timestamp=0, content="Hello there", post_id="p1"),
        Droplet(user_id="bob", timestamp=1, content="Hi Alice", post_id="p2"),
        Droplet(user_id="alice", timestamp=2, content="How are you", post_id="p3"),
    ]


@pytest.fixture
def pid_droplets():
    """Sample triplet for PID testing (two sources, one target)."""
    return [
        Droplet(user_id="source1", timestamp=0, content="First premise", post_id="s1"),
        Droplet(user_id="source2", timestamp=1, content="Second premise", post_id="s2"),
        Droplet(user_id="target", timestamp=2, content="The claim", post_id="t1"),
    ]


@pytest.fixture
def empty_droplets():
    """Empty list for edge case testing."""
    return []


@pytest.fixture
def single_droplet():
    """Single droplet for edge case testing."""
    return [
        Droplet(user_id="alice", timestamp=0, content="Only message", post_id="p1"),
    ]


# ============================================================================
# Mock Model Configuration
# ============================================================================

@pytest.fixture
def mock_model_config():
    """Mock model config with max_position_embeddings."""
    config = MagicMock()
    config.max_position_embeddings = 4096
    return config
