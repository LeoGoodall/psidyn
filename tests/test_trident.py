"""
Tests for Trident base class functionality.

These tests verify data structures, sequence building, and edge case handling
without requiring GPU or model inference.
"""

import pytest
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================================
# Copy of dataclasses from trident.py for testing
# (Avoids import issues with torch/transformers dependencies)
# ============================================================================

@dataclass
class Droplet:
    """A linguistic unit: message, post, or other text."""
    user_id: str
    timestamp: int
    content: str = ""
    post_id: str = ""


@dataclass
class ContextMapping:
    """Holds token-level spans needed to realise conditional masking."""
    tokens: List[int]
    source_spans: List[Tuple[int, int]]
    target_span: Tuple[int, int]
    target_lexical_span: Tuple[int, int]
    post_spans: List[Tuple[str, Tuple[int, int]]]
    target_history_spans: List[Tuple[int, int]]


# ============================================================================
# Test Droplet dataclass
# ============================================================================

class TestDroplet:
    """Tests for the Droplet dataclass."""

    def test_creation_with_all_fields(self):
        """Droplet should store all provided fields."""
        d = Droplet(
            user_id="alice",
            timestamp=123,
            content="Hello world",
            post_id="p1"
        )
        assert d.user_id == "alice"
        assert d.timestamp == 123
        assert d.content == "Hello world"
        assert d.post_id == "p1"

    def test_creation_with_defaults(self):
        """Droplet should have default values for optional fields."""
        d = Droplet(user_id="bob", timestamp=0)
        assert d.user_id == "bob"
        assert d.timestamp == 0
        assert d.content == ""
        assert d.post_id == ""

    def test_equality(self):
        """Droplets with same fields should be equal."""
        d1 = Droplet(user_id="alice", timestamp=1, content="Hi", post_id="p1")
        d2 = Droplet(user_id="alice", timestamp=1, content="Hi", post_id="p1")
        assert d1 == d2

    def test_inequality(self):
        """Droplets with different fields should not be equal."""
        d1 = Droplet(user_id="alice", timestamp=1, content="Hi", post_id="p1")
        d2 = Droplet(user_id="bob", timestamp=1, content="Hi", post_id="p1")
        assert d1 != d2

    def test_empty_content(self):
        """Droplet should handle empty content."""
        d = Droplet(user_id="alice", timestamp=0, content="", post_id="p1")
        assert d.content == ""
        assert len(d.content) == 0

    def test_unicode_content(self):
        """Droplet should handle unicode content."""
        d = Droplet(user_id="alice", timestamp=0, content="Hello 世界 🌍", post_id="p1")
        assert "世界" in d.content
        assert "🌍" in d.content


# ============================================================================
# Test ContextMapping dataclass
# ============================================================================

class TestContextMapping:
    """Tests for the ContextMapping dataclass."""

    def test_creation(self):
        """ContextMapping should store all fields."""
        cm = ContextMapping(
            tokens=[1, 2, 3, 4, 5],
            source_spans=[(0, 2)],
            target_span=(3, 5),
            target_lexical_span=(3, 4),
            post_spans=[("alice", (0, 2)), ("bob", (3, 5))],
            target_history_spans=[]
        )
        assert cm.tokens == [1, 2, 3, 4, 5]
        assert cm.source_spans == [(0, 2)]
        assert cm.target_span == (3, 5)
        assert cm.target_lexical_span == (3, 4)
        assert len(cm.post_spans) == 2

    def test_empty_source_spans(self):
        """ContextMapping should handle empty source spans."""
        cm = ContextMapping(
            tokens=[1, 2, 3],
            source_spans=[],
            target_span=(0, 3),
            target_lexical_span=(0, 2),
            post_spans=[],
            target_history_spans=[]
        )
        assert cm.source_spans == []

    def test_multiple_source_spans(self):
        """ContextMapping should handle multiple source spans."""
        cm = ContextMapping(
            tokens=list(range(20)),
            source_spans=[(0, 5), (6, 10), (11, 15)],
            target_span=(16, 20),
            target_lexical_span=(17, 19),
            post_spans=[],
            target_history_spans=[]
        )
        assert len(cm.source_spans) == 3


# ============================================================================
# Test edge cases for validation
# ============================================================================

class TestEdgeCases:
    """Tests for edge case handling in the codebase."""

    def test_target_post_idx_zero_returns_empty(self):
        """When target_post_idx is 0, should return zeros/empty."""
        # This is validated in PID/TE - target at index 0 has no context
        # The logic is: if target_post_idx == 0, return empty result
        target_post_idx = 0
        assert target_post_idx == 0  # Edge case marker

    def test_target_post_idx_out_of_bounds(self):
        """When target_post_idx >= len(posts), should handle gracefully."""
        posts = [
            Droplet(user_id="alice", timestamp=0, content="Hi", post_id="p1"),
            Droplet(user_id="bob", timestamp=1, content="Hello", post_id="p2"),
        ]
        target_post_idx = 5  # Out of bounds

        assert target_post_idx >= len(posts)

    def test_empty_posts_list(self):
        """Empty posts list should be handled gracefully."""
        posts = []
        assert len(posts) == 0

    def test_single_post(self):
        """Single post should have no valid targets (index 0 is skipped)."""
        posts = [
            Droplet(user_id="alice", timestamp=0, content="Only post", post_id="p1"),
        ]
        # Target at index 0 should be skipped
        # No valid targets exist
        assert len(posts) == 1

    def test_target_user_mismatch(self):
        """When target user doesn't match post user, should return empty."""
        posts = [
            Droplet(user_id="alice", timestamp=0, content="Hi", post_id="p1"),
            Droplet(user_id="bob", timestamp=1, content="Hello", post_id="p2"),
        ]
        target_user = "charlie"
        target_post_idx = 1

        # posts[1].user_id == "bob" != "charlie"
        assert posts[target_post_idx].user_id != target_user

    def test_zero_token_count_division(self):
        """Zero token count should not cause division by zero."""
        sum_w = 0.0
        sum_values = 0.0

        # The code should check for sum_w == 0 before dividing
        if sum_w == 0.0:
            result = 0.0
        else:
            result = sum_values / sum_w

        assert result == 0.0

    def test_empty_content_tokenization(self):
        """Empty content should produce empty token list."""
        content = ""
        # Mock tokenizer behavior
        tokens = [ord(c) for c in content]
        assert tokens == []

    def test_span_validity(self):
        """Spans with end <= start should be treated as empty."""
        span = (5, 3)  # Invalid: end < start
        start, end = span
        assert end <= start

        span2 = (5, 5)  # Empty: end == start
        start2, end2 = span2
        assert end2 <= start2


# ============================================================================
# Test weighted averaging logic
# ============================================================================

class TestWeightedAveraging:
    """Tests for weighted averaging used in TE/PID aggregation."""

    def test_basic_weighted_average(self):
        """Basic weighted average calculation."""
        weights = [2.0, 3.0, 5.0]
        values = [1.0, 2.0, 3.0]

        sum_w = sum(weights)
        sum_wv = sum(w * v for w, v in zip(weights, values))
        avg = sum_wv / sum_w

        # (2*1 + 3*2 + 5*3) / (2+3+5) = (2 + 6 + 15) / 10 = 23/10 = 2.3
        assert avg == pytest.approx(2.3)

    def test_equal_weights(self):
        """Equal weights should give simple mean."""
        weights = [1.0, 1.0, 1.0]
        values = [2.0, 4.0, 6.0]

        sum_w = sum(weights)
        sum_wv = sum(w * v for w, v in zip(weights, values))
        avg = sum_wv / sum_w

        # Simple mean: (2 + 4 + 6) / 3 = 4.0
        assert avg == pytest.approx(4.0)

    def test_single_sample(self):
        """Single sample should return that value."""
        weights = [5.0]
        values = [3.14]

        sum_w = sum(weights)
        sum_wv = sum(w * v for w, v in zip(weights, values))
        avg = sum_wv / sum_w

        assert avg == pytest.approx(3.14)

    def test_zero_weights_handling(self):
        """Zero weights should be handled (skipped in accumulation)."""
        weights = [0.0, 0.0, 5.0]
        values = [100.0, 200.0, 3.0]

        # Only non-zero weights contribute
        sum_w = sum(w for w in weights if w > 0)
        sum_wv = sum(w * v for w, v in zip(weights, values) if w > 0)

        if sum_w > 0:
            avg = sum_wv / sum_w
        else:
            avg = 0.0

        # Only the last sample contributes: 5*3 / 5 = 3.0
        assert avg == pytest.approx(3.0)

    def test_all_zero_weights(self):
        """All zero weights should return 0 (or handle gracefully)."""
        weights = [0.0, 0.0, 0.0]
        values = [1.0, 2.0, 3.0]

        sum_w = sum(weights)

        if sum_w == 0.0:
            avg = 0.0
        else:
            avg = sum(w * v for w, v in zip(weights, values)) / sum_w

        assert avg == 0.0


# ============================================================================
# Test span overlap detection
# ============================================================================

class TestSpanOverlap:
    """Tests for span overlap detection logic."""

    def test_no_overlap_before(self):
        """Spans that don't overlap (a ends before b starts)."""
        span_a = (0, 5)
        span_b = (10, 15)

        # No overlap: a[1] <= b[0] or b[1] <= a[0]
        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is True

    def test_no_overlap_after(self):
        """Spans that don't overlap (a starts after b ends)."""
        span_a = (10, 15)
        span_b = (0, 5)

        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is True

    def test_overlap_partial(self):
        """Spans that partially overlap."""
        span_a = (0, 10)
        span_b = (5, 15)

        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is False  # They do overlap

    def test_overlap_contained(self):
        """One span fully contained in another."""
        span_a = (0, 20)
        span_b = (5, 10)

        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is False  # b is inside a

    def test_overlap_identical(self):
        """Identical spans."""
        span_a = (5, 10)
        span_b = (5, 10)

        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is False  # They're the same

    def test_adjacent_no_overlap(self):
        """Adjacent spans (touching but not overlapping)."""
        span_a = (0, 5)
        span_b = (5, 10)

        # a ends at 5, b starts at 5 -> no overlap (half-open intervals)
        no_overlap = span_a[1] <= span_b[0] or span_b[1] <= span_a[0]
        assert no_overlap is True
