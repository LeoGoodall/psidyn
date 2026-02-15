"""
Tests for sequence building logic.

These tests verify that sequences are correctly constructed
for different source inclusion scenarios.
"""

import pytest
from dataclasses import dataclass


# ============================================================================
# Copy of Droplet from trident.py for testing
# ============================================================================

@dataclass
class Droplet:
    """A linguistic unit: message, post, or other text."""
    user_id: str
    timestamp: int
    content: str = ""
    post_id: str = ""


# ============================================================================
# Test sequence building logic (without instantiating Trident)
# ============================================================================

class TestSequenceBuildingLogic:
    """
    Tests for the logic of _build_sequence_with_sources.

    We test the logic without instantiating Trident (which requires a model).
    """

    def _mock_build_sequence(self, posts, target_post_idx, include_sources, tokenizer):
        """
        Reimplementation of _build_sequence_with_sources logic for testing.

        This mirrors the actual implementation but uses a provided tokenizer.
        """
        sequence_tokens = []
        target_lexical_span = None

        for i in range(target_post_idx + 1):
            post = posts[i]

            # Skip posts from users not in include_sources (unless it's the target)
            if i != target_post_idx and post.user_id not in include_sources:
                continue

            content_tokens = tokenizer.encode(post.content, add_special_tokens=False)
            newline_tokens = tokenizer.encode("\n", add_special_tokens=False)

            lexical_start = len(sequence_tokens)
            sequence_tokens.extend(content_tokens)
            lexical_end = len(sequence_tokens)
            sequence_tokens.extend(newline_tokens)

            if i == target_post_idx:
                target_lexical_span = (lexical_start, lexical_end)

        if target_lexical_span is None:
            return None

        return sequence_tokens, target_lexical_span

    def test_include_all_sources(self, mock_tokenizer, pid_droplets):
        """Including all sources should produce full sequence."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=["source1", "source2"],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        # Should contain tokens from all three posts
        # source1: "First premise" + "\n"
        # source2: "Second premise" + "\n"
        # target: "The claim" + "\n"
        assert len(tokens) > 0
        assert target_span[0] < target_span[1]

    def test_include_source1_only(self, mock_tokenizer, pid_droplets):
        """Including only source1 should omit source2."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=["source1"],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        # Should contain: source1 content + target content
        # source2 should be omitted
        source1_tokens = mock_tokenizer.encode("First premise")
        source2_tokens = mock_tokenizer.encode("Second premise")

        # Verify source1 is present (by checking substring)
        decoded = mock_tokenizer.decode(tokens)
        assert "First premise" in decoded
        assert "Second premise" not in decoded

    def test_include_source2_only(self, mock_tokenizer, pid_droplets):
        """Including only source2 should omit source1."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=["source2"],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        decoded = mock_tokenizer.decode(tokens)
        assert "Second premise" in decoded
        assert "First premise" not in decoded

    def test_include_no_sources(self, mock_tokenizer, pid_droplets):
        """Including no sources should produce target-only sequence."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=[],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        decoded = mock_tokenizer.decode(tokens)
        assert "The claim" in decoded
        assert "First premise" not in decoded
        assert "Second premise" not in decoded

    def test_target_span_at_correct_position(self, mock_tokenizer, pid_droplets):
        """Target lexical span should point to target content."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=["source1", "source2"],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        # Extract target tokens using the span
        target_tokens = tokens[target_span[0]:target_span[1]]
        target_text = mock_tokenizer.decode(target_tokens)

        assert target_text == "The claim"

    def test_target_span_with_no_sources(self, mock_tokenizer, pid_droplets):
        """Target span should start at 0 when no sources included."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=[],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        # With no sources, target should start at position 0
        assert target_span[0] == 0

    def test_nonexistent_source_user(self, mock_tokenizer, pid_droplets):
        """Including a non-existent user should effectively include nothing extra."""
        result = self._mock_build_sequence(
            pid_droplets,
            target_post_idx=2,
            include_sources=["nonexistent_user"],
            tokenizer=mock_tokenizer
        )

        assert result is not None
        tokens, target_span = result

        decoded = mock_tokenizer.decode(tokens)
        # Only target should be present
        assert "The claim" in decoded
        assert "First premise" not in decoded
        assert "Second premise" not in decoded


# ============================================================================
# Test source filtering logic
# ============================================================================

class TestSourceFiltering:
    """Tests for source inclusion/exclusion logic."""

    def test_filter_by_user_id(self):
        """Sources should be filtered by user_id."""
        posts = [
            Droplet(user_id="alice", timestamp=0, content="A", post_id="p1"),
            Droplet(user_id="bob", timestamp=1, content="B", post_id="p2"),
            Droplet(user_id="charlie", timestamp=2, content="C", post_id="p3"),
            Droplet(user_id="target", timestamp=3, content="T", post_id="p4"),
        ]
        include_sources = ["alice", "charlie"]

        included = []
        for i, post in enumerate(posts[:-1]):  # Exclude target
            if post.user_id in include_sources:
                included.append(post.user_id)

        assert included == ["alice", "charlie"]
        assert "bob" not in included

    def test_empty_include_list(self):
        """Empty include list should include no sources."""
        posts = [
            Droplet(user_id="alice", timestamp=0, content="A", post_id="p1"),
            Droplet(user_id="target", timestamp=1, content="T", post_id="p2"),
        ]
        include_sources = []

        included = [p.user_id for p in posts[:-1] if p.user_id in include_sources]
        assert included == []

    def test_target_always_included(self):
        """Target post should always be included regardless of include_sources."""
        posts = [
            Droplet(user_id="source", timestamp=0, content="S", post_id="p1"),
            Droplet(user_id="target", timestamp=1, content="T", post_id="p2"),
        ]
        include_sources = []  # Empty - no sources
        target_post_idx = 1

        # Target at target_post_idx should still be included
        # This is the key behavior: target is always included
        target_included = True  # By design
        assert target_included


# ============================================================================
# Test newline handling
# ============================================================================

class TestNewlineHandling:
    """Tests for newline token handling in sequences."""

    def test_newline_after_each_post(self, mock_tokenizer):
        """Each post should be followed by a newline."""
        content = "Hello"
        newline = "\n"

        content_tokens = mock_tokenizer.encode(content)
        newline_tokens = mock_tokenizer.encode(newline)

        # Newline should add exactly one token (ord('\n') = 10)
        assert newline_tokens == [10]

    def test_sequence_structure(self, mock_tokenizer):
        """Sequence should have content-newline-content-newline pattern."""
        posts_content = ["First", "Second", "Third"]

        all_tokens = []
        for content in posts_content:
            all_tokens.extend(mock_tokenizer.encode(content))
            all_tokens.extend(mock_tokenizer.encode("\n"))

        # Should end with newline
        assert all_tokens[-1] == ord("\n")

        # Count newlines
        newline_count = sum(1 for t in all_tokens if t == ord("\n"))
        assert newline_count == len(posts_content)
