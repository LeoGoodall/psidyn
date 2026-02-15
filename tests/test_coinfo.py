"""
Tests for co-information (interaction information) mathematical functions.

These tests verify the inclusion-exclusion formula and algebraic properties
without requiring GPU or model inference.
"""

import pytest
from itertools import combinations


# ============================================================================
# Pure implementation of co-information for testing
# (Avoids import issues with torch/transformers dependencies)
# ============================================================================

def coinformation(mi_values: dict) -> float:
    """Compute co-information from a dict mapping frozensets of source indices to MI values.

    Args:
        mi_values: Maps frozenset({i, j, ...}) -> I(Y; X_i, X_j, ...) for every
                   non-empty subset of source indices {0, 1, ..., n-1}.

    Returns:
        Co-information (inclusion-exclusion alternating sum).
    """
    coinfo = 0.0
    for subset, mi in mi_values.items():
        k = len(subset)
        sign = (-1) ** (k + 1)
        coinfo += sign * mi
    return coinfo


def build_all_subset_mis(individual_mis: list, joint_mi: float, pairwise_mis: dict = None,
                          higher_order: dict = None) -> dict:
    """Helper to build the mi_values dict for testing.

    Args:
        individual_mis: List of I(Y; X_i) for i = 0, ..., n-1.
        joint_mi: I(Y; X_1, ..., X_n).
        pairwise_mis: Optional dict mapping (i, j) tuples to I(Y; X_i, X_j).
        higher_order: Optional dict mapping tuples to MI for subsets of size 3+.
    """
    n = len(individual_mis)
    result = {}

    # Singletons
    for i in range(n):
        result[frozenset({i})] = individual_mis[i]

    # Pairs
    if pairwise_mis:
        for (i, j), mi in pairwise_mis.items():
            result[frozenset({i, j})] = mi

    # Higher-order subsets
    if higher_order:
        for indices, mi in higher_order.items():
            result[frozenset(indices)] = mi

    # Full joint
    result[frozenset(range(n))] = joint_mi

    return result


# ============================================================================
# Test co-information for n=1
# ============================================================================

class TestCoInfoN1:
    """Co-information with a single source is just I(Y; X1)."""

    def test_single_source(self):
        mi_values = {frozenset({0}): 2.5}
        assert coinformation(mi_values) == pytest.approx(2.5)

    def test_single_source_zero(self):
        mi_values = {frozenset({0}): 0.0}
        assert coinformation(mi_values) == pytest.approx(0.0)

    def test_single_source_negative(self):
        mi_values = {frozenset({0}): -1.0}
        assert coinformation(mi_values) == pytest.approx(-1.0)


# ============================================================================
# Test co-information for n=2
# ============================================================================

class TestCoInfoN2:
    """Co-information for two sources: I(Y;X1) + I(Y;X2) - I(Y;X1,X2)."""

    def test_basic_redundancy(self):
        """Overlapping sources -> positive co-information."""
        # Both sources provide 3 bits, joint is 4 -> 3 + 3 - 4 = 2 (redundancy-dominated)
        mi_values = build_all_subset_mis([3.0, 3.0], joint_mi=4.0)
        assert coinformation(mi_values) == pytest.approx(2.0)

    def test_basic_synergy(self):
        """Synergistic sources -> negative co-information."""
        # Each source provides 1 bit alone, but 5 together -> 1 + 1 - 5 = -3 (synergy-dominated)
        mi_values = build_all_subset_mis([1.0, 1.0], joint_mi=5.0)
        assert coinformation(mi_values) == pytest.approx(-3.0)

    def test_independent_sources(self):
        """Independent sources: I(Y;X1,X2) = I(Y;X1) + I(Y;X2) -> co-info = 0."""
        mi_values = build_all_subset_mis([2.0, 3.0], joint_mi=5.0)
        assert coinformation(mi_values) == pytest.approx(0.0)

    def test_matches_pid_coinfo(self):
        """For n=2, co-info = i1 + i2 - i12 (same as CCS's 'c' variable in pid.py)."""
        i1, i2, i12 = 2.0, 3.0, 4.0
        mi_values = build_all_subset_mis([i1, i2], joint_mi=i12)
        expected = i1 + i2 - i12  # = 1.0
        assert coinformation(mi_values) == pytest.approx(expected)


# ============================================================================
# Test co-information for n=3
# ============================================================================

class TestCoInfoN3:
    """Co-information for three sources (interaction information)."""

    def test_formula(self):
        """Verify the n=3 inclusion-exclusion formula."""
        i1, i2, i3 = 2.0, 3.0, 1.5
        i12, i13, i23 = 4.0, 3.0, 3.5
        i123 = 5.0

        mi_values = build_all_subset_mis(
            [i1, i2, i3],
            joint_mi=i123,
            pairwise_mis={(0, 1): i12, (0, 2): i13, (1, 2): i23},
        )

        # CI = (i1 + i2 + i3) - (i12 + i13 + i23) + i123
        expected = (i1 + i2 + i3) - (i12 + i13 + i23) + i123
        assert coinformation(mi_values) == pytest.approx(expected)

    def test_all_redundant(self):
        """Three identical sources: high positive co-info."""
        # Each provides 5 bits, each pair provides 5, triple provides 5
        mi_values = build_all_subset_mis(
            [5.0, 5.0, 5.0],
            joint_mi=5.0,
            pairwise_mis={(0, 1): 5.0, (0, 2): 5.0, (1, 2): 5.0},
        )
        # CI = (5+5+5) - (5+5+5) + 5 = 5
        assert coinformation(mi_values) == pytest.approx(5.0)

    def test_purely_synergistic(self):
        """Sources with no individual or pairwise info, only joint."""
        mi_values = build_all_subset_mis(
            [0.0, 0.0, 0.0],
            joint_mi=3.0,
            pairwise_mis={(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0},
        )
        # CI = 0 - 0 + 3 = 3 (positive! triple synergy shows up as positive
        # in the n=3 co-information because of the alternating sign)
        assert coinformation(mi_values) == pytest.approx(3.0)

    def test_independent_sources(self):
        """Fully independent sources: all joint MIs are additive -> co-info = 0."""
        i1, i2, i3 = 1.0, 2.0, 3.0
        mi_values = build_all_subset_mis(
            [i1, i2, i3],
            joint_mi=i1 + i2 + i3,
            pairwise_mis={(0, 1): i1 + i2, (0, 2): i1 + i3, (1, 2): i2 + i3},
        )
        # CI = (1+2+3) - (3+4+5) + 6 = 6 - 12 + 6 = 0
        assert coinformation(mi_values) == pytest.approx(0.0)


# ============================================================================
# Test co-information for n=4
# ============================================================================

class TestCoInfoN4:
    """Co-information for four sources."""

    def test_formula(self):
        """Verify the n=4 inclusion-exclusion formula with known values."""
        individual = [1.0, 1.0, 1.0, 1.0]
        joint = 4.0
        pairwise = {(0, 1): 2.0, (0, 2): 2.0, (0, 3): 2.0,
                     (1, 2): 2.0, (1, 3): 2.0, (2, 3): 2.0}
        triples = {(0, 1, 2): 3.0, (0, 1, 3): 3.0, (0, 2, 3): 3.0, (1, 2, 3): 3.0}

        mi_values = build_all_subset_mis(
            individual, joint_mi=joint, pairwise_mis=pairwise, higher_order=triples
        )

        # CI = sum_singles - sum_pairs + sum_triples - joint
        # = 4 - 12 + 12 - 4 = 0
        assert coinformation(mi_values) == pytest.approx(0.0)

    def test_sign_alternation(self):
        """Verify signs: +singletons, -pairs, +triples, -quadruple."""
        # All MI = 1.0 for simplicity
        n = 4
        mi_values = {}
        for k in range(1, n + 1):
            for combo in combinations(range(n), k):
                mi_values[frozenset(combo)] = 1.0

        # Counts: C(4,1)=4, C(4,2)=6, C(4,3)=4, C(4,4)=1
        # CI = 4*(+1) + 6*(-1) + 4*(+1) + 1*(-1) = 4 - 6 + 4 - 1 = 1
        assert coinformation(mi_values) == pytest.approx(1.0)


# ============================================================================
# Test algebraic properties
# ============================================================================

class TestCoInfoProperties:
    """General properties of co-information."""

    def test_subset_count(self):
        """Number of non-empty subsets of n elements is 2^n - 1."""
        for n in range(1, 6):
            count = sum(1 for k in range(1, n + 1) for _ in combinations(range(n), k))
            assert count == 2**n - 1

    def test_n2_coinfo_equals_redundancy_minus_synergy(self):
        """For n=2 with MMI: co-info = redundancy - synergy."""
        i1, i2, i12 = 3.0, 5.0, 7.0

        # MMI PID
        red = min(i1, i2)  # 3.0
        unq1 = i1 - red     # 0.0
        unq2 = i2 - red     # 2.0
        syn = i12 - unq1 - unq2 - red  # 7 - 0 - 2 - 3 = 2.0

        # Co-info
        coinfo = i1 + i2 - i12  # 3 + 5 - 7 = 1.0

        # Should equal red - syn
        assert coinfo == pytest.approx(red - syn)

    def test_symmetry(self):
        """Co-information is symmetric in sources (permutation-invariant)."""
        from itertools import permutations

        i_vals = [2.0, 3.0, 1.5]
        pair_vals = {(0, 1): 4.0, (0, 2): 3.0, (1, 2): 3.5}
        joint = 5.0

        mi_values = build_all_subset_mis(
            i_vals, joint_mi=joint, pairwise_mis=pair_vals
        )
        original = coinformation(mi_values)

        # Permute source indices and verify same result
        for perm in permutations(range(3)):
            perm_i_vals = [i_vals[p] for p in perm]
            perm_pair_vals = {}
            for (a, b), v in pair_vals.items():
                new_a, new_b = sorted([perm.index(a), perm.index(b)])
                perm_pair_vals[(new_a, new_b)] = v

            perm_mi = build_all_subset_mis(
                perm_i_vals, joint_mi=joint, pairwise_mis=perm_pair_vals
            )
            assert coinformation(perm_mi) == pytest.approx(original)
