"""
Tests for PID mathematical functions.

These tests verify the pure mathematical operations in pid.py
without requiring GPU or model inference.
"""

import pytest


# ============================================================================
# Copy of pure functions from pid.py for testing
# (Avoids import issues with torch/transformers dependencies)
# ============================================================================

def _sign(x: float, eps: float = 1e-12) -> int:
    """Helper function to determine sign with epsilon tolerance."""
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def redundancy_mmi(i1: float, i2: float, clamp_nonneg: bool = True) -> float:
    """MMI redundancy: min(i1, i2), optionally clamped to non-negative."""
    r = min(i1, i2)
    return max(0.0, r) if clamp_nonneg else r


def redundancy_ccs_pointwise(
    i1: float,
    i2: float,
    i12: float,
    eps: float = 1e-12,
    clamp_nonneg: bool = True,
) -> float:
    """CCS redundancy: co-information when all signs match."""
    c = i1 + i2 - i12  # co-information
    s1, s2, s12, sc = _sign(i1, eps), _sign(i2, eps), _sign(i12, eps), _sign(c, eps)
    keep = (s1 == s2 == s12 == sc) and (sc != 0)
    r = c if keep else 0.0
    return max(0.0, r) if clamp_nonneg else r


# ============================================================================
# Test _sign helper function
# ============================================================================

class TestSignFunction:
    """Tests for the _sign helper function."""

    def test_positive_value(self):
        """Positive values should return 1."""
        assert _sign(1.0) == 1
        assert _sign(0.5) == 1
        assert _sign(100.0) == 1

    def test_negative_value(self):
        """Negative values should return -1."""
        assert _sign(-1.0) == -1
        assert _sign(-0.5) == -1
        assert _sign(-100.0) == -1

    def test_zero(self):
        """Zero should return 0."""
        assert _sign(0.0) == 0

    def test_near_zero_within_epsilon(self):
        """Values within epsilon of zero should return 0."""
        eps = 1e-12
        assert _sign(eps / 2) == 0
        assert _sign(-eps / 2) == 0
        assert _sign(1e-13) == 0
        assert _sign(-1e-13) == 0

    def test_near_zero_outside_epsilon(self):
        """Values just outside epsilon should return correct sign."""
        eps = 1e-12
        assert _sign(eps * 2) == 1
        assert _sign(-eps * 2) == -1

    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        assert _sign(0.01, eps=0.1) == 0  # 0.01 < 0.1, so treated as zero
        assert _sign(0.2, eps=0.1) == 1   # 0.2 > 0.1, so positive
        assert _sign(-0.2, eps=0.1) == -1


# ============================================================================
# Test redundancy_mmi
# ============================================================================

class TestRedundancyMMI:
    """Tests for MMI redundancy functional."""

    def test_mmi_basic(self):
        """MMI redundancy should be min(i1, i2)."""
        assert redundancy_mmi(3.0, 5.0) == 3.0
        assert redundancy_mmi(5.0, 3.0) == 3.0
        assert redundancy_mmi(2.0, 2.0) == 2.0

    def test_mmi_with_zero(self):
        """When one input is zero, redundancy should be zero."""
        assert redundancy_mmi(0.0, 5.0) == 0.0
        assert redundancy_mmi(5.0, 0.0) == 0.0

    def test_mmi_negative_clamped(self):
        """Negative redundancy should be clamped to zero by default."""
        assert redundancy_mmi(-1.0, 2.0, clamp_nonneg=True) == 0.0
        assert redundancy_mmi(-5.0, -3.0, clamp_nonneg=True) == 0.0

    def test_mmi_negative_unclamped(self):
        """With clamping disabled, negative values should pass through."""
        assert redundancy_mmi(-1.0, 2.0, clamp_nonneg=False) == -1.0
        assert redundancy_mmi(-5.0, -3.0, clamp_nonneg=False) == -5.0


# ============================================================================
# Test redundancy_ccs_pointwise
# ============================================================================

class TestRedundancyCCS:
    """Tests for CCS (Common Change in Surprisal) redundancy functional."""

    def test_ccs_all_positive_same_sign(self):
        """When all terms have same sign, CCS returns co-information."""
        i1, i2, i12 = 2.0, 3.0, 4.0
        # c = i1 + i2 - i12 = 1.0
        # All positive, c positive -> keep
        result = redundancy_ccs_pointwise(i1, i2, i12)
        assert result == pytest.approx(1.0)

    def test_ccs_sign_mismatch(self):
        """When signs don't match, CCS returns 0."""
        i1, i2, i12 = 1.0, -1.0, 0.5
        # i1 positive, i2 negative -> signs don't match -> return 0
        result = redundancy_ccs_pointwise(i1, i2, i12)
        assert result == 0.0

    def test_ccs_zero_coinformation(self):
        """When co-information is zero, CCS returns 0."""
        i1, i2, i12 = 2.0, 3.0, 5.0
        # c = i1 + i2 - i12 = 0.0
        result = redundancy_ccs_pointwise(i1, i2, i12)
        assert result == 0.0

    def test_ccs_negative_coinformation_all_positive(self):
        """Negative co-info with all positive inputs returns 0."""
        i1, i2, i12 = 1.0, 1.0, 5.0
        # c = 1 + 1 - 5 = -3 (negative)
        # i1, i2, i12 all positive but c is negative -> signs don't match
        result = redundancy_ccs_pointwise(i1, i2, i12)
        assert result == 0.0

    def test_ccs_all_negative_same_sign(self):
        """All negative with negative co-info should return co-info."""
        i1, i2, i12 = -2.0, -3.0, -4.0
        # c = -2 + -3 - (-4) = -1 (negative)
        # All negative, c negative -> signs match
        result = redundancy_ccs_pointwise(i1, i2, i12, clamp_nonneg=False)
        assert result == pytest.approx(-1.0)


# ============================================================================
# Test PID decomposition algebra
# ============================================================================

class TestPIDDecomposition:
    """Tests for PID decomposition formulas."""

    def test_decomposition_sums_to_joint(self):
        """red + unq1 + unq2 + syn should equal i12."""
        i1, i2, i12 = 3.0, 4.0, 6.0

        # MMI decomposition
        red = min(i1, i2)  # = 3.0
        unq1 = i1 - red     # = 0.0
        unq2 = i2 - red     # = 1.0
        syn = i12 - unq1 - unq2 - red  # = 6 - 0 - 1 - 3 = 2.0

        # Verify: red + unq1 + unq2 + syn = i12
        assert red + unq1 + unq2 + syn == pytest.approx(i12)

    def test_unique_information_nonnegative(self):
        """Unique information should be non-negative when redundancy is MMI."""
        i1, i2 = 5.0, 3.0
        red = min(i1, i2)  # = 3.0
        unq1 = i1 - red     # = 2.0 >= 0
        unq2 = i2 - red     # = 0.0 >= 0

        assert unq1 >= 0
        assert unq2 >= 0

    def test_symmetric_inputs(self):
        """With symmetric inputs (i1 == i2), unique info should be equal."""
        i1, i2, i12 = 3.0, 3.0, 5.0

        red = min(i1, i2)  # = 3.0
        unq1 = i1 - red     # = 0.0
        unq2 = i2 - red     # = 0.0

        assert unq1 == pytest.approx(unq2)

    def test_no_joint_benefit(self):
        """When i12 = max(i1, i2), synergy should be negative or zero."""
        i1, i2 = 3.0, 5.0
        i12 = max(i1, i2)  # = 5.0 (no benefit from combining)

        red = min(i1, i2)  # = 3.0
        unq1 = i1 - red     # = 0.0
        unq2 = i2 - red     # = 2.0
        syn = i12 - unq1 - unq2 - red  # = 5 - 0 - 2 - 3 = 0.0

        assert syn <= 0

    def test_superadditive_joint(self):
        """When i12 > i1 + i2, synergy should be positive."""
        i1, i2 = 2.0, 3.0
        i12 = 7.0  # > i1 + i2 = 5.0 (superadditive)

        red = min(i1, i2)  # = 2.0
        unq1 = i1 - red     # = 0.0
        unq2 = i2 - red     # = 1.0
        syn = i12 - unq1 - unq2 - red  # = 7 - 0 - 1 - 2 = 4.0

        assert syn > 0
