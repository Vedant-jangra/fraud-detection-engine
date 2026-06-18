import pytest
import numpy as np
from processing.drift_detector import compute_psi


class TestPSIComputation:
    def test_identical_distributions_returns_zero(self):
        data = np.random.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_detects_drift(self):
        expected = np.random.randn(1000)
        actual = np.random.randn(1000) + 3.0
        psi = compute_psi(expected, actual)
        assert psi > 0.2

    def test_moderate_drift(self):
        expected = np.random.randn(1000)
        actual = np.random.randn(1000) + 0.5
        psi = compute_psi(expected, actual)
        assert 0.01 < psi < 1.0

    def test_handles_different_sizes(self):
        expected = np.random.randn(500)
        actual = np.random.randn(2000)
        psi = compute_psi(expected, actual)
        assert isinstance(psi, float)
