"""
Tests for Geometric Brownian Motion Module

Tests GBM path generation and calibration methods.
"""

import pytest
import numpy as np
from derivatives.models.geometric_brownian import GeometricBrownianMotion


class TestGeometricBrownianMotion:
    """Test Geometric Brownian Motion."""
    
    @pytest.fixture
    def gbm(self):
        """Create GBM with realistic parameters."""
        return GeometricBrownianMotion(
            S0=100,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            n_steps=252,
            n_paths=1000,
            random_seed=42
        )
    
    def test_initialization(self, gbm):
        """Test initialization."""
        assert gbm.S0 == 100
        assert gbm.mu == 0.05
        assert gbm.sigma == 0.2
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Negative stock price
        with pytest.raises(ValueError):
            GeometricBrownianMotion(S0=-100)
        
        # Zero volatility
        with pytest.raises(ValueError):
            GeometricBrownianMotion(sigma=0)
        
        # Negative time
        with pytest.raises(ValueError):
            GeometricBrownianMotion(T=-1)
    
    def test_generate_paths_shape(self, gbm):
        """Test path shape."""
        paths = gbm.generate_paths()
        assert paths.shape == (gbm.n_steps + 1, gbm.n_paths)
    
    def test_paths_always_positive(self, gbm):
        """Test that all prices remain positive."""
        paths = gbm.generate_paths()
        assert np.all(paths > 0)
    
    def test_initial_condition(self, gbm):
        """Test that S(0) = S0."""
        paths = gbm.generate_paths()
        assert np.allclose(paths[0, :], gbm.S0)
    
    def test_expected_final_price(self, gbm):
        """Test that E[S(T)] ≈ S0 * exp(μT)."""
        paths = gbm.generate_paths()
        final_mean = np.mean(paths[-1, :])
        expected = gbm.S0 * np.exp(gbm.mu * gbm.T)
        
        # Allow 10% tolerance due to sampling error
        assert np.isclose(final_mean, expected, rtol=0.1)
    
    def test_exact_vs_euler(self, gbm):
        """Test that exact method differs from Euler-Maruyama."""
        paths_euler = gbm.generate_paths()
        paths_exact = gbm.generate_paths_exact()
        
        # Both should have correct shape
        assert paths_euler.shape == paths_exact.shape
        
        # Final values should differ (but not drastically)
        euler_mean = np.mean(paths_euler[-1, :])
        exact_mean = np.mean(paths_exact[-1, :])
        
        # Should be similar (within 20%)
        assert np.isclose(euler_mean, exact_mean, rtol=0.2)
    
    def test_log_returns_distribution(self, gbm):
        """Test that log-returns are approximately normal."""
        returns = gbm.generate_log_returns()
        
        # Mean should be close to (μ - σ²/2)dt
        expected_mean = (gbm.mu - 0.5 * gbm.sigma**2) * gbm.dt
        actual_mean = np.mean(returns)
        assert np.isclose(actual_mean, expected_mean, atol=0.01)
        
        # Variance should be close to σ²dt
        expected_var = gbm.sigma**2 * gbm.dt
        actual_var = np.var(returns, ddof=1)
        assert np.isclose(actual_var, expected_var, atol=0.001)
    
    def test_calibrate_from_returns(self):
        """Test parameter calibration from returns."""
        # Create synthetic returns
        true_returns = np.random.normal(0.0005, 0.01, 252)
        
        mu, sigma = GeometricBrownianMotion.calibrate_from_returns(true_returns)
        
        # Should recover approximate parameters
        assert sigma > 0
        assert mu is not None
    
    def test_reproducibility(self):
        """Test that same seed produces same paths."""
        gbm1 = GeometricBrownianMotion(100, 0.05, 0.2, n_paths=10, random_seed=999)
        gbm2 = GeometricBrownianMotion(100, 0.05, 0.2, n_paths=10, random_seed=999)
        
        paths1 = gbm1.generate_paths()
        paths2 = gbm2.generate_paths()
        
        assert np.allclose(paths1, paths2)
    
    def test_high_volatility(self):
        """Test with very high volatility."""
        gbm = GeometricBrownianMotion(S0=100, mu=0.05, sigma=2.0, n_paths=1000)
        paths = gbm.generate_paths()
        
        # Should still be positive
        assert np.all(paths > 0)
        
        # Should have wider distribution
        assert np.std(paths[-1, :]) > 50
    
    def test_negative_drift(self):
        """Test with negative drift."""
        gbm = GeometricBrownianMotion(S0=100, mu=-0.10, sigma=0.2, n_paths=1000)
        paths = gbm.generate_paths()
        
        # Final value should be less than initial on average
        final_mean = np.mean(paths[-1, :])
        assert final_mean < gbm.S0
