"""
Tests for Brownian Motion Module

Tests generation of standard Brownian motion paths and their properties.
"""

import pytest
import numpy as np
from derivatives.models.brownian import BrownianMotion, BridgeBrownianMotion


class TestBrownianMotion:
    """Test standard Brownian motion."""
    
    @pytest.fixture
    def bm(self):
        """Create standard Brownian motion."""
        return BrownianMotion(T=1.0, n_steps=252, n_paths=1000, random_seed=42)
    
    def test_initialization(self, bm):
        """Test initialization parameters."""
        assert bm.T == 1.0
        assert bm.n_steps == 252
        assert bm.n_paths == 1000
        assert np.isclose(bm.dt, 1.0 / 252)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Negative time
        with pytest.raises(ValueError):
            BrownianMotion(T=-1.0)
        
        # Zero steps
        with pytest.raises(ValueError):
            BrownianMotion(n_steps=0)
        
        # Negative paths
        with pytest.raises(ValueError):
            BrownianMotion(n_paths=-1)
    
    def test_generate_paths_shape(self, bm):
        """Test that paths have correct shape."""
        paths = bm.generate_paths()
        assert paths.shape == (bm.n_steps + 1, bm.n_paths)
    
    def test_initial_condition(self, bm):
        """Test that W(0) = 0."""
        paths = bm.generate_paths()
        assert np.allclose(paths[0, :], 0, atol=1e-10)
    
    def test_final_variance(self, bm):
        """Test that Var(W(T)) ≈ T."""
        paths = bm.generate_paths()
        final_variance = np.var(paths[-1, :], ddof=1)
        assert np.isclose(final_variance, bm.T, atol=0.1)
    
    def test_final_mean(self, bm):
        """Test that E[W(T)] ≈ 0."""
        paths = bm.generate_paths()
        final_mean = np.mean(paths[-1, :])
        assert np.abs(final_mean) < 0.05
    
    def test_increments_normality(self, bm):
        """Test that increments are approximately normal."""
        dW = bm.generate_increments()
        
        # Mean should be close to 0
        assert np.abs(np.mean(dW)) < 0.05
        
        # Variance should be close to dt
        assert np.isclose(np.var(dW, ddof=1), bm.dt, atol=0.01)
    
    def test_increments_independence(self, bm):
        """Test that increments are approximately independent."""
        dW = bm.generate_increments()
        
        # Correlation between consecutive increments should be small
        if dW.shape[0] > 1:
            dW_flat = dW.flatten()
            corr = np.corrcoef(dW_flat[:-1], dW_flat[1:])[0, 1]
            assert np.abs(corr) < 0.1
    
    def test_reproducibility(self):
        """Test that same seed produces same paths."""
        bm1 = BrownianMotion(T=1.0, n_steps=100, n_paths=10, random_seed=123)
        bm2 = BrownianMotion(T=1.0, n_steps=100, n_paths=10, random_seed=123)
        
        paths1 = bm1.generate_paths()
        paths2 = bm2.generate_paths()
        
        assert np.allclose(paths1, paths2)
    
    def test_verify_properties(self, bm):
        """Test property verification method."""
        paths = bm.generate_paths()
        props = BrownianMotion.verify_properties(paths, T=bm.T)
        
        assert 'initial_value' in props
        assert 'mean_final' in props
        assert 'var_final' in props
        assert 'increments_independent' in props
        assert 'normality_test' in props
        
        # Check values
        assert props['initial_value'] < 1e-10
        assert np.abs(props['mean_final']) < 0.1
        assert np.isclose(props['var_final'], bm.T, atol=0.2)


class TestBridgeBrownianMotion:
    """Test Brownian bridge."""
    
    def test_boundary_conditions(self):
        """Test that B(0) = 0 and B(T) = 0."""
        bridge = BridgeBrownianMotion(T=1.0, n_steps=100, n_paths=100)
        paths = bridge.generate_paths()
        
        # Check initial condition
        assert np.allclose(paths[0, :], 0)
        
        # Check final condition
        assert np.allclose(paths[-1, :], 0, atol=1e-10)
    
    def test_intermediate_values(self):
        """Test that intermediate values are bounded."""
        bridge = BridgeBrownianMotion(T=1.0, n_steps=100, n_paths=1000)
        paths = bridge.generate_paths()
        
        # Middle values should be smaller than standard BM
        bm = BrownianMotion(T=1.0, n_steps=100, n_paths=1000)
        bm_paths = bm.generate_paths()
        
        # Standard deviation of bridge should be smaller
        bridge_std = np.std(paths)
        bm_std = np.std(bm_paths)
        assert bridge_std < bm_std
