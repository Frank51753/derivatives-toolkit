"""
Tests for Monte Carlo Simulation Module
"""

import pytest
import numpy as np
from derivatives.pricing.monte_carlo import MonteCarloSimulator


class TestMonteCarloSimulator:
    """Test Monte Carlo simulator."""
    
    @pytest.fixture
    def mc(self):
        """Create MC simulator."""
        return MonteCarloSimulator(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            n_paths=10000, n_steps=252, random_seed=42
        )
    
    def test_initialization(self, mc):
        """Test initialization."""
        assert mc.S0 == 100
        assert mc.K == 100
        assert mc.n_paths == 10000
    
    def test_paths_shape(self, mc):
        """Test that paths have correct shape."""
        assert mc.paths.shape == (253, 10000)
    
    def test_paths_always_positive(self, mc):
        """Test that all prices are positive."""
        assert np.all(mc.paths > 0)
    
    def test_european_call_price(self, mc):
        """Test European call pricing."""
        price, se = mc.price_european_call()
        
        # Black-Scholes price ~ 10.45
        assert 9 < price < 12
        # Standard error should be reasonable
        assert se < price / 10
    
    def test_european_put_price(self, mc):
        """Test European put pricing."""
        price, se = mc.price_european_put()
        
        # Should be positive
        assert price > 0
        assert se < price / 10
    
    def test_put_call_parity(self, mc):
        """Test put-call parity."""
        call_price, _ = mc.price_european_call()
        put_price, _ = mc.price_european_put()
        
        # C - P ≈ S - K*e^(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = mc.S0 - mc.K * np.exp(-mc.r * mc.T)
        
        # Allow 1% tolerance
        assert np.isclose(parity_lhs, parity_rhs, rtol=0.01)
    
    def test_american_option_price(self, mc):
        """Test American option pricing."""
        am_price, se = mc.price_american_option('call')
        
        # American call >= European call
        eu_price, _ = mc.price_european_call()
        assert am_price >= eu_price * 0.95  # Allow 5% MC error
    
    def test_asian_option_price(self, mc):
        """Test Asian option pricing."""
        price, se = mc.price_asian_option('call', 'arithmetic')
        
        assert price > 0
        assert se < price / 10
    
    def test_barrier_option_knockout(self, mc):
        """Test barrier knock-out option."""
        price, se = mc.price_barrier_option(
            'call', 'knockout', barrier_level=130
        )
        
        # Knock-out should be cheaper than vanilla
        vanilla_price, _ = mc.price_european_call()
        assert price <= vanilla_price
    
    def test_reproducibility(self):
        """Test that same seed gives same prices."""
        mc1 = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2, n_paths=1000, random_seed=123)
        mc2 = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2, n_paths=1000, random_seed=123)
        
        price1, _ = mc1.price_european_call()
        price2, _ = mc2.price_european_call()
        
        assert np.isclose(price1, price2)
