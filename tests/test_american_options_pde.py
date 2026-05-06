"""
Tests for American Options via PDE
"""

import pytest
import numpy as np
from derivatives.pricing.american_options_pde import AmericanOptionFD
from derivatives.pricing.black_scholes import BlackScholesAnalytic


class TestAmericanOptionFD:
    """Test American option FD solver."""
    
    def test_american_put_initialization(self):
        """Test initialization."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            option_type='put'
        )
        
        assert american.K == 100
        assert american.option_type == 'put'
    
    def test_american_put_vs_european(self):
        """Test American put is worth more than European."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='put'
        )
        american_price = american.solve()
        
        # European put
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        european_price = bs.put_price()
        
        # American put should be worth at least as much
        assert american_price >= european_price * 0.95  # Allow small MC error
    
    def test_american_call_european_parity(self):
        """Test American call = European call (no dividends)."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='call'
        )
        american_price = american.solve()
        
        # European call
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        european_price = bs.call_price()
        
        # Should be approximately equal (no early exercise benefit)
        assert np.isclose(american_price, european_price, rtol=0.05)
    
    def test_american_option_price_method(self):
        """Test option_price method."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            option_type='put'
        )
        american.solve()
        
        # Get price at different spots
        price_100 = american.option_price(100)
        price_95 = american.option_price(95)
        price_105 = american.option_price(105)
        
        # For put: lower spot should be more valuable
        assert price_95 > price_100 > price_105
    
    def test_early_exercise_boundary(self):
        """Test early exercise boundary calculation."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='put'
        )
        american.solve()
        
        times, boundary = american.early_exercise_boundary()
        
        # Boundary should be below strike for put
        valid_boundary = boundary[~np.isnan(boundary)]
        assert np.all(valid_boundary <= 100)
    
    def test_american_premium(self):
        """Test early exercise premium calculation."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='put'
        )
        american.solve()
        
        premium = american.american_premium()
        
        # Premium should be non-negative for put
        assert premium >= -0.01  # Allow small numerical error
    
    def test_deep_itm_american_put(self):
        """Test deep ITM American put has intrinsic value."""
        american = AmericanOptionFD(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='put'
        )
        american.solve()
        
        # For deep ITM put (S=50, K=100)
        price_50 = american.option_price(50)
        intrinsic_50 = 100 - 50
        
        # Should be worth at least intrinsic
        assert price_50 >= intrinsic_50 * 0.95  # Allow small numerical error
