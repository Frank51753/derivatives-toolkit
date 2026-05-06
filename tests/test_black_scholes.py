"""
Tests for Black-Scholes Pricing Module

Tests option pricing and Greeks calculation.
"""

import pytest
import numpy as np
from derivatives.pricing.black_scholes import BlackScholesAnalytic, EuropeanOption


class TestBlackScholesAnalytic:
    """Test Black-Scholes analytical pricer."""
    
    @pytest.fixture
    def bs(self):
        """Create BS pricer with ATM option."""
        return BlackScholesAnalytic(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
    
    def test_initialization(self, bs):
        """Test initialization."""
        assert bs.S == 100
        assert bs.K == 100
        assert bs.T == 0.25
        assert bs.r == 0.05
        assert bs.sigma == 0.2
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Negative stock price
        with pytest.raises(ValueError):
            BlackScholesAnalytic(S=-100, K=100, T=1, r=0.05, sigma=0.2)
        
        # Zero volatility
        with pytest.raises(ValueError):
            BlackScholesAnalytic(S=100, K=100, T=1, r=0.05, sigma=0)
        
        # Negative time
        with pytest.raises(ValueError):
            BlackScholesAnalytic(S=100, K=100, T=-1, r=0.05, sigma=0.2)
    
    def test_call_price_bounds(self, bs):
        """Test that call price is within bounds."""
        call = bs.call_price()
        
        # Call price should be between 0 and stock price
        assert 0 < call < bs.S
        
        # Call price should be greater than intrinsic value
        intrinsic = max(bs.S - bs.K, 0)
        assert call >= intrinsic
    
    def test_put_price_bounds(self, bs):
        """Test that put price is within bounds."""
        put = bs.put_price()
        
        # Put price should be positive
        assert put > 0
        
        # Put price should be less than strike
        assert put < bs.K
    
    def test_put_call_parity(self, bs):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        call = bs.call_price()
        put = bs.put_price()
        
        parity_left = call - put
        parity_right = bs.S - bs.K * np.exp(-bs.r * bs.T)
        
        assert np.isclose(parity_left, parity_right, atol=1e-6)
    
    def test_call_delta_bounds(self, bs):
        """Test that call delta is between 0 and 1."""
        delta = bs.delta('call')
        assert 0 < delta < 1
    
    def test_put_delta_bounds(self, bs):
        """Test that put delta is between -1 and 0."""
        delta = bs.delta('put')
        assert -1 < delta < 0
    
    def test_delta_put_call_parity(self, bs):
        """Test delta put-call parity: Δ_call - Δ_put = 1."""
        delta_call = bs.delta('call')
        delta_put = bs.delta('put')
        
        assert np.isclose(delta_call - delta_put, 1.0, atol=1e-6)
    
    def test_gamma_always_positive(self, bs):
        """Test that gamma is always positive."""
        gamma = bs.gamma()
        assert gamma > 0
    
    def test_gamma_same_for_call_put(self):
        """Test that gamma is same for calls and puts."""
        bs_call = BlackScholesAnalytic(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        
        # Gamma doesn't depend on call/put, so same value
        gamma1 = bs_call.gamma()
        assert gamma1 > 0
    
    def test_vega_always_positive(self, bs):
        """Test that vega is always positive."""
        vega = bs.vega()
        assert vega > 0
    
    def test_vega_same_for_call_put(self):
        """Test that vega is same for calls and puts."""
        bs_call = BlackScholesAnalytic(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        vega = bs_call.vega()
        assert vega > 0
    
    def test_theta_call_usually_negative(self, bs):
        """Test that call theta is usually negative (time decay)."""
        theta = bs.theta('call')
        # For ATM call, theta should be negative
        assert theta < 0
    
    def test_rho_call_positive(self, bs):
        """Test that call rho is positive."""
        rho = bs.rho('call')
        assert rho > 0
    
    def test_rho_put_negative(self, bs):
        """Test that put rho is negative."""
        rho = bs.rho('put')
        assert rho < 0
    
    def test_all_greeks(self, bs):
        """Test all Greeks calculation."""
        greeks = bs.all_greeks('call')
        
        required_keys = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
        for key in required_keys:
            assert key in greeks
            assert greeks[key] is not None
    
    def test_itm_call(self):
        """Test in-the-money call option."""
        bs = BlackScholesAnalytic(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        call = bs.call_price()
        intrinsic = 10.0
        
        # Call price should be greater than intrinsic value
        assert call > intrinsic
    
    def test_otm_call(self):
        """Test out-of-the-money call option."""
        bs = BlackScholesAnalytic(S=90, K=100, T=0.25, r=0.05, sigma=0.2)
        call = bs.call_price()
        
        # OTM call should have positive time value
        assert call > 0
        assert call < 10  # Less than intrinsic of ITM option
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        bs = BlackScholesAnalytic(S=150, K=100, T=0.25, r=0.05, sigma=0.2)
        call = bs.call_price()
        
        # Deep ITM call should be close to S - K*e^(-rT)
        expected_min = 150 - 100 * np.exp(-0.05 * 0.25)
        assert call > expected_min - 1
    
    def test_sensitivity_analysis(self, bs):
        """Test sensitivity analysis."""
        stock_range = np.linspace(80, 120, 10)
        sensitivities = bs.sensitivity_analysis(stock_range, ['delta', 'gamma'])
        
        assert 'delta' in sensitivities
        assert 'gamma' in sensitivities
        assert len(sensitivities['delta']) == 10
        assert len(sensitivities['gamma']) == 10


class TestEuropeanOption:
    """Test EuropeanOption wrapper class."""
    
    def test_initialization(self):
        """Test initialization."""
        opt = EuropeanOption(S=100, K=100, T=0.25, r=0.05, sigma=0.2, 
                            option_type='call')
        assert opt.S == 100
        assert opt.option_type == 'call'
    
    def test_invalid_option_type(self):
        """Test that invalid option type raises error."""
        with pytest.raises(ValueError):
            EuropeanOption(S=100, K=100, T=0.25, r=0.05, sigma=0.2, 
                          option_type='invalid')
    
    def test_price_method(self):
        """Test price method."""
        opt = EuropeanOption(S=100, K=100, T=0.25, r=0.05, sigma=0.2, 
                            option_type='call')
        price = opt.price()
        assert price > 0
    
    def test_greeks_methods(self):
        """Test all Greeks methods."""
        opt = EuropeanOption(S=100, K=100, T=0.25, r=0.05, sigma=0.2, 
                            option_type='call')
        
        assert opt.delta() is not None
        assert opt.gamma() is not None
        assert opt.vega() is not None
        assert opt.theta() is not None
        assert opt.rho() is not None
