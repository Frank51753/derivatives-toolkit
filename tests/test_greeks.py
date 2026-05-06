"""
Tests for Greeks Calculation Module

Tests Greeks calculations and risk analysis utilities.
"""

import pytest
import numpy as np
from derivatives.analytics.greeks import (
    GreeksCalculator, 
    OptionRiskAnalyzer, 
    compute_greeks_vectorized
)


class TestGreeksCalculator:
    """Test Greeks calculator."""
    
    @pytest.fixture
    def calc(self):
        """Create Greeks calculator."""
        return GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
    
    def test_initialization(self, calc):
        """Test initialization."""
        assert calc.S == 100
        assert calc.K == 100
        assert calc.T == 0.25


class TestOptionRiskAnalyzer:
    """Test option risk analyzer."""
    
    def test_call_payoff(self):
        """Test call payoff calculation."""
        S_range = np.array([90, 100, 110])
        payoff = OptionRiskAnalyzer.payoff_diagram(K=100, S_range=S_range, 
                                                  premium=5, option_type='call')
        
        expected = np.array([-5, -5, 5])
        assert np.allclose(payoff, expected)
    
    def test_put_payoff(self):
        """Test put payoff calculation."""
        S_range = np.array([90, 100, 110])
        payoff = OptionRiskAnalyzer.payoff_diagram(K=100, S_range=S_range, 
                                                  premium=5, option_type='put')
        
        expected = np.array([5, -5, -5])
        assert np.allclose(payoff, expected)
    
    def test_call_breakeven(self):
        """Test call breakeven calculation."""
        be = OptionRiskAnalyzer.breakeven(K=100, premium=5, option_type='call')
        assert be == 105
    
    def test_put_breakeven(self):
        """Test put breakeven calculation."""
        be = OptionRiskAnalyzer.breakeven(K=100, premium=5, option_type='put')
        assert be == 95
    
    def test_long_call_max_profit(self):
        """Test long call maximum profit is unlimited."""
        max_profit = OptionRiskAnalyzer.max_profit(K=100, premium=5, 
                                                   option_type='call', 
                                                   position_type='long')
        assert max_profit == np.inf
    
    def test_long_call_max_loss(self):
        """Test long call maximum loss is premium."""
        max_loss = OptionRiskAnalyzer.max_loss(K=100, premium=5, 
                                               option_type='call', 
                                               position_type='long')
        assert max_loss == 5
    
    def test_long_put_max_profit(self):
        """Test long put maximum profit is K - premium."""
        max_profit = OptionRiskAnalyzer.max_profit(K=100, premium=5, 
                                                   option_type='put', 
                                                   position_type='long')
        assert max_profit == 95
    
    def test_long_put_max_loss(self):
        """Test long put maximum loss is premium."""
        max_loss = OptionRiskAnalyzer.max_loss(K=100, premium=5, 
                                               option_type='put', 
                                               position_type='long')
        assert max_loss == 5
    
    def test_short_call_max_profit(self):
        """Test short call maximum profit is premium."""
        max_profit = OptionRiskAnalyzer.max_profit(K=100, premium=5, 
                                                   option_type='call', 
                                                   position_type='short')
        assert max_profit == 5
    
    def test_short_call_max_loss(self):
        """Test short call maximum loss is unlimited."""
        max_loss = OptionRiskAnalyzer.max_loss(K=100, premium=5, 
                                               option_type='call', 
                                               position_type='short')
        assert max_loss == np.inf


class TestComputeGreeksVectorized:
    """Test vectorized Greeks computation."""
    
    def test_single_values(self):
        """Test with scalar inputs."""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        greeks = compute_greeks_vectorized(S, K, T, r, sigma, option_type='call')
        
        required_keys = ['delta', 'gamma', 'vega', 'theta', 'rho']
        for key in required_keys:
            assert key in greeks
    
    def test_array_inputs(self):
        """Test with array inputs."""
        S = np.array([90, 100, 110])
        K = 100
        T = 0.25
        r = 0.05
        sigma = np.array([0.15, 0.20, 0.25])
        
        greeks = compute_greeks_vectorized(S, K, T, r, sigma, option_type='call')
        
        # Should return arrays of same length
        assert len(greeks['delta']) == 3
        assert len(greeks['gamma']) == 3
        assert len(greeks['vega']) == 3
    
    def test_put_vs_call_vega(self):
        """Test that vega is same for calls and puts."""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        greeks_call = compute_greeks_vectorized(S, K, T, r, sigma, 
                                               option_type='call')
        greeks_put = compute_greeks_vectorized(S, K, T, r, sigma, 
                                              option_type='put')
        
        # Vega should be same for both
        assert np.isclose(greeks_call['vega'], greeks_put['vega'])
    
    def test_delta_call_put_parity(self):
        """Test delta put-call parity."""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        greeks_call = compute_greeks_vectorized(S, K, T, r, sigma, 
                                               option_type='call')
        greeks_put = compute_greeks_vectorized(S, K, T, r, sigma, 
                                              option_type='put')
        
        # Δ_call - Δ_put = 1
        delta_diff = greeks_call['delta'] - greeks_put['delta']
        assert np.isclose(delta_diff, 1.0, atol=1e-6)
