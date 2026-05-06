"""
Tests for Foreign Exchange Options Pricing Module
"""

import pytest
import numpy as np
from derivatives.pricing.foreign_exchange import FXOptionPricer, CurrencyPortfolio


class TestFXOptionPricer:
    """Test FX option pricer."""
    
    @pytest.fixture
    def fx(self):
        """Create FX option pricer."""
        return FXOptionPricer(
            spot=1.20, K=1.20, T=0.25,
            r_domestic=0.05, r_foreign=0.03,
            sigma=0.12
        )
    
    def test_initialization(self, fx):
        """Test initialization."""
        assert fx.spot == 1.20
        assert fx.K == 1.20
        assert fx.T == 0.25
    
    def test_call_price_bounds(self, fx):
        """Test call price is within bounds."""
        call = fx.call_price()
        
        assert call > 0
        assert call < fx.forward  # Less than forward
    
    def test_put_price_bounds(self, fx):
        """Test put price is within bounds."""
        put = fx.put_price()
        
        assert put > 0
        assert put < fx.K * np.exp(-fx.r_d * fx.T)
    
    def test_put_call_parity_fx(self, fx):
        """Test FX put-call parity."""
        call = fx.call_price()
        put = fx.put_price()
        
        # C - P = e^(-r_d*T) * (F - K)
        lhs = call - put
        rhs = np.exp(-fx.r_d * fx.T) * (fx.forward - fx.K)
        
        assert np.isclose(lhs, rhs, atol=1e-6)
    
    def test_delta_bounds(self, fx):
        """Test delta bounds."""
        call_delta = fx.delta('call')
        put_delta = fx.delta('put')
        
        assert 0 < call_delta < 1
        assert -1 < put_delta < 0
    
    def test_greeks_consistency(self, fx):
        """Test Greeks are consistent."""
        greeks = fx.all_greeks('call')
        
        required = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho_domestic', 'rho_foreign']
        for greek in required:
            assert greek in greeks


class TestCurrencyPortfolio:
    """Test currency portfolio."""
    
    def test_add_position(self):
        """Test adding FX position."""
        portfolio = CurrencyPortfolio('USD')
        portfolio.add_fx_position('EUR', quantity=1000000, spot_rate=1.20)
        
        assert 'EUR' in portfolio.positions
        assert portfolio.positions['EUR']['quantity'] == 1000000
    
    def test_portfolio_value(self):
        """Test portfolio valuation."""
        portfolio = CurrencyPortfolio('USD')
        portfolio.add_fx_position('EUR', quantity=1000000, spot_rate=1.20)
        
        market_rates = {'EUR': 1.25}
        value = portfolio.value(market_rates)
        
        assert value == 1000000 * 1.25
