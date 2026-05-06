"""
Tests for Forwards and Futures Module
"""

import pytest
import numpy as np
from derivatives.pricing.forwards_futures import (
    ForwardContract, FuturesContract, CarryBasedPricer
)


class TestForwardContract:
    """Test forward contract."""
    
    @pytest.fixture
    def forward(self):
        """Create forward contract."""
        return ForwardContract(
            spot=100, T=1.0, r=0.05, q=0.02,
            forward_price=103.05
        )
    
    def test_fair_value(self, forward):
        """Test fair forward price calculation."""
        fair = forward.fair_value()
        
        # F = S * e^((r-q)*T) = 100 * e^(0.03)
        expected = 100 * np.exp(0.03)
        assert np.isclose(fair, expected)
    
    def test_value_at_initiation(self, forward):
        """Test value is zero at initiation."""
        # Fair forward should be 103.05
        fair_forward = forward.fair_value()
        forward2 = ForwardContract(
            spot=100, T=1.0, r=0.05, q=0.02,
            forward_price=fair_forward
        )
        
        value = forward2.value()
        assert np.isclose(value, 0, atol=0.01)
    
    def test_profit_loss(self, forward):
        """Test P&L calculation."""
        pnl = forward.profit_loss(current_spot=110)
        
        # At maturity: P&L = current spot - forward price
        expected = 110 - 103.05
        assert np.isclose(pnl, expected, atol=0.01)
    
    def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection."""
        # Overpriced forward
        forward = ForwardContract(
            spot=100, T=1.0, r=0.05, q=0.02,
            forward_price=106  # Too high
        )
        
        arb = forward.arbitrage_opportunity()
        assert arb is not None
        assert 'Cash and Carry' in arb['strategy']


class TestFuturesContract:
    """Test futures contract."""
    
    def test_initialization(self):
        """Test futures initialization."""
        futures = FuturesContract(
            spot=4500, T=0.25, r=0.05,
            multiplier=250, contract_name='ES'
        )
        
        assert futures.contract_name == 'ES'
        assert futures.multiplier == 250
    
    def test_mark_to_market(self):
        """Test mark-to-market settlement."""
        futures = FuturesContract(spot=4500, T=0.25, r=0.05, multiplier=250)
        
        daily_pnl = futures.mark_to_market(4510)
        
        # P&L = (4510 - 4500) * 250 = $2500
        assert np.isclose(daily_pnl, 2500)
    
    def test_cumulative_pnl(self):
        """Test cumulative P&L tracking."""
        futures = FuturesContract(spot=4500, T=0.25, r=0.05, multiplier=250)
        
        futures.mark_to_market(4510)
        futures.mark_to_market(4505)
        futures.mark_to_market(4520)
        
        cumulative = futures.cumulative_pnl()
        
        # (4510-4500)*250 + (4505-4510)*250 + (4520-4505)*250
        expected = 10*250 - 5*250 + 15*250
        assert np.isclose(cumulative, expected)
    
    def test_contract_value(self):
        """Test contract notional value."""
        futures = FuturesContract(spot=100, T=1, r=0.05, multiplier=50)
        
        value = futures.contract_value()
        assert value == 100 * 50


class TestCarryBasedPricer:
    """Test carry-based pricing."""
    
    def test_fair_price(self):
        """Test fair price calculation with carry."""
        pricer = CarryBasedPricer(
            spot=1800, r=0.05, 
            storage_cost=0.01, convenience_yield=0.02,
            T=0.5
        )
        
        fair = pricer.fair_price()
        
        # F = 1800 * e^((0.05 + 0.01 - 0.02) * 0.5)
        expected = 1800 * np.exp(0.04 * 0.5)
        assert np.isclose(fair, expected)
    
    def test_implied_carry(self):
        """Test inferring carry from market price."""
        pricer = CarryBasedPricer(spot=100, r=0.05, T=1.0)
        
        market_price = 105.1  # Roughly e^(0.05)
        implied = pricer.implied_carry(market_price)
        
        assert np.isclose(implied, 0.05, atol=0.01)
