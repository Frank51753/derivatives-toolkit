"""
Tests for Hedging Strategies Module
"""

import pytest
import numpy as np
from derivatives.analytics.hedging import (
    DeltaHedger, PortfolioHedge, HedgingStrategies
)


class TestDeltaHedger:
    """Test delta hedging."""
    
    @pytest.fixture
    def hedger(self):
        """Create delta hedger."""
        return DeltaHedger(
            short_call_qty=100,
            option_delta=0.6,
            option_gamma=0.02,
            initial_stock_price=100
        )
    
    def test_initialization(self, hedger):
        """Test initialization."""
        assert hedger.short_calls == 100
        assert hedger.option_delta == 0.6
        assert hedger.long_stock == 60  # 100 * 0.6
    
    def test_rebalance(self, hedger):
        """Test rebalancing."""
        cost = hedger.rebalance(new_stock_price=105, new_delta=0.7)
        
        # Need 70 shares, have 60 -> buy 10 @ $105
        expected_cost = 10 * 105
        assert np.isclose(cost, expected_cost)
    
    def test_portfolio_delta(self, hedger):
        """Test portfolio delta."""
        delta = hedger.portfolio_delta()
        
        # 100 short calls * (-0.6 delta) + 60 long shares = 0
        assert np.isclose(delta, 0)
    
    def test_hedging_effectiveness(self, hedger):
        """Test hedge effectiveness."""
        effectiveness = hedger.hedging_effectiveness()
        
        # Perfectly hedged
        assert effectiveness == 1.0


class TestPortfolioHedge:
    """Test portfolio hedging."""
    
    def test_add_option(self):
        """Test adding option."""
        portfolio = PortfolioHedge()
        portfolio.add_option('call', strike=100, delta=0.6, quantity=100,
                           Greeks={'gamma': 0.02, 'vega': 10})
        
        assert len(portfolio.positions) == 1
    
    def test_portfolio_delta(self):
        """Test portfolio delta calculation."""
        portfolio = PortfolioHedge()
        portfolio.add_option('call', strike=100, delta=0.6, quantity=100)
        portfolio.add_option('put', strike=100, delta=-0.4, quantity=100)
        
        delta = portfolio.portfolio_delta()
        
        # 100 * 0.6 + 100 * (-0.4) = 20
        assert delta == 20
    
    def test_hedge_ratio(self):
        """Test hedge ratio calculation."""
        portfolio = PortfolioHedge()
        portfolio.add_option('call', strike=100, delta=0.6, quantity=100)
        
        ratio = portfolio.hedge_ratio()
        
        # Need to short 60 shares
        assert ratio == -60
    
    def test_portfolio_greeks(self):
        """Test Greeks summary."""
        portfolio = PortfolioHedge()
        portfolio.add_option('call', strike=100, delta=0.6, quantity=100,
                           Greeks={'gamma': 0.02, 'vega': 10, 'theta': -0.5})
        
        greeks = portfolio.get_greeks_summary()
        
        assert greeks['delta'] == 60
        assert greeks['gamma'] == 2  # 100 * 0.02
        assert greeks['vega'] == 1000  # 100 * 10
        assert greeks['theta'] == -50  # 100 * -0.5


class TestHedgingStrategies:
    """Test hedging strategy templates."""
    
    def test_long_call_hedge(self):
        """Test protective call strategy."""
        strategy = HedgingStrategies.long_call_hedge(
            stock_position=1000,
            strike=110,
            call_delta=0.4
        )
        
        assert 'Long Call Protection' in strategy['strategy']
        assert strategy['stock_qty'] == 1000
        assert strategy['max_loss'] is not None
    
    def test_put_spread_hedge(self):
        """Test put spread strategy."""
        strategy = HedgingStrategies.put_spread_hedge(
            stock_position=1000,
            long_put_strike=95,
            short_put_strike=90,
            long_put_price=2,
            short_put_price=0.5
        )
        
        assert 'Put Spread' in strategy['strategy']
        assert strategy['net_cost'] > 0
        assert strategy['max_loss'] > 0
