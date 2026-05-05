"""Tests for derivatives.core.instruments module"""

import pytest
from derivatives.core.instruments import Option, EuropeanOption, AmericanOption, Portfolio


class TestOption:
    """Test Option class"""
    
    def test_call_payoff(self):
        """Test call option payoff"""
        call = Option(strike=100, maturity=1, option_type='call')
        assert call.payoff(105) == 5
        assert call.payoff(100) == 0
        assert call.payoff(95) == 0
    
    def test_put_payoff(self):
        """Test put option payoff"""
        put = Option(strike=100, maturity=1, option_type='put')
        assert put.payoff(95) == 5
        assert put.payoff(100) == 0
        assert put.payoff(105) == 0
    
    def test_option_profit(self):
        """Test option profit calculation"""
        call = Option(strike=100, maturity=1, option_type='call')
        profit = call.profit(105, premium=3)
        assert profit == 2
    
    def test_invalid_strike(self):
        """Test that negative strike raises error"""
        with pytest.raises(ValueError):
            Option(strike=-100, maturity=1)
    
    def test_invalid_maturity(self):
        """Test that negative maturity raises error"""
        with pytest.raises(ValueError):
            Option(strike=100, maturity=-1)
    
    def test_invalid_option_type(self):
        """Test that invalid option type raises error"""
        with pytest.raises(ValueError):
            Option(strike=100, maturity=1, option_type='invalid')


class TestPortfolio:
    """Test Portfolio class"""
    
    def test_portfolio_creation(self):
        """Test portfolio creation"""
        portfolio = Portfolio(cash=10000)
        assert portfolio.cash == 10000
        assert portfolio.initial_cash == 10000
    
    def test_buy(self):
        """Test buying assets"""
        portfolio = Portfolio(cash=10000)
        portfolio.buy('AAPL', 100, 50)
        assert portfolio.cash == 5000
        assert portfolio.positions['AAPL'] == 100
    
    def test_short(self):
        """Test short selling"""
        portfolio = Portfolio(cash=1000)
        portfolio.short('MSFT', 50, 30)
        assert portfolio.cash == 2500
        assert portfolio.positions['MSFT'] == -50
    
    def test_portfolio_value(self):
        """Test portfolio valuation"""
        portfolio = Portfolio(cash=5000)
        portfolio.buy('AAPL', 100, 50)
        
        prices = {'AAPL': 55}
        value = portfolio.value(prices)
        assert value == 10500  # 5000 cash + 100*55 stocks
    
    def test_insufficient_cash(self):
        """Test that buying without sufficient cash raises error"""
        portfolio = Portfolio(cash=1000)
        with pytest.raises(ValueError):
            portfolio.buy('AAPL', 100, 50)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
