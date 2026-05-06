"""
Tests for Exotic Options Module
"""

import pytest
import numpy as np
from derivatives.pricing.exotic_options import (
    BarrierOption, LookbackOption, BasketOption,
    DigitalOption, AsianOption
)


class TestBarrierOption:
    """Test barrier option pricing."""
    
    def test_up_out_call(self):
        """Test up-and-out call."""
        barrier = BarrierOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            barrier=120, barrier_type='up-out',
            n_paths=5000, random_seed=42
        )
        
        price, se = barrier.price()
        
        # Up-out call should be cheaper than vanilla call
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        vanilla_price = bs.call_price()
        
        assert price < vanilla_price
        assert price > 0
        assert se > 0
    
    def test_down_out_put(self):
        """Test down-and-out put."""
        barrier = BarrierOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            barrier=80, barrier_type='down-out',
            n_paths=5000, random_seed=42
        )
        
        price, se = barrier.price()
        
        # Down-out put should be cheaper than vanilla put
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        vanilla_price = bs.put_price()
        
        assert price < vanilla_price
        assert price > 0
    
    def test_knock_in_call(self):
        """Test knock-in call."""
        barrier = BarrierOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            barrier=120, barrier_type='up-in',
            n_paths=5000
        )
        
        price, se = barrier.price()
        
        # Knock-in + knock-out should equal vanilla
        ko = BarrierOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            barrier=120, barrier_type='up-out',
            n_paths=5000
        )
        ko_price, _ = ko.price()
        
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        vanilla_price = bs.call_price()
        
        # knockin + knockout ≈ vanilla
        total = price + ko_price
        assert np.isclose(total, vanilla_price, rtol=0.1)


class TestLookbackOption:
    """Test lookback option pricing."""
    
    def test_floating_lookback_call(self):
        """Test floating strike lookback call."""
        lookback = LookbackOption(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            lookback_type='floating', option_type='call',
            n_paths=5000
        )
        
        price, se = lookback.price()
        
        # Floating strike lookback is always ITM
        assert price > 0
        # More expensive than vanilla call
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        vanilla = bs.call_price()
        
        assert price > vanilla
    
    def test_fixed_lookback_call(self):
        """Test fixed strike lookback call."""
        lookback = LookbackOption(
            S0=100, K=90, T=1.0, r=0.05, sigma=0.2,
            lookback_type='fixed', option_type='call',
            n_paths=5000
        )
        
        price, se = lookback.price()
        assert price > 0
        assert se > 0
    
    def test_geometric_closed_form(self):
        """Test geometric averaging closed form."""
        lookback = LookbackOption(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            lookback_type='floating', option_type='call',
            n_paths=1000
        )
        
        price_cf = lookback.price_geometric_closed_form()
        
        assert price_cf > 0
        # Should be more than vanilla call
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        assert price_cf > bs.call_price()


class TestBasketOption:
    """Test basket option pricing."""
    
    def test_basket_initialization(self):
        """Test basket option initialization."""
        S0 = np.array([100, 50, 150])
        weights = np.array([0.5, 0.3, 0.2])
        sigma = np.array([0.2, 0.25, 0.15])
        correlation = np.eye(3)
        
        basket = BasketOption(
            S0=S0, weights=weights, K=100, T=1.0, r=0.05,
            sigma=sigma, correlation=correlation
        )
        
        assert basket.n_assets == 3
        assert np.isclose(np.sum(basket.weights), 1.0)
    
    def test_basket_call_price(self):
        """Test basket call pricing."""
        S0 = np.array([100, 100, 100])
        weights = np.array([1/3, 1/3, 1/3])
        sigma = np.array([0.2, 0.2, 0.2])
        
        basket = BasketOption(
            S0=S0, weights=weights, K=100, T=1.0, r=0.05,
            sigma=sigma, n_paths=5000, random_seed=42
        )
        
        price, se = basket.price()
        
        # Equal weighted portfolio should behave like single stock
        assert price > 0
        assert se > 0
    
    def test_basket_with_correlation(self):
        """Test basket with non-zero correlation."""
        S0 = np.array([100, 100])
        weights = np.array([0.5, 0.5])
        sigma = np.array([0.2, 0.2])
        
        # High correlation
        corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
        
        # Low correlation
        corr_low = np.array([[1.0, 0.1], [0.1, 1.0]])
        
        basket_high = BasketOption(
            S0=S0, weights=weights, K=100, T=1.0, r=0.05,
            sigma=sigma, correlation=corr_high, n_paths=5000
        )
        
        basket_low = BasketOption(
            S0=S0, weights=weights, K=100, T=1.0, r=0.05,
            sigma=sigma, correlation=corr_low, n_paths=5000
        )
        
        price_high, _ = basket_high.price()
        price_low, _ = basket_low.price()
        
        # High correlation should give different price than low correlation
        assert abs(price_high - price_low) > 0.1


class TestDigitalOption:
    """Test digital option pricing."""
    
    def test_digital_call_atm(self):
        """Test ATM digital call."""
        digital = DigitalOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            payoff_amount=100
        )
        
        price, se = digital.price()
        
        # Digital call should be positive
        assert price > 0
        assert price < 100 * np.exp(-0.05)  # Less than discounted payoff
    
    def test_digital_put_itm(self):
        """Test ITM digital put."""
        digital = DigitalOption(
            S0=90, K=100, T=1.0, r=0.05, sigma=0.2,
            payoff_amount=100, option_type='put'
        )
        
        price, se = digital.price()
        
        # ITM digital put should be valuable
        assert price > 50  # More than 50% of payoff
    
    def test_digital_as_spread(self):
        """Test digital can be replicated with call spread."""
        # Digital = narrow call spread
        K = 100
        spread_width = 1
        
        digital = DigitalOption(S0=100, K=K, T=1.0, r=0.05, sigma=0.2)
        
        # Digital approximation: (C(K) - C(K+width)) / width
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        
        bs1 = BlackScholesAnalytic(100, K, 1.0, 0.05, 0.2)
        bs2 = BlackScholesAnalytic(100, K+spread_width, 1.0, 0.05, 0.2)
        
        spread_price = (bs1.call_price() - bs2.call_price()) / spread_width
        digital_price, _ = digital.price()
        
        # Prices should be close
        assert abs(spread_price - digital_price) < 5


class TestAsianOption:
    """Test Asian option pricing."""
    
    def test_arithmetic_asian_call(self):
        """Test arithmetic Asian call."""
        asian = AsianOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            averaging='arithmetic', n_paths=5000
        )
        
        price, se = asian.price()
        
        # Asian should be cheaper than European (averaging reduces payoff)
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        
        assert price < bs.call_price()
        assert price > 0
    
    def test_geometric_asian_call(self):
        """Test geometric Asian call."""
        asian = AsianOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            averaging='geometric', n_paths=5000
        )
        
        price_mc, se = asian.price()
        price_cf = asian.price_geometric_closed_form()
        
        # Closed form and MC should be close
        assert np.isclose(price_mc, price_cf, rtol=0.1)
    
    def test_asian_price_reduction(self):
        """Test that Asian is cheaper than European."""
        asian = AsianOption(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            n_paths=10000
        )
        asian_price, _ = asian.price()
        
        from derivatives.pricing.black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        european_price = bs.call_price()
        
        # Asian should be cheaper due to averaging
        reduction = (european_price - asian_price) / european_price
        assert 0 < reduction < 0.5  # 0-50% cheaper
