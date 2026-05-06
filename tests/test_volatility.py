"""
Tests for Volatility Analysis Module
"""

import pytest
import numpy as np
from derivatives.analytics.volatility import (
    VolatilityTermStructure, ImpliedVolatilitySurface,
    VolatilitySwap
)


class TestVolatilityTermStructure:
    """Test volatility term structure."""
    
    @pytest.fixture
    def term_struct(self):
        """Create term structure."""
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        vols = np.array([0.18, 0.17, 0.16, 0.15])
        return VolatilityTermStructure(maturities, vols, method='cubic')
    
    def test_initialization(self, term_struct):
        """Test initialization."""
        assert len(term_struct.maturities) == 4
        assert len(term_struct.volatilities) == 4
    
    def test_interpolation(self, term_struct):
        """Test interpolation."""
        # At known points
        assert np.isclose(term_struct.interpolate(0.25), 0.18, atol=1e-6)
        
        # Between points
        vol_mid = term_struct.interpolate(0.375)
        assert 0.17 < vol_mid < 0.18
    
    def test_slope(self, term_struct):
        """Test slope calculation."""
        slope = term_struct.slope(1.0)
        
        # Downward sloping curve should have negative slope
        assert slope < 0
    
    def test_is_upward_sloping(self, term_struct):
        """Test slope direction."""
        # This curve is downward sloping
        assert not term_struct.is_upward_sloping()


class TestImpliedVolatilitySurface:
    """Test IV surface."""
    
    @pytest.fixture
    def surface(self):
        """Create IV surface."""
        strikes = np.array([0.95, 1.0, 1.05])
        maturities = np.array([0.25, 0.5, 1.0])
        ivs = np.array([
            [0.18, 0.17, 0.16],
            [0.17, 0.16, 0.15],
            [0.16, 0.15, 0.14]
        ])
        return ImpliedVolatilitySurface(strikes, maturities, ivs)
    
    def test_interpolation(self, surface):
        """Test surface interpolation."""
        # At known points
        vol = surface.interpolate(1.0, 0.5)
        assert np.isclose(vol, 0.16, atol=0.01)
    
    def test_smile(self, surface):
        """Test volatility smile."""
        K, vols = surface.smile(0.5)
        
        assert len(K) == 3
        assert len(vols) == 3
    
    def test_skew(self, surface):
        """Test skew calculation."""
        skew = surface.skew(0.5)
        
        # This surface has put skew (higher vol at lower strikes)
        assert skew > 0


class TestVolatilitySwap:
    """Test volatility swap."""
    
    def test_initialization(self):
        """Test initialization."""
        swap = VolatilitySwap(strike=0.20, vega_notional=100000, T=1.0)
        
        assert swap.strike == 0.20
        assert swap.vega_notional == 100000
    
    def test_payoff(self):
        """Test payoff calculation."""
        swap = VolatilitySwap(strike=0.20, vega_notional=100000, T=1.0)
        
        payoff = swap.payoff(realized_vol=0.22)
        
        # (0.22 - 0.20) * 100000 = $2000
        assert payoff == 2000
    
    def test_negative_payoff(self):
        """Test negative payoff."""
        swap = VolatilitySwap(strike=0.20, vega_notional=100000, T=1.0)
        
        payoff = swap.payoff(realized_vol=0.18)
        
        # (0.18 - 0.20) * 100000 = -$2000
        assert payoff == -2000
