"""
Volatility Analysis and Forecasting

Tools for analyzing historical volatility, term structure,
and making volatility forecasts.

Includes:
    - Volatility surface visualization
    - Term structure analysis
    - Volatility forecasting
    - Implied volatility extraction
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy import interpolate, optimize
from ..utils.math import black_scholes


class VolatilityTermStructure:
    """
    Analyze term structure of volatility.
    
    Term structure shows how implied volatility varies with maturity.
    
    Examples:
        >>> # Build term structure
        >>> maturities = [0.25, 0.5, 1.0, 2.0]  # 3m, 6m, 1y, 2y
        >>> vols = [0.18, 0.17, 0.16, 0.15]
        >>> term_struct = VolatilityTermStructure(maturities, vols)
        >>> 
        >>> # Interpolate volatility at arbitrary maturity
        >>> vol_3month = term_struct.interpolate(0.25)
        >>> vol_18month = term_struct.interpolate(1.5)
    """
    
    def __init__(
        self,
        maturities: np.ndarray,
        volatilities: np.ndarray,
        method: str = 'cubic'
    ):
        """
        Initialize term structure.
        
        Parameters:
            maturities (np.ndarray): Array of times to maturity
            volatilities (np.ndarray): Implied volatilities at each maturity
            method (str): Interpolation method ('linear', 'cubic')
        """
        self.maturities = np.asarray(maturities, dtype=float)
        self.volatilities = np.asarray(volatilities, dtype=float)
        
        if len(self.maturities) != len(self.volatilities):
            raise ValueError("maturities and volatilities must have same length")
        
        # Sort by maturity
        idx = np.argsort(self.maturities)
        self.maturities = self.maturities[idx]
        self.volatilities = self.volatilities[idx]
        
        # Set up interpolation
        if method == 'cubic':
            self.interp_func = interpolate.CubicSpline(self.maturities, self.volatilities)
        elif method == 'linear':
            self.interp_func = interpolate.interp1d(
                self.maturities, self.volatilities, kind='linear', 
                fill_value='extrapolate'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def interpolate(self, T: float) -> float:
        """
        Interpolate volatility at maturity T.
        
        Parameters:
            T (float): Time to maturity
        
        Returns:
            float: Interpolated volatility
        
        Examples:
            >>> term_struct = VolatilityTermStructure([0.25, 1.0], [0.20, 0.15])
            >>> vol_6m = term_struct.interpolate(0.5)
        """
        return float(self.interp_func(T))
    
    def slope(self, T: float) -> float:
        """
        Slope of term structure at maturity T.
        
        Returns:
            float: dσ/dT (positive = upward sloping)
        """
        delta = 0.0001
        return (self.interpolate(T + delta) - self.interpolate(T - delta)) / (2 * delta)
    
    def is_upward_sloping(self, T: float = None) -> bool:
        """Check if term structure is upward sloping."""
        if T is None:
            T = np.mean(self.maturities)
        return self.slope(T) > 0
    
    def level(self) -> float:
        """Get average volatility level."""
        return np.mean(self.volatilities)
    
    def steepness(self) -> float:
        """Get steepness: difference between long and short end."""
        return self.volatilities[-1] - self.volatilities[0]


class ImpliedVolatilitySurface:
    """
    Build and interpolate implied volatility surface.
    
    Surface shows how IV varies with both strike and maturity.
    
    Examples:
        >>> # 2D surface data
        >>> strikes = np.array([0.95, 1.0, 1.05])
        >>> maturities = np.array([0.25, 0.5, 1.0])
        >>> ivs = np.array([[0.18, 0.17, 0.16],
        ...                 [0.17, 0.16, 0.15],
        ...                 [0.16, 0.15, 0.14]])
        >>> 
        >>> surface = ImpliedVolatilitySurface(strikes, maturities, ivs)
        >>> vol = surface.interpolate(0.98, 0.4)  # IV at K=0.98, T=0.4
    """
    
    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ):
        """
        Initialize IV surface.
        
        Parameters:
            strikes (np.ndarray): Strike prices (1D array)
            maturities (np.ndarray): Times to maturity (1D array)
            implied_vols (np.ndarray): IV values (2D: len(maturities) x len(strikes))
        """
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.implied_vols = np.asarray(implied_vols, dtype=float)
        
        if self.implied_vols.shape != (len(self.maturities), len(self.strikes)):
            raise ValueError("implied_vols shape must be (len(maturities), len(strikes))")
        
        # Set up 2D interpolation
        self.interp_func = interpolate.RectBivariateSpline(
            self.maturities, self.strikes, self.implied_vols,
            kx=min(3, len(self.maturities)-1),
            ky=min(3, len(self.strikes)-1)
        )
    
    def interpolate(self, K: float, T: float) -> float:
        """
        Interpolate IV at strike K and maturity T.
        
        Parameters:
            K (float): Strike price (as fraction of spot)
            T (float): Time to maturity
        
        Returns:
            float: Implied volatility
        """
        return float(self.interp_func(T, K)[0, 0])
    
    def smile(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility smile at maturity T.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (strikes, volatilities)
        
        Examples:
            >>> surface = ImpliedVolatilitySurface(strikes, maturities, ivs)
            >>> K, vols = surface.smile(0.5)
            >>> # vols shows smile pattern (higher at extremes)
        """
        vols = np.array([self.interpolate(K, T) for K in self.strikes])
        return self.strikes, vols
    
    def skew(self, T: float) -> float:
        """
        Calculate volatility skew at maturity T.
        
        Skew = IV(OTM put) - IV(OTM call)
        
        Returns:
            float: Skew value (positive = put skew)
        """
        if len(self.strikes) < 2:
            return 0.0
        
        iv_low = self.interpolate(self.strikes[0], T)
        iv_high = self.interpolate(self.strikes[-1], T)
        return iv_low - iv_high


class VolatilitySwap:
    """
    Valuation of volatility swaps.
    
    A volatility swap pays off on realized volatility vs. strike.
    
    Payoff at maturity:
        Vega * (σ_realized - K_vol)
    
    Examples:
        >>> swap = VolatilitySwap(
        ...     strike=0.20,           # Vol strike (20%)
        ...     vega_notional=100000,  # $100k vega exposure
        ...     T=1.0                  # 1 year
        ... )
        >>> 
        >>> # At maturity with realized vol = 22%
        >>> payoff = swap.payoff(realized_vol=0.22)
    """
    
    def __init__(
        self,
        strike: float,
        vega_notional: float,
        T: float
    ):
        """
        Initialize volatility swap.
        
        Parameters:
            strike (float): Volatility strike (e.g., 0.20 for 20%)
            vega_notional (float): Notional size in vega terms
            T (float): Time to maturity
        """
        self.strike = strike
        self.vega_notional = vega_notional
        self.T = T
    
    def payoff(self, realized_vol: float) -> float:
        """
        Calculate payoff at maturity.
        
        Parameters:
            realized_vol (float): Realized volatility
        
        Returns:
            float: Payoff to long position
        
        Examples:
            >>> swap = VolatilitySwap(strike=0.20, vega_notional=100000, T=1.0)
            >>> payoff = swap.payoff(realized_vol=0.22)
            >>> print(f"Payoff: ${payoff:.2f}")  # +$200,000
        """
        return self.vega_notional * (realized_vol - self.strike)
    
    def breakeven_vol(self) -> float:
        """Breakeven volatility (= strike)."""
        return self.strike


def compare_volatility_models(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Compare different volatility estimation methods.
    
    Returns estimates from:
    - Historical volatility
    - EWMA
    - GARCH
    
    Parameters:
        returns (np.ndarray): Time series of returns
        periods_per_year (int): Trading periods per year
    
    Returns:
        Dict[str, float]: Volatility estimates
    
    Examples:
        >>> returns = np.random.normal(0, 0.01, 252)
        >>> vols = compare_volatility_models(returns)
        >>> for model, vol in vols.items():
        ...     print(f"{model}: {vol:.4f}")
    """
    from ..models.volatility import (
        HistoricalVolatility, EWMAVolatility, GARCHVolatility
    )
    
    results = {}
    
    # Historical
    hist = HistoricalVolatility(returns, periods_per_year)
    results['Historical'] = hist.estimate()
    
    # EWMA
    ewma = EWMAVolatility(returns, periods_per_year=periods_per_year)
    results['EWMA'] = ewma.estimate()
    
    # GARCH
    try:
        garch = GARCHVolatility(returns, periods_per_year)
        results['GARCH(1,1)'] = garch.estimate()
    except:
        results['GARCH(1,1)'] = np.nan
    
    return results
