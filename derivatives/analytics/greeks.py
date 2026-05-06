"""
Greeks Calculation and Risk Management

This module provides comprehensive Greeks calculation for options,
including convenience functions for quick access and batch computation.

The Greeks measure option price sensitivity to various factors:
- Delta (Δ): Stock price sensitivity
- Gamma (Γ): Delta sensitivity (convexity)
- Vega (ν): Volatility sensitivity
- Theta (Θ): Time decay
- Rho (ρ): Interest rate sensitivity
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from ..utils.math import normal_pdf, normal_cdf


class GreeksCalculator:
    """
    Comprehensive Greeks calculator for options.
    
    Computes all Greeks and provides utilities for risk monitoring
    and sensitivity analysis.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Greeks calculator.
        
        Parameters:
            S (float): Stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Pre-calculate d1 and d2
        self._update_d1_d2()
    
    def _update_d1_d2(self):
        """Update d1 and d2 when parameters change."""
        ln_S_K = np.log(self.S / self.K)
        sqrt_T = np.sqrt(self.T)
        sigma_sqrt_T = self.sigma * sqrt_T
        
        self.d1 = (ln_S_K + (self.r + 0.5 * self.sigma**2) * self.T) / sigma_sqrt_T
        self.d2 = self.d1 - sigma_sqrt_T
        self.sqrt_T = sqrt_T
    
    def portfolio_greeks(
        self,
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio Greeks from multiple option positions.
        
        Parameters:
            positions (Dict[str, float]): Dictionary of {symbol: quantity}
                                         where quantity is shares held
                                         (positive for long, negative for short)
        
        Returns:
            Dict[str, float]: Aggregated portfolio Greeks
        
        Examples:
            >>> calc = GreeksCalculator(100, 100, 0.25, 0.05, 0.2)
            >>> positions = {
            ...     'stock': 100,      # 100 shares
            ...     'call': -10,       # 10 short calls
            ...     'put': 5           # 5 long puts
            ... }
            >>> portfolio_greeks = calc.portfolio_greeks(positions)
        """
        pass  # Implementation would aggregate Greeks from multiple options


class OptionRiskAnalyzer:
    """
    Analyze and visualize option risk metrics.
    """
    
    @staticmethod
    def payoff_diagram(
        K: float,
        S_range: np.ndarray,
        premium: float = 0,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Calculate option payoff at maturity.
        
        Parameters:
            K (float): Strike price
            S_range (np.ndarray): Stock price range at maturity
            premium (float): Option premium paid (for profit calculation)
            option_type (str): 'call' or 'put'
        
        Returns:
            np.ndarray: Payoffs at each stock price
        
        Examples:
            >>> S_range = np.linspace(80, 120, 50)
            >>> payoff = OptionRiskAnalyzer.payoff_diagram(
            ...     K=100, S_range=S_range, premium=5, option_type='call'
            ... )
            >>> # payoff[i] = profit at S_range[i]
        """
        if option_type == 'call':
            return np.maximum(S_range - K, 0) - premium
        elif option_type == 'put':
            return np.maximum(K - S_range, 0) - premium
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    @staticmethod
    def breakeven(
        K: float,
        premium: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate breakeven stock price.
        
        Parameters:
            K (float): Strike price
            premium (float): Option premium
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Breakeven stock price
        """
        if option_type == 'call':
            return K + premium
        elif option_type == 'put':
            return K - premium
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    @staticmethod
    def max_profit(
        K: float,
        premium: float,
        option_type: str = 'call',
        position_type: str = 'long'
    ) -> float:
        """
        Calculate maximum profit.
        
        Parameters:
            K (float): Strike price
            premium (float): Premium paid/received
            option_type (str): 'call' or 'put'
            position_type (str): 'long' or 'short'
        
        Returns:
            float: Maximum profit
        """
        if position_type == 'long':
            if option_type == 'call':
                return np.inf  # Unlimited profit
            else:  # put
                return K - premium
        else:  # short
            if option_type == 'call':
                return premium
            else:  # put
                return premium
    
    @staticmethod
    def max_loss(
        K: float,
        premium: float,
        option_type: str = 'call',
        position_type: str = 'long'
    ) -> float:
        """
        Calculate maximum loss.
        
        Parameters:
            K (float): Strike price
            premium (float): Premium paid/received
            option_type (str): 'call' or 'put'
            position_type (str): 'long' or 'short'
        
        Returns:
            float: Maximum loss (absolute value)
        """
        if position_type == 'long':
            if option_type == 'call':
                return premium  # Lose the premium
            else:  # put
                return premium
        else:  # short
            if option_type == 'call':
                return np.inf  # Unlimited loss
            else:  # put
                return K - premium


def compute_greeks_vectorized(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    option_type: str = 'call'
) -> Dict[str, np.ndarray]:
    """
    Compute all Greeks vectorized across multiple parameters.
    
    Parameters:
        S (np.ndarray): Stock prices (scalar or array)
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        sigma (np.ndarray): Volatilities (scalar or array)
        option_type (str): 'call' or 'put'
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of Greeks arrays
    
    Examples:
        >>> S = np.array([95, 100, 105])
        >>> sigma = np.array([0.15, 0.20, 0.25])
        >>> greeks = compute_greeks_vectorized(S, K=100, T=0.25, r=0.05, 
        ...                                   sigma=sigma, option_type='call')
        >>> print(greeks['delta'])  # Array of deltas
    """
    # Ensure arrays
    S = np.atleast_1d(S)
    sigma = np.atleast_1d(sigma)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Greeks
    greeks = {}
    
    if option_type == 'call':
        greeks['delta'] = normal_cdf(d1)
    else:
        greeks['delta'] = normal_cdf(d1) - 1
    
    greeks['gamma'] = normal_pdf(d1) / (S * sigma * np.sqrt(T))
    greeks['vega'] = S * normal_pdf(d1) * np.sqrt(T)
    
    if option_type == 'call':
        greeks['theta'] = -(S * normal_pdf(d1) * sigma / (2 * np.sqrt(T))) - \
                         r * K * np.exp(-r * T) * normal_cdf(d2)
    else:
        greeks['theta'] = -(S * normal_pdf(d1) * sigma / (2 * np.sqrt(T))) + \
                         r * K * np.exp(-r * T) * normal_cdf(-d2)
    
    if option_type == 'call':
        greeks['rho'] = K * T * np.exp(-r * T) * normal_cdf(d2)
    else:
        greeks['rho'] = -K * T * np.exp(-r * T) * normal_cdf(-d2)
    
    return greeks
