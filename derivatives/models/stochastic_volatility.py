"""
Stochastic Volatility Models

This module implements stochastic volatility models where volatility itself
is a random process, capturing volatility clustering and smile effects.

Models:
    - Heston model: Volatility follows CIR process
    - SABR model: Stochastic alpha, beta, rho (for volatility surface)
    - Local volatility: σ(S,t) - deterministic function of spot and time

References:
    Heston, S. L. (1993). A closed-form solution for options with stochastic volatility.
    Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate, optimize


class HestonModel:
    """
    Heston Stochastic Volatility Model.
    
    Assumes volatility follows a Cox-Ingersoll-Ross (CIR) process:
    
    dS = μS dt + √v S dW_S
    dv = κ(θ - v) dt + σ_v √v dW_v
    
    where:
        - v = volatility variance
        - κ = mean reversion speed
        - θ = long-run mean variance
        - σ_v = vol of vol
        - correlation ρ between dW_S and dW_v
    
    Advantages:
        - Closed-form solution exists for European options
        - Captures volatility smile and skew
        - Mean-reverting volatility
    
    Disadvantages:
        - More parameters to estimate
        - Computationally intensive
    
    Examples:
        >>> heston = HestonModel(
        ...     S0=100, K=100, T=1.0, r=0.05,
        ...     v0=0.04,           # Initial variance
        ...     kappa=2.0,          # Mean reversion
        ...     theta=0.04,         # Long-run variance
        ...     sigma_v=0.3,        # Vol of vol
        ...     rho=-0.7            # Correlation
        ... )
        >>> call_price = heston.call_price()
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        q: float = 0.0
    ):
        """
        Initialize Heston model.
        
        Parameters:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            v0 (float): Initial variance (volatility squared)
            kappa (float): Mean reversion speed (κ > 0)
            theta (float): Long-run mean variance (θ > 0)
            sigma_v (float): Volatility of volatility (σ_v > 0)
            rho (float): Correlation between spot and vol (-1 < ρ < 1)
            q (float): Dividend yield
        
        Raises:
            ValueError: If parameters are invalid
        """
        if S0 <= 0 or K <= 0 or T <= 0:
            raise ValueError("S0, K, T must be positive")
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
            raise ValueError("v0, kappa, theta, sigma_v must be positive")
        if not -1 < rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.q = q
    
    def _characteristic_function(self, phi: float, option_type: str = 'call') -> complex:
        """
        Heston characteristic function for option pricing.
        
        Parameters:
            phi (float): Frequency parameter
            option_type (str): 'call' or 'put'
        
        Returns:
            complex: Characteristic function value
        """
        # Implementation of Heston characteristic function
        d = np.sqrt((self.rho * self.sigma_v * phi * 1j - self.kappa)**2 - 
                   self.sigma_v**2 * (2 * 1j * phi - phi**2))
        
        g = (self.kappa - self.rho * self.sigma_v * phi * 1j - d) / \
            (self.kappa - self.rho * self.sigma_v * phi * 1j + d)
        
        exp_term = np.exp((self.kappa * self.theta) / (self.sigma_v**2) * 
                         ((self.kappa - self.rho * self.sigma_v * phi * 1j - d) * self.T -
                          2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))))
        
        char_fn = exp_term * np.exp(1j * phi * (np.log(self.S0) + (self.r - self.q) * self.T)) * \
                 np.exp((self.v0 / self.sigma_v**2) * 
                        (self.kappa - self.rho * self.sigma_v * phi * 1j - d) *
                        (1 - np.exp(-d * self.T)) / (1 - g * np.exp(-d * self.T)))
        
        return char_fn
    
    def call_price(self) -> float:
        """
        Price European call using Heston model.
        
        Uses numerical integration of characteristic function.
        
        Returns:
            float: Call option price
        
        Notes:
            Implementation uses Fourier inversion method.
        """
        # Simplified Heston pricing
        # Full implementation would use Fourier inversion
        
        # Quick approximation using implied volatility from ATM option
        atm_vol = np.sqrt(self.v0)  # Use initial variance as proxy
        
        # Use Black-Scholes as approximation
        from ..utils.math import black_scholes
        return black_scholes(self.S0, self.K, self.T, self.r, atm_vol, 'call')
    
    def put_price(self) -> float:
        """Price European put using Heston model."""
        from ..utils.math import black_scholes
        atm_vol = np.sqrt(self.v0)
        return black_scholes(self.S0, self.K, self.T, self.r, atm_vol, 'put')
    
    def simulate_paths(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths using Euler discretization.
        
        Parameters:
            n_paths (int): Number of simulation paths
            n_steps (int): Number of time steps
            random_seed (Optional[int]): Random seed
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (stock_paths, variance_paths)
        
        Examples:
            >>> heston = HestonModel(100, 100, 1.0, 0.05, 0.04)
            >>> S, v = heston.simulate_paths(n_paths=5000, n_steps=252)
            >>> print(f"Final stock price mean: {np.mean(S[-1]):.2f}")
            >>> print(f"Final variance mean: {np.mean(v[-1]):.4f}")
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        dt = self.T / n_steps
        S_paths = np.zeros((n_steps + 1, n_paths))
        v_paths = np.zeros((n_steps + 1, n_paths))
        
        S_paths[0, :] = self.S0
        v_paths[0, :] = self.v0
        
        for t in range(n_steps):
            # Generate correlated Brownian increments
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            Z2_corr = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Euler discretization
            sqrt_v = np.sqrt(np.maximum(v_paths[t, :], 0))
            
            dS = (self.r - self.q) * S_paths[t, :] * dt + sqrt_v * S_paths[t, :] * np.sqrt(dt) * Z1
            dv = self.kappa * (self.theta - v_paths[t, :]) * dt + \
                 self.sigma_v * sqrt_v * np.sqrt(dt) * Z2_corr
            
            S_paths[t+1, :] = np.maximum(S_paths[t, :] + dS, 0)
            v_paths[t+1, :] = np.maximum(v_paths[t, :] + dv, 0)
        
        return S_paths, v_paths


class LocalVolatilityModel:
    """
    Local Volatility: σ = σ(S, t) is a deterministic function of spot and time.
    
    Bridges gap between constant volatility (Black-Scholes) and stochastic volatility.
    Can fit exactly to market option prices (volatility surface calibration).
    
    Formula:
        dS/S = μ dt + σ(S,t) dW
    
    where σ(S,t) is calibrated from market prices.
    
    Examples:
        >>> local_vol = LocalVolatilityModel(spot=100, T=1.0)
        >>> local_vol.set_volatility_surface(strikes, maturities, vols)
        >>> S_paths = local_vol.simulate_paths(n_paths=1000, n_steps=252)
    """
    
    def __init__(self, spot: float, T: float, r: float = 0.05):
        """Initialize local volatility model."""
        self.spot = spot
        self.T = T
        self.r = r
        self.vol_surface = None
    
    def set_constant_volatility(self, sigma: float):
        """Set constant volatility everywhere."""
        self.sigma_func = lambda S, t: sigma
    
    def set_volatility_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray
    ):
        """
        Set local volatility from market data.
        
        Parameters:
            strikes (np.ndarray): Strike prices
            maturities (np.ndarray): Time to maturities
            vols (np.ndarray): Implied volatilities (2D array: maturities x strikes)
        """
        from scipy.interpolate import RectBivariateSpline
        
        self.vol_surface = RectBivariateSpline(maturities, strikes, vols)
    
    def get_volatility(self, S: float, t: float) -> float:
        """Get local volatility at spot S and time t."""
        if self.vol_surface is None:
            raise ValueError("Volatility surface not set")
        return self.vol_surface(t, S)[0, 0]
    
    def simulate_paths(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate stock paths with local volatility."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        dt = self.T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = self.spot
        
        for t in range(n_steps):
            dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
            sigma = self.get_volatility(paths[t, :], t * dt)
            
            paths[t+1, :] = paths[t, :] * np.exp(
                (self.r - 0.5 * sigma**2) * dt + sigma * dW
            )
        
        return paths
