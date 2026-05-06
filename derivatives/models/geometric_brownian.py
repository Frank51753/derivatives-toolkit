"""
Geometric Brownian Motion (GBM) for Stock Price Modeling

This module implements Geometric Brownian Motion, the standard model for
stock price evolution in derivatives pricing.

Theory:
    GBM is defined by the SDE:
    dS/S = μ dt + σ dW
    
    where μ is drift (expected return) and σ is volatility.
    
    The solution is:
    S(t) = S(0) * exp((μ - σ²/2)t + σW(t))

References:
    Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
"""

import numpy as np
from typing import Optional, Tuple
from .ito_process import ItoProcess
from .brownian import BrownianMotion


class GeometricBrownianMotion(ItoProcess):
    """
    Geometric Brownian Motion for stock price modeling.
    
    Models stock prices as:
        dS/S = μ dt + σ dW
    
    Closed-form solution:
        S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
    
    This is the standard model for stock prices in the Black-Scholes framework.
    
    Attributes:
        S0 (float): Initial stock price
        mu (float): Drift coefficient (annualized expected return)
        sigma (float): Volatility coefficient (annualized)
        T (float): Time to maturity
        n_steps (int): Number of time steps
        n_paths (int): Number of sample paths
    
    Examples:
        >>> # Create GBM with realistic parameters
        >>> gbm = GeometricBrownianMotion(
        ...     S0=100,           # Initial stock price
        ...     mu=0.05,          # 5% expected annual return
        ...     sigma=0.2,        # 20% annual volatility
        ...     T=1.0,            # 1 year
        ...     n_steps=252,      # Daily steps
        ...     n_paths=10000     # 10k simulation paths
        ... )
        >>> 
        >>> # Generate paths
        >>> paths = gbm.generate_paths()
        >>> print(f"Shape: {paths.shape}")
        >>> print(f"Mean final price: {np.mean(paths[-1]):.2f}")
        >>> print(f"Expected final price: {100 * np.exp(0.05):.2f}")
    """
    
    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Geometric Brownian Motion.
        
        Parameters:
            S0 (float): Initial stock price (must be positive)
            mu (float): Drift coefficient (expected return, typically 0.02-0.15)
            sigma (float): Volatility coefficient (typically 0.05-0.5)
            T (float): Time to maturity in years (default 1.0)
            n_steps (int): Number of time steps (default 252 for daily)
            n_paths (int): Number of paths (default 1000)
            random_seed (Optional[int]): Seed for reproducibility
        
        Raises:
            ValueError: If S0 <= 0, sigma <= 0, T <= 0
        
        Notes:
            - Expected return μ should typically be 0.02 to 0.15 annually
            - Volatility σ should typically be 0.05 to 0.5 annually
            - For real stock data, μ and σ can be estimated from historical returns
        """
        if S0 <= 0:
            raise ValueError(f"Initial stock price S0 must be positive, got {S0}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")
        if T <= 0:
            raise ValueError(f"Time T must be positive, got {T}")
        
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        
        # Define drift and volatility functions for ItoProcess
        drift_func = lambda x, t: self.mu * x
        volatility_func = lambda x, t: self.sigma * x
        
        # Initialize parent class
        super().__init__(
            X0=S0,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            drift_func=drift_func,
            volatility_func=volatility_func,
            random_seed=random_seed
        )
    
    def generate_paths(self) -> np.ndarray:
        """
        Generate GBM paths using Euler-Maruyama scheme.
        
        Returns:
            np.ndarray: Array of shape (n_steps+1, n_paths) containing
                       stock price paths S(t).
        
        Mathematical Formula:
            S_{n+1} = S_n + μ S_n Δt + σ S_n ΔW_n
        
        Examples:
            >>> gbm = GeometricBrownianMotion(100, 0.05, 0.2, T=1.0, n_steps=252, n_paths=1000)
            >>> paths = gbm.generate_paths()
            >>> 
            >>> # Check properties
            >>> print(f"Initial price: {paths[0, 0]}")  # Should be 100
            >>> print(f"Mean final price: {np.mean(paths[-1])}")  # ≈ 100*exp(0.05)
            >>> print(f"Min price: {np.min(paths)}")  # Should be > 0
        """
        return super().solve()
    
    def generate_paths_exact(self) -> np.ndarray:
        """
        Generate GBM paths using exact closed-form solution.
        
        Uses the exact solution:
            S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        
        Returns:
            np.ndarray: Array of shape (n_steps+1, n_paths) of exact GBM paths
        
        Notes:
            This method is more accurate than Euler-Maruyama but requires
            generating full Brownian paths. Use when accuracy is critical.
            
        Mathematical Derivation:
            Solving dS/S = μdt + σdW by Itô lemma on ln(S):
            d(ln S) = (μ - σ²/2)dt + σdW
            
            Integrating:
            ln(S(t)/S(0)) = (μ - σ²/2)t + σW(t)
            
            Therefore:
            S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        
        Examples:
            >>> gbm = GeometricBrownianMotion(100, 0.05, 0.2, T=1.0, n_steps=252, n_paths=1000)
            >>> paths_exact = gbm.generate_paths_exact()
            >>> 
            >>> # Compare with Euler method
            >>> paths_euler = gbm.generate_paths()
            >>> print(f"Exact method mean: {np.mean(paths_exact[-1]):.2f}")
            >>> print(f"Euler method mean: {np.mean(paths_euler[-1]):.2f}")
        """
        # Generate Brownian motion
        bm = BrownianMotion(T=self.T, n_steps=self.n_steps, n_paths=self.n_paths,
                           random_seed=self.random_seed)
        W = bm.generate_paths()
        
        # Time grid
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        # Exact solution: S(t) = S0 * exp((μ - σ²/2)t + σW(t))
        exponent = (self.mu - 0.5 * self.sigma**2) * t[:, np.newaxis] + self.sigma * W
        paths = self.S0 * np.exp(exponent)
        
        return paths
    
    def generate_log_returns(self) -> np.ndarray:
        """
        Generate log-returns from GBM paths.
        
        Returns:
            np.ndarray: Array of shape (n_steps, n_paths) of log-returns
                       r_t = ln(S_t / S_{t-1})
        
        Notes:
            The log-returns of GBM are approximately:
            r_t ~ N((μ - σ²/2)dt, σ²dt)
        
        Examples:
            >>> gbm = GeometricBrownianMotion(100, 0.05, 0.2, T=1.0, n_steps=252, n_paths=10000)
            >>> returns = gbm.generate_log_returns()
            >>> 
            >>> # Check normality
            >>> mean_return = np.mean(returns)
            >>> expected_mean = (0.05 - 0.5*0.2**2) / 252
            >>> print(f"Mean return: {mean_return:.6f}, Expected: {expected_mean:.6f}")
        """
        paths = self.generate_paths_exact()
        returns = np.diff(np.log(paths), axis=0)
        return returns
    
    def get_implied_volatility(
        self,
        market_price: float,
        strike: float,
        maturity: float,
        risk_free_rate: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate implied volatility for a given market price.
        
        Uses Newton-Raphson method to find σ such that
        Black-Scholes(σ) = market_price.
        
        Parameters:
            market_price (float): Current market option price
            strike (float): Strike price
            maturity (float): Time to maturity
            risk_free_rate (float): Risk-free rate
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Implied volatility σ
        
        Raises:
            ValueError: If no solution found or market_price is invalid
        
        Examples:
            >>> gbm = GeometricBrownianMotion(100, 0.05, 0.2)
            >>> 
            >>> # European call option price
            >>> market_price = 10.5  # Some observed market price
            >>> strike = 100
            >>> maturity = 0.25
            >>> r = 0.05
            >>> 
            >>> iv = gbm.get_implied_volatility(market_price, strike, maturity, r, 'call')
            >>> print(f"Implied volatility: {iv:.4f}")
        """
        from ..utils.math import black_scholes, newton_raphson
        
        # Validate input
        if market_price <= 0:
            raise ValueError(f"Market price must be positive, got {market_price}")
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        
        def bs_price_diff(sigma):
            """Difference between BS price and market price"""
            bs_price = black_scholes(self.S0, strike, maturity, risk_free_rate, sigma, option_type)
            return bs_price - market_price
        
        def bs_vega(sigma):
            """Vega = sensitivity to volatility"""
            # ∂C/∂σ = S * n(d1) * √T
            from scipy.stats import norm
            d1 = (np.log(self.S0 / strike) + (risk_free_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
            vega = self.S0 * norm.pdf(d1) * np.sqrt(maturity)
            return vega if vega > 1e-10 else 1e-10
        
        # Initial guess: use current volatility
        sigma_0 = self.sigma if self.sigma > 0 else 0.2
        
        try:
            sigma_implied = newton_raphson(bs_price_diff, lambda s: bs_vega(s), sigma_0)
            return sigma_implied
        except ValueError as e:
            raise ValueError(f"Failed to find implied volatility: {str(e)}")
    
    @staticmethod
    def calibrate_from_returns(
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> Tuple[float, float]:
        """
        Calibrate GBM parameters from historical returns.
        
        Parameters:
            returns (np.ndarray): Array of log-returns
            annualize (bool): Whether to annualize (default True)
            periods_per_year (int): Trading periods per year (default 252 for daily)
        
        Returns:
            Tuple[float, float]: (mu, sigma) estimated drift and volatility
        
        Notes:
            μ = mean(returns)
            σ = std(returns)
            
            If annualize=True, scale by √(periods_per_year)
        
        Examples:
            >>> # From real stock data
            >>> prices = np.array([100, 102, 101, 103, ...])
            >>> returns = np.diff(np.log(prices))
            >>> mu, sigma = GeometricBrownianMotion.calibrate_from_returns(returns)
            >>> print(f"Estimated annual return: {mu:.2%}")
            >>> print(f"Estimated annual volatility: {sigma:.2%}")
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        if annualize:
            mu = mu * periods_per_year
            sigma = sigma * np.sqrt(periods_per_year)
        
        return mu, sigma
