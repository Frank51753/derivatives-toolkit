"""
Monte Carlo Simulation for Derivatives Pricing

Uses stochastic simulation to price options and other derivatives.
Particularly useful for path-dependent options (Asian, barrier) and
multidimensional problems.

Theory:
    Price = E[exp(-rT) * Payoff(S_T)]
    ≈ (1/N) * Σ exp(-rT) * Payoff(S_i)
    
    where S_i are simulated final stock prices.

References:
    Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
    Boyle, P. (1977). Options: A Monte Carlo approach.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from ..models.geometric_brownian import GeometricBrownianMotion


class MonteCarloSimulator:
    """
    Basic Monte Carlo option pricer.
    
    Examples:
        >>> # Price European call
        >>> mc = MonteCarloSimulator(
        ...     S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        ...     n_paths=10000, n_steps=252
        ... )
        >>> call_price = mc.price_european_call()
        >>> put_price = mc.price_european_put()
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.
        
        Parameters:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            n_paths (int): Number of simulation paths
            n_steps (int): Number of time steps
            random_seed (Optional[int]): Random seed for reproducibility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.random_seed = random_seed
        
        # Generate paths
        gbm = GeometricBrownianMotion(S0, r, sigma, T, n_steps, n_paths, random_seed)
        self.paths = gbm.generate_paths_exact()
        self.discount_factor = np.exp(-r * T)
    
    def price_european_call(self) -> Tuple[float, float]:
        """
        Price European call using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Examples:
            >>> mc = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2)
            >>> price, se = mc.price_european_call()
            >>> print(f"Price: ${price:.2f} ± ${1.96*se:.2f}")  # 95% CI
        """
        payoffs = np.maximum(self.paths[-1, :] - self.K, 0)
        price = self.discount_factor * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * self.discount_factor
        
        return price, se
    
    def price_european_put(self) -> Tuple[float, float]:
        """
        Price European put using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        payoffs = np.maximum(self.K - self.paths[-1, :], 0)
        price = self.discount_factor * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * self.discount_factor
        
        return price, se
    
    def price_american_option(
        self,
        option_type: str = 'call',
        n_regression_terms: int = 2
    ) -> Tuple[float, float]:
        """
        Price American option using Longstaff-Schwartz algorithm.
        
        Parameters:
            option_type (str): 'call' or 'put'
            n_regression_terms (int): Polynomial degree for regression
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Notes:
            Uses least-squares Monte Carlo (LSM) for lower bound approximation.
        
        Examples:
            >>> mc = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2, n_paths=10000)
            >>> am_price, se = mc.price_american_option('call')
            >>> print(f"American call price: ${am_price:.2f}")
        """
        dt = self.T / self.n_steps
        discount = np.exp(-self.r * dt)
        
        # Initialize values at final time
        if option_type == 'call':
            values = np.maximum(self.paths[-1, :] - self.K, 0)
        else:
            values = np.maximum(self.K - self.paths[-1, :], 0)
        
        # Backward induction
        for t in range(self.n_steps - 1, 0, -1):
            # Discount future values
            values = values * discount
            
            # Intrinsic value
            spot = self.paths[t, :]
            if option_type == 'call':
                intrinsic = np.maximum(spot - self.K, 0)
            else:
                intrinsic = np.maximum(self.K - spot, 0)
            
            # Regression to find continuation value
            X = np.vstack([np.ones(self.n_paths), spot, spot**2])
            coeffs = np.linalg.lstsq(X.T, values, rcond=None)[0]
            continuation = X.T @ coeffs
            
            # Exercise decision
            exercise = intrinsic > continuation
            values[exercise] = intrinsic[exercise]
        
        # Discount to present
        values = values * np.exp(-self.r * dt)
        price = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_asian_option(
        self,
        option_type: str = 'call',
        averaging: str = 'arithmetic'
    ) -> Tuple[float, float]:
        """
        Price Asian option (averaging option).
        
        Parameters:
            option_type (str): 'call' or 'put'
            averaging (str): 'arithmetic' or 'geometric'
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Notes:
            Asian options have payoff based on average price:
            Call: max(S_avg - K, 0)
            Put: max(K - S_avg, 0)
        
        Examples:
            >>> mc = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2)
            >>> price, se = mc.price_asian_option('call', 'arithmetic')
        """
        if averaging == 'arithmetic':
            avg_prices = np.mean(self.paths, axis=0)
        elif averaging == 'geometric':
            avg_prices = np.exp(np.mean(np.log(self.paths), axis=0))
        else:
            raise ValueError(f"Unknown averaging method: {averaging}")
        
        if option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)
        
        price = self.discount_factor * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * self.discount_factor
        
        return price, se
    
    def price_barrier_option(
        self,
        option_type: str = 'call',
        barrier_type: str = 'knockout',
        barrier_level: float = None
    ) -> Tuple[float, float]:
        """
        Price barrier option (knock-out or knock-in).
        
        Parameters:
            option_type (str): 'call' or 'put'
            barrier_type (str): 'knockout' or 'knockin'
            barrier_level (float): Barrier level (default: 1.5*K for call, 0.5*K for put)
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Examples:
            >>> mc = MonteCarloSimulator(100, 100, 1.0, 0.05, 0.2)
            >>> price, se = mc.price_barrier_option('call', 'knockout', barrier_level=120)
        """
        if barrier_level is None:
            barrier_level = 1.5 * self.K if option_type == 'call' else 0.5 * self.K
        
        # Check if barrier is crossed
        if barrier_type == 'knockout':
            # Option dies if barrier is touched
            barrier_crossed = np.any(self.paths > barrier_level, axis=0) if option_type == 'call' \
                            else np.any(self.paths < barrier_level, axis=0)
            payoffs = np.where(barrier_crossed, 0, 
                             np.maximum(self.paths[-1, :] - self.K, 0) if option_type == 'call'
                             else np.maximum(self.K - self.paths[-1, :], 0))
        else:  # knockin
            # Option activated only if barrier is touched
            barrier_crossed = np.any(self.paths > barrier_level, axis=0) if option_type == 'call' \
                            else np.any(self.paths < barrier_level, axis=0)
            payoffs = np.where(barrier_crossed,
                             np.maximum(self.paths[-1, :] - self.K, 0) if option_type == 'call'
                             else np.maximum(self.K - self.paths[-1, :], 0), 0)
        
        price = self.discount_factor * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * self.discount_factor
        
        return price, se


class MonteCarloWithControlVariate(MonteCarloSimulator):
    """
    Monte Carlo with control variate variance reduction.
    
    Uses European call as control to reduce variance.
    
    Examples:
        >>> mc = MonteCarloWithControlVariate(100, 100, 1.0, 0.05, 0.2, n_paths=10000)
        >>> am_price, se = mc.price_american_option('call')
        >>> print(f"Reduced variance: {se:.4f}")
    """
    
    def price_american_option(
        self,
        option_type: str = 'call',
        n_regression_terms: int = 2
    ) -> Tuple[float, float]:
        """
        Price American option with variance reduction.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        # Get American price
        am_price, _ = super().price_american_option(option_type, n_regression_terms)
        
        # Get European price (analytical)
        from .black_scholes import BlackScholesAnalytic
        bs = BlackScholesAnalytic(self.S0, self.K, self.T, self.r, self.sigma)
        eu_price_analytical = bs.call_price() if option_type == 'call' else bs.put_price()
        
        # Get European price from same paths
        if option_type == 'call':
            eu_payoffs = np.maximum(self.paths[-1, :] - self.K, 0)
        else:
            eu_payoffs = np.maximum(self.K - self.paths[-1, :], 0)
        
        eu_price_mc = self.discount_factor * np.mean(eu_payoffs)
        
        # Control variate adjustment
        beta = 1.0  # Optimal would require computing covariance
        adjusted_price = am_price + beta * (eu_price_analytical - eu_price_mc)
        
        # Reduced standard error
        se = np.std(eu_payoffs, ddof=1) / np.sqrt(self.n_paths) * self.discount_factor
        
        return adjusted_price, se
