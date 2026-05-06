"""
Exotic Options Pricing

Implements pricing for exotic/non-standard options including:
- Barrier options (knock-out, knock-in)
- Lookback options (min/max payoff)
- Asian options (average payoff)
- Basket options (multiple underlyings)
- Digital/Binary options (binary payoff)
- Ladder options (step payoff)

Theory:
    These options require Monte Carlo simulation or PDE methods
    as closed-form solutions often don't exist.

References:
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
    Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, List
from scipy import optimize, stats
from abc import ABC, abstractmethod


class ExoticOption(ABC):
    """Base class for exotic options."""
    
    def __init__(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ):
        """
        Initialize exotic option.
        
        Parameters:
            S0 (float): Initial stock price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
        """
        if S0 <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S0, T, sigma must be positive")
        
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
    
    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate payoff for each path."""
        pass
    
    @abstractmethod
    def price(self) -> Tuple[float, float]:
        """Price the option. Returns (price, std_error)."""
        pass


class BarrierOption(ExoticOption):
    """
    Barrier Option: Option dies (knock-out) or comes alive (knock-in)
    if barrier is reached.
    
    Types:
        - Knock-out: Option expires worthless if barrier touched
        - Knock-in: Option only becomes active if barrier touched
    
    Barriers:
        - Up-and-out: Barrier above current price
        - Down-and-out: Barrier below current price
        - Up-and-in: Barrier above current price
        - Down-and-in: Barrier below current price
    
    Examples:
        >>> barrier = BarrierOption(
        ...     S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        ...     barrier=120, barrier_type='up-out'
        ... )
        >>> price, se = barrier.price()
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        barrier_type: str = 'up-out',
        option_type: str = 'call',
        rebate: float = 0.0,
        n_paths: int = 10000,
        n_steps: int = 252
    ):
        """
        Initialize barrier option.
        
        Parameters:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            barrier (float): Barrier level
            barrier_type (str): 'up-out', 'down-out', 'up-in', 'down-in'
            option_type (str): 'call' or 'put'
            rebate (float): Payment if barrier is hit (for knock-out)
            n_paths (int): Number of simulation paths
            n_steps (int): Number of time steps
        
        Raises:
            ValueError: If barrier_type is invalid
        """
        super().__init__(S0, T, r, sigma, option_type)
        
        if barrier_type not in ['up-out', 'down-out', 'up-in', 'down-in']:
            raise ValueError(f"Invalid barrier_type: {barrier_type}")
        
        self.K = K
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.rebate = rebate
        self.n_paths = n_paths
        self.n_steps = n_steps
    
    def _check_barrier(self, paths: np.ndarray) -> np.ndarray:
        """
        Check if barrier is crossed.
        
        Returns:
            np.ndarray: Boolean array indicating barrier crossing
        """
        if 'up' in self.barrier_type:
            return np.any(paths > self.barrier, axis=0)
        else:  # down
            return np.any(paths < self.barrier, axis=0)
    
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate payoff for barrier option.
        
        Parameters:
            paths (np.ndarray): Stock price paths (time x paths)
        
        Returns:
            np.ndarray: Payoff for each path
        """
        barrier_crossed = self._check_barrier(paths)
        
        # Calculate vanilla payoff
        final_price = paths[-1, :]
        if self.option_type == 'call':
            vanilla_payoff = np.maximum(final_price - self.K, 0)
        else:
            vanilla_payoff = np.maximum(self.K - final_price, 0)
        
        # Apply barrier logic
        if 'out' in self.barrier_type:
            # Knock-out: payoff only if barrier not crossed
            payoff = np.where(barrier_crossed, self.rebate, vanilla_payoff)
        else:  # knock-in
            # Knock-in: payoff only if barrier crossed
            payoff = np.where(barrier_crossed, vanilla_payoff, 0)
        
        return payoff
    
    def price(self) -> Tuple[float, float]:
        """
        Price barrier option using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Examples:
            >>> barrier = BarrierOption(100, 100, 1.0, 0.05, 0.2, 120)
            >>> price, se = barrier.price()
            >>> print(f"Price: ${price:.2f} ± ${1.96*se:.2f}")
        """
        from ..models.geometric_brownian import GeometricBrownianMotion
        
        gbm = GeometricBrownianMotion(
            self.S0, self.r, self.sigma, self.T,
            self.n_steps, self.n_paths
        )
        paths = gbm.generate_paths_exact()
        
        payoffs = self.payoff(paths)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * np.exp(-self.r * self.T)
        
        return price, se


class LookbackOption(ExoticOption):
    """
    Lookback Option: Payoff depends on minimum or maximum price
    reached during the option's life.
    
    Types:
        - Fixed strike lookback: Payoff = max(S_max - K, 0) or max(K - S_min, 0)
        - Floating strike lookback: Payoff = S_T - S_min or S_max - S_T
    
    These options are always in-the-money if floating strike.
    
    Examples:
        >>> lookback = LookbackOption(
        ...     S0=100, T=1.0, r=0.05, sigma=0.2,
        ...     lookback_type='floating'
        ... )
        >>> price, se = lookback.price()
    """
    
    def __init__(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        K: Optional[float] = None,
        option_type: str = 'call',
        lookback_type: str = 'fixed',
        n_paths: int = 10000,
        n_steps: int = 252
    ):
        """
        Initialize lookback option.
        
        Parameters:
            S0 (float): Initial stock price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            K (Optional[float]): Strike price (for fixed lookback)
            option_type (str): 'call' or 'put'
            lookback_type (str): 'fixed' or 'floating'
            n_paths (int): Number of simulation paths
            n_steps (int): Number of time steps
        """
        super().__init__(S0, T, r, sigma, option_type)
        
        if lookback_type not in ['fixed', 'floating']:
            raise ValueError("lookback_type must be 'fixed' or 'floating'")
        
        self.K = K if lookback_type == 'fixed' else S0
        self.lookback_type = lookback_type
        self.n_paths = n_paths
        self.n_steps = n_steps
    
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate lookback payoff.
        
        Parameters:
            paths (np.ndarray): Stock price paths
        
        Returns:
            np.ndarray: Payoff for each path
        """
        S_min = np.min(paths, axis=0)
        S_max = np.max(paths, axis=0)
        S_T = paths[-1, :]
        
        if self.lookback_type == 'fixed':
            if self.option_type == 'call':
                payoff = np.maximum(S_max - self.K, 0)
            else:
                payoff = np.maximum(self.K - S_min, 0)
        else:  # floating
            if self.option_type == 'call':
                payoff = S_T - S_min
            else:
                payoff = S_max - S_T
        
        return payoff
    
    def price(self) -> Tuple[float, float]:
        """
        Price lookback option using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        from ..models.geometric_brownian import GeometricBrownianMotion
        
        gbm = GeometricBrownianMotion(
            self.S0, self.r, self.sigma, self.T,
            self.n_steps, self.n_paths
        )
        paths = gbm.generate_paths_exact()
        
        payoffs = self.payoff(paths)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * np.exp(-self.r * self.T)
        
        return price, se


class BasketOption(ExoticOption):
    """
    Basket Option: Payoff depends on a weighted average of
    multiple underlying assets.
    
    Common basket examples:
        - Index options (S&P 500, etc.)
        - Currency basket options
        - Commodity basket options
    
    Examples:
        >>> basket = BasketOption(
        ...     S0=[100, 50, 150],
        ...     weights=[0.5, 0.3, 0.2],
        ...     K=100,
        ...     T=1.0,
        ...     r=0.05,
        ...     sigma=[0.2, 0.25, 0.15],
        ...     correlation=[[1, 0.5, -0.3],
        ...                  [0.5, 1, 0.2],
        ...                  [-0.3, 0.2, 1]]
        ... )
        >>> price, se = basket.price()
    """
    
    def __init__(
        self,
        S0: np.ndarray,
        weights: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        option_type: str = 'call',
        n_paths: int = 10000,
        n_steps: int = 252
    ):
        """
        Initialize basket option.
        
        Parameters:
            S0 (np.ndarray): Initial prices of n assets
            weights (np.ndarray): Weights of assets (sum = 1)
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (np.ndarray): Volatilities of n assets
            correlation (Optional[np.ndarray]): n x n correlation matrix
            option_type (str): 'call' or 'put'
            n_paths (int): Number of paths
            n_steps (int): Number of time steps
        """
        super().__init__(S0[0], T, r, sigma[0], option_type)
        
        self.S0 = np.asarray(S0, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.K = K
        self.sigma = np.asarray(sigma, dtype=float)
        self.n_assets = len(S0)
        
        # Validate
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1")
        
        if len(self.weights) != self.n_assets:
            raise ValueError("Weights must match number of assets")
        
        # Set correlation
        if correlation is None:
            # Identity matrix (no correlation)
            self.correlation = np.eye(self.n_assets)
        else:
            self.correlation = np.asarray(correlation, dtype=float)
            if self.correlation.shape != (self.n_assets, self.n_assets):
                raise ValueError("Correlation matrix shape mismatch")
        
        self.n_paths = n_paths
        self.n_steps = n_steps
    
    def payoff(self, basket_prices: np.ndarray) -> np.ndarray:
        """
        Calculate basket option payoff.
        
        Parameters:
            basket_prices (np.ndarray): Basket prices (n_paths,)
        
        Returns:
            np.ndarray: Payoff for each path
        """
        if self.option_type == 'call':
            return np.maximum(basket_prices - self.K, 0)
        else:
            return np.maximum(self.K - basket_prices, 0)
    
    def price(self) -> Tuple[float, float]:
        """
        Price basket option using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Notes:
            Simulates n correlated asset prices and averages them.
        """
        dt = self.T / self.n_steps
        
        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(self.correlation)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite")
        
        # Simulate asset paths
        asset_paths = np.zeros((self.n_steps + 1, self.n_paths, self.n_assets))
        asset_paths[0, :, :] = self.S0
        
        for t in range(self.n_steps):
            # Independent standard normals
            Z = np.random.standard_normal((self.n_paths, self.n_assets))
            
            # Apply correlation
            Z_corr = Z @ L.T
            
            for i in range(self.n_assets):
                drift = (self.r - 0.5 * self.sigma[i]**2) * dt
                diffusion = self.sigma[i] * np.sqrt(dt) * Z_corr[:, i]
                
                asset_paths[t+1, :, i] = asset_paths[t, :, i] * np.exp(drift + diffusion)
        
        # Calculate basket values
        final_prices = asset_paths[-1, :, :]
        basket_prices = final_prices @ self.weights
        
        payoffs = self.payoff(basket_prices)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * np.exp(-self.r * self.T)
        
        return price, se


class DigitalOption(ExoticOption):
    """
    Digital (Binary) Option: Pays fixed amount if in-the-money,
    zero otherwise.
    
    Payoff:
        - Call: Pays K if S_T > strike, 0 otherwise
        - Put: Pays K if S_T < strike, 0 otherwise
    
    Also called "binary options" or "cash-or-nothing options".
    
    Examples:
        >>> digital = DigitalOption(
        ...     S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        ...     payoff_amount=100
        ... )
        >>> price, se = digital.price()
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        payoff_amount: float = 1.0,
        option_type: str = 'call'
    ):
        """
        Initialize digital option.
        
        Parameters:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            payoff_amount (float): Amount paid if in-the-money
            option_type (str): 'call' or 'put'
        """
        super().__init__(S0, T, r, sigma, option_type)
        self.K = K
        self.payoff_amount = payoff_amount
    
    def payoff(self, final_price: float) -> float:
        """
        Calculate digital payoff.
        
        Parameters:
            final_price (float): Final stock price
        
        Returns:
            float: Payoff amount
        """
        if self.option_type == 'call':
            return self.payoff_amount if final_price > self.K else 0
        else:
            return self.payoff_amount if final_price < self.K else 0
    
    def price(self) -> Tuple[float, float]:
        """
        Price digital option using risk-neutral valuation.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        
        Formula:
            Digital call = e^(-rT) * Q(S_T > K)
            where Q is risk-neutral probability
        """
        from scipy.stats import norm
        
        # Use Black-Scholes framework
        d2 = (np.log(self.S0 / self.K) + 
              (self.r - 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        if self.option_type == 'call':
            prob = norm.cdf(d2)
        else:
            prob = norm.cdf(-d2)
        
        price = np.exp(-self.r * self.T) * self.payoff_amount * prob
        
        # Approximate standard error
        se = 0  # Analytical formula has no sampling error
        
        return price, se


class AsianOption(ExoticOption):
    """
    Asian Option: Payoff depends on average price over option's life.
    
    Advantages over standard options:
        - Cheaper due to averaging effect
        - Less sensitive to price spikes
        - Lower volatility implied
    
    Examples:
        >>> asian = AsianOption(
        ...     S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        ...     averaging='arithmetic'
        ... )
        >>> price, se = asian.price()
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        averaging: str = 'arithmetic',
        n_paths: int = 10000,
        n_steps: int = 252
    ):
        """
        Initialize Asian option.
        
        Parameters:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            averaging (str): 'arithmetic' or 'geometric'
            n_paths (int): Number of paths
            n_steps (int): Number of time steps
        """
        super().__init__(S0, T, r, sigma, option_type)
        
        if averaging not in ['arithmetic', 'geometric']:
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")
        
        self.K = K
        self.averaging = averaging
        self.n_paths = n_paths
        self.n_steps = n_steps
    
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate Asian option payoff.
        
        Parameters:
            paths (np.ndarray): Stock price paths
        
        Returns:
            np.ndarray: Payoff for each path
        """
        if self.averaging == 'arithmetic':
            avg_price = np.mean(paths, axis=0)
        else:  # geometric
            avg_price = np.exp(np.mean(np.log(paths), axis=0))
        
        if self.option_type == 'call':
            return np.maximum(avg_price - self.K, 0)
        else:
            return np.maximum(self.K - avg_price, 0)
    
    def price(self) -> Tuple[float, float]:
        """
        Price Asian option using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        from ..models.geometric_brownian import GeometricBrownianMotion
        
        gbm = GeometricBrownianMotion(
            self.S0, self.r, self.sigma, self.T,
            self.n_steps, self.n_paths
        )
        paths = gbm.generate_paths_exact()
        
        payoffs = self.payoff(paths)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths) * np.exp(-self.r * self.T)
        
        return price, se
    
    def price_geometric_closed_form(self) -> float:
        """
        Price geometric Asian option using closed-form solution.
        
        Returns:
            float: Option price
        
        Notes:
            Only available for geometric averaging.
            Based on Black-Scholes formula with adjusted parameters.
        """
        if self.averaging != 'geometric':
            raise ValueError("Closed-form only available for geometric averaging")
        
        # Adjusted parameters for geometric averaging
        sigma_a = self.sigma / np.sqrt(3)
        b = (self.r - 0.5 * self.sigma**2) / (2 * self.sigma**2)
        
        # Effective drift and volatility
        mu = self.r - b * self.sigma**2
        
        # Use Black-Scholes with adjusted parameters
        from .black_scholes import BlackScholesAnalytic
        
        bs = BlackScholesAnalytic(
            self.S0, self.K, self.T, mu, sigma_a
        )
        
        if self.option_type == 'call':
            return bs.call_price()
        else:
            return bs.put_price()
