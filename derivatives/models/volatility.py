"""
Volatility Modeling

This module implements various volatility estimation and forecasting models
including historical, EWMA, and GARCH models.

Theory:
    Volatility is the standard deviation of returns over time.
    Different models capture different aspects:
    - Historical: Past realized volatility
    - EWMA: Recent observations more important
    - GARCH: Volatility clustering and mean reversion

References:
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
    Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import optimize


class VolatilityEstimator:
    """
    Base class for volatility estimation methods.
    
    Provides interface for different volatility models.
    """
    
    def __init__(self, returns: np.ndarray, periods_per_year: int = 252):
        """
        Initialize volatility estimator.
        
        Parameters:
            returns (np.ndarray): Array of log-returns
            periods_per_year (int): Trading periods per year (default 252 for daily)
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns to estimate volatility")
        
        self.returns = np.asarray(returns, dtype=float)
        self.periods_per_year = periods_per_year
        self.n = len(returns)
    
    def estimate(self) -> float:
        """Estimate volatility. Must be implemented by subclasses."""
        raise NotImplementedError


class HistoricalVolatility(VolatilityEstimator):
    """
    Historical volatility: Standard deviation of returns.
    
    The simplest volatility measure - just the sample standard deviation
    of historical returns.
    
    Formula:
        σ_hist = sqrt(Σ(r_t - r̄)² / (n-1)) * sqrt(periods_per_year)
    
    Advantages:
        - Simple and intuitive
        - Easy to compute
    
    Disadvantages:
        - Past volatility may not predict future volatility
        - Equal weight to all observations
    
    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> vol = HistoricalVolatility(returns)
        >>> annual_vol = vol.estimate()
        >>> print(f"Annual volatility: {annual_vol:.4f}")
    """
    
    def estimate(self) -> float:
        """
        Estimate annualized historical volatility.
        
        Returns:
            float: Annualized volatility
        """
        return np.std(self.returns, ddof=1) * np.sqrt(self.periods_per_year)
    
    def rolling_volatility(self, window: int = 20) -> np.ndarray:
        """
        Calculate rolling window volatility.
        
        Parameters:
            window (int): Window size in periods
        
        Returns:
            np.ndarray: Array of rolling volatilities
        
        Examples:
            >>> rolling_vols = vol.rolling_volatility(window=30)
            >>> print(rolling_vols[-1])  # Most recent volatility
        """
        if window > self.n:
            raise ValueError(f"Window size {window} exceeds data length {self.n}")
        
        rolling_vols = np.zeros(self.n - window + 1)
        for i in range(self.n - window + 1):
            rolling_vols[i] = np.std(self.returns[i:i+window], ddof=1) * np.sqrt(self.periods_per_year)
        
        return rolling_vols


class EWMAVolatility(VolatilityEstimator):
    """
    Exponentially Weighted Moving Average (EWMA) volatility.
    
    Recent observations have more weight than older ones.
    
    Formula:
        σ_t² = λ*σ_{t-1}² + (1-λ)*r_t²
    
    where λ is the decay factor (typically 0.94 for daily data).
    
    Advantages:
        - Gives more weight to recent data
        - Responds quickly to volatility changes
        - Used by RiskMetrics
    
    Disadvantages:
        - Arbitrary choice of λ
        - Doesn't model volatility clusters
    
    Examples:
        >>> returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> ewma_vol = EWMAVolatility(returns, lambda_=0.94)
        >>> vol_forecast = ewma_vol.estimate()
        >>> print(f"EWMA volatility: {vol_forecast:.4f}")
    """
    
    def __init__(
        self, 
        returns: np.ndarray, 
        lambda_: float = 0.94,
        periods_per_year: int = 252
    ):
        """
        Initialize EWMA volatility estimator.
        
        Parameters:
            returns (np.ndarray): Array of log-returns
            lambda_ (float): Decay factor (default 0.94). Higher value = more weight to history
            periods_per_year (int): Trading periods per year
        
        Raises:
            ValueError: If lambda_ not in (0, 1)
        """
        super().__init__(returns, periods_per_year)
        
        if not 0 < lambda_ < 1:
            raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
        
        self.lambda_ = lambda_
        self.sigma_t = None
    
    def estimate(self) -> float:
        """
        Estimate EWMA volatility.
        
        Returns:
            float: Annualized EWMA volatility (most recent)
        
        Notes:
            Computes the full EWMA series and returns the last value.
        """
        sigma2_t = np.zeros(self.n)
        
        # Initialize with historical volatility
        sigma2_t[0] = np.var(self.returns, ddof=1)
        
        # Recursive EWMA formula
        for t in range(1, self.n):
            sigma2_t[t] = self.lambda_ * sigma2_t[t-1] + (1 - self.lambda_) * self.returns[t]**2
        
        self.sigma_t = np.sqrt(sigma2_t)
        return self.sigma_t[-1] * np.sqrt(self.periods_per_year)
    
    def get_series(self) -> np.ndarray:
        """
        Get the full EWMA volatility series.
        
        Returns:
            np.ndarray: Time series of EWMA volatilities
        
        Examples:
            >>> ewma = EWMAVolatility(returns)
            >>> vol_series = ewma.get_series()
            >>> print(vol_series[-10:])  # Last 10 values
        """
        if self.sigma_t is None:
            self.estimate()
        return self.sigma_t * np.sqrt(self.periods_per_year)


class GARCHVolatility(VolatilityEstimator):
    """
    GARCH(1,1) Model: Generalized Autoregressive Conditional Heteroskedasticity.
    
    Models volatility clustering: periods of high volatility followed by
    periods of low volatility.
    
    Formula:
        σ_t² = ω + α*r_{t-1}² + β*σ_{t-1}²
    
    where:
        - ω is the constant (unconditional volatility)
        - α is the ARCH coefficient (react to shocks)
        - β is the GARCH coefficient (persistence)
    
    Advantages:
        - Captures volatility clustering
        - More realistic than EWMA
        - Good for forecasting
    
    Disadvantages:
        - Requires parameter estimation (MLE)
        - More complex
    
    Examples:
        >>> returns = np.random.normal(0, 0.01, 252)
        >>> garch = GARCHVolatility(returns)
        >>> params = garch.fit()
        >>> vol = garch.estimate()
        >>> forecast = garch.forecast_volatility(steps=10)
    """
    
    def __init__(self, returns: np.ndarray, periods_per_year: int = 252):
        """Initialize GARCH volatility estimator."""
        super().__init__(returns, periods_per_year)
        self.params = None
        self.sigma2_t = None
    
    def _likelihood(self, params: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for MLE estimation.
        
        Parameters:
            params (np.ndarray): [omega, alpha, beta]
        
        Returns:
            float: Negative log-likelihood
        """
        omega, alpha, beta = params
        
        # Constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        sigma2 = np.zeros(self.n)
        sigma2[0] = np.var(self.returns)
        
        loglik = 0
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * sigma2[t-1]
            if sigma2[t] <= 0:
                return 1e10
            loglik += np.log(sigma2[t]) + self.returns[t]**2 / sigma2[t]
        
        return loglik
    
    def fit(self) -> Tuple[float, float, float]:
        """
        Fit GARCH(1,1) parameters using Maximum Likelihood Estimation.
        
        Returns:
            Tuple[float, float, float]: (omega, alpha, beta)
        
        Notes:
            Uses numerical optimization. Initial guess is based on sample.
        
        Examples:
            >>> garch = GARCHVolatility(returns)
            >>> omega, alpha, beta = garch.fit()
            >>> print(f"Persistence: {alpha + beta:.4f}")  # Should be < 1
        """
        # Initial guess
        sample_var = np.var(self.returns)
        x0 = np.array([0.0001 * sample_var, 0.1, 0.8])
        
        # Optimize
        result = optimize.minimize(
            self._likelihood,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise RuntimeError("GARCH fitting failed to converge")
        
        self.params = result.x
        return tuple(self.params)
    
    def estimate(self) -> float:
        """
        Estimate current GARCH volatility.
        
        Returns:
            float: Annualized volatility
        """
        if self.params is None:
            self.fit()
        
        omega, alpha, beta = self.params
        sigma2 = np.zeros(self.n)
        sigma2[0] = np.var(self.returns)
        
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * sigma2[t-1]
        
        self.sigma2_t = sigma2
        return np.sqrt(sigma2[-1]) * np.sqrt(self.periods_per_year)
    
    def forecast_volatility(self, steps: int = 10) -> np.ndarray:
        """
        Forecast future volatility using fitted GARCH model.
        
        Parameters:
            steps (int): Number of periods to forecast
        
        Returns:
            np.ndarray: Forecast volatilities
        
        Notes:
            GARCH volatility reverts to long-run mean.
        
        Examples:
            >>> garch = GARCHVolatility(returns)
            >>> garch.fit()
            >>> forecast = garch.forecast_volatility(steps=20)
            >>> print(forecast)
        """
        if self.params is None:
            self.fit()
        if self.sigma2_t is None:
            self.estimate()
        
        omega, alpha, beta = self.params
        long_run_var = omega / (1 - alpha - beta)
        
        forecast = np.zeros(steps)
        current_sigma2 = self.sigma2_t[-1]
        
        for i in range(steps):
            current_sigma2 = omega + (alpha + beta) * (current_sigma2 - long_run_var) + long_run_var
            forecast[i] = np.sqrt(current_sigma2) * np.sqrt(self.periods_per_year)
        
        return forecast
    
    def get_series(self) -> np.ndarray:
        """Get full GARCH volatility series."""
        if self.sigma2_t is None:
            self.estimate()
        return np.sqrt(self.sigma2_t) * np.sqrt(self.periods_per_year)


class ParkinsonVolatility(VolatilityEstimator):
    """
    Parkinson Volatility: Uses high and low prices, more efficient than close-only.
    
    Formula:
        σ_parkinson = sqrt(ln(H/L)² / (4*ln(2))) * sqrt(periods_per_year)
    
    Advantages:
        - Uses intraday price range
        - More efficient than close-only volatility
        - Lower standard error
    
    Disadvantages:
        - Requires high/low prices (not just returns)
        - Assumes gap-free trading
    
    Examples:
        >>> high_prices = np.array([101, 102, 103, 104])
        >>> low_prices = np.array([99, 100, 101, 102])
        >>> parkinson = ParkinsonVolatility.from_prices(high_prices, low_prices)
        >>> vol = parkinson.estimate()
    """
    
    def __init__(self, high_low_ratios: np.ndarray, periods_per_year: int = 252):
        """
        Initialize Parkinson volatility.
        
        Parameters:
            high_low_ratios (np.ndarray): Array of ln(H/L) ratios
            periods_per_year (int): Trading periods per year
        """
        super().__init__(high_low_ratios, periods_per_year)
    
    @staticmethod
    def from_prices(high_prices: np.ndarray, low_prices: np.ndarray, 
                   periods_per_year: int = 252) -> 'ParkinsonVolatility':
        """
        Create Parkinson estimator from high and low prices.
        
        Parameters:
            high_prices (np.ndarray): Array of daily high prices
            low_prices (np.ndarray): Array of daily low prices
            periods_per_year (int): Trading periods per year
        
        Returns:
            ParkinsonVolatility: Initialized estimator
        """
        high_prices = np.asarray(high_prices, dtype=float)
        low_prices = np.asarray(low_prices, dtype=float)
        
        if len(high_prices) != len(low_prices):
            raise ValueError("high_prices and low_prices must have same length")
        
        ratios = np.log(high_prices / low_prices)
        return ParkinsonVolatility(ratios, periods_per_year)
    
    def estimate(self) -> float:
        """
        Estimate Parkinson volatility.
        
        Returns:
            float: Annualized volatility
        """
        return np.sqrt(np.mean(self.returns**2) / (4 * np.log(2))) * np.sqrt(self.periods_per_year)


def compare_volatility_models(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> dict:
    """
    Compare different volatility estimation methods.
    
    Parameters:
        returns (np.ndarray): Array of log-returns
        periods_per_year (int): Trading periods per year
    
    Returns:
        dict: Volatility estimates from different models
    
    Examples:
        >>> returns = np.random.normal(0, 0.01, 252)
        >>> comparison = compare_volatility_models(returns)
        >>> 
        >>> for model, vol in comparison.items():
        ...     print(f"{model}: {vol:.4f}")
    """
    results = {}
    
    # Historical volatility
    hist_vol = HistoricalVolatility(returns, periods_per_year)
    results['Historical'] = hist_vol.estimate()
    
    # EWMA volatility
    ewma_vol = EWMAVolatility(returns, periods_per_year=periods_per_year)
    results['EWMA (λ=0.94)'] = ewma_vol.estimate()
    
    # GARCH volatility
    try:
        garch_vol = GARCHVolatility(returns, periods_per_year)
        results['GARCH(1,1)'] = garch_vol.estimate()
    except:
        results['GARCH(1,1)'] = np.nan
    
    return results
