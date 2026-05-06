"""
Black-Scholes Option Pricing Model

Implements the Black-Scholes formula for European option pricing and
the calculation of option Greeks for risk management.

Theory:
    The Black-Scholes formula prices European options as:
    
    C = S*N(d1) - K*e^(-rT)*N(d2)
    P = K*e^(-rT)*N(-d2) - S*N(-d1)
    
    where:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    N(x) = standard normal CDF

References:
    Black, F., & Scholes, M. (1973). The pricing of options and 
        corporate liabilities. The journal of political economy, 81(3), 637-654.
    Merton, R. C. (1973). Theory of rational option pricing. The Bell 
        Journal of Economics and Management Science, 4(1), 141-183.
"""

import numpy as np
from typing import Optional, Dict
from scipy import stats
from ..utils.math import black_scholes as bs_formula, normal_cdf, normal_pdf


class BlackScholesAnalytic:
    """
    Black-Scholes analytical pricer and Greeks calculator.
    
    Provides:
    - Option price (call and put)
    - All five Greeks: Delta, Gamma, Vega, Theta, Rho
    - Implied volatility solver
    - Sensitivity analysis
    
    Attributes:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
    
    Examples:
        >>> # Initialize with option parameters
        >>> option = BlackScholesAnalytic(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        >>> 
        >>> # Get price
        >>> call_price = option.call_price()
        >>> put_price = option.put_price()
        >>> print(f"Call: ${call_price:.2f}, Put: ${put_price:.2f}")
        >>> 
        >>> # Get Greeks
        >>> delta = option.delta('call')
        >>> gamma = option.gamma()
        >>> vega = option.vega()
        >>> print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")
        >>> 
        >>> # Greeks are annualized (Vega per 1% change in volatility)
        >>> # Theta is per day
        >>> theta = option.theta('call')
        >>> print(f"Theta (per day): {theta/365:.4f}")
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Black-Scholes calculator.
        
        Parameters:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity in years (T > 0)
            r (float): Risk-free interest rate
            sigma (float): Volatility (annualized, sigma > 0)
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if S <= 0:
            raise ValueError(f"Stock price S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to maturity T must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Pre-calculate d1 and d2
        self._recalculate_d1_d2()
    
    def _recalculate_d1_d2(self):
        """Recalculate d1 and d2 if parameters change."""
        ln_S_K = np.log(self.S / self.K)
        sqrt_T = np.sqrt(self.T)
        sigma_sqrt_T = self.sigma * sqrt_T
        
        self.d1 = (ln_S_K + (self.r + 0.5 * self.sigma**2) * self.T) / sigma_sqrt_T
        self.d2 = self.d1 - sigma_sqrt_T
    
    def call_price(self) -> float:
        """
        Black-Scholes European call option price.
        
        Returns:
            float: Call option price
        
        Formula:
            C = S*N(d1) - K*e^(-rT)*N(d2)
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> price = bs.call_price()
            >>> print(f"Call price: ${price:.2f}")
            >>> assert 0 < price < 100  # Call bounded by stock price
        """
        call = (self.S * normal_cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * normal_cdf(self.d2))
        return call
    
    def put_price(self) -> float:
        """
        Black-Scholes European put option price.
        
        Returns:
            float: Put option price
        
        Formula:
            P = K*e^(-rT)*N(-d2) - S*N(-d1)
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> price = bs.put_price()
            >>> print(f"Put price: ${price:.2f}")
            >>> assert 0 < price < self.K * np.exp(-self.r * self.T)
        """
        put = (self.K * np.exp(-self.r * self.T) * normal_cdf(-self.d2) - 
               self.S * normal_cdf(-self.d1))
        return put
    
    def delta(self, option_type: str = 'call') -> float:
        """
        Delta: Sensitivity to stock price changes (dC/dS).
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Delta in range [-1, 1]
        
        Formulas:
            Call Delta = N(d1)
            Put Delta = N(d1) - 1 = -N(-d1)
        
        Notes:
            - Call delta ranges from 0 to 1 (increases with S)
            - Put delta ranges from -1 to 0 (decreases with S)
            - Delta represents hedge ratio: shares to hold per short option
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> 
            >>> call_delta = bs.delta('call')
            >>> put_delta = bs.delta('put')
            >>> print(f"Call delta: {call_delta:.4f}")
            >>> print(f"Put delta: {put_delta:.4f}")
            >>> 
            >>> # Put-call parity: C - P = S - K*e^(-rT)
            >>> assert np.isclose(call_delta - put_delta, 1.0)
        """
        if option_type.lower() == 'call':
            return normal_cdf(self.d1)
        elif option_type.lower() == 'put':
            return normal_cdf(self.d1) - 1
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    def gamma(self) -> float:
        """
        Gamma: Convexity of option price (d²C/dS²).
        
        Returns:
            float: Gamma (always positive)
        
        Formula:
            Γ = n(d1) / (S * σ√T)
        
        Notes:
            - Same for calls and puts
            - Represents rate of change of delta
            - Important for hedging: measures rehedging frequency needed
            - Gamma is highest at-the-money
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> gamma = bs.gamma()
            >>> print(f"Gamma: {gamma:.6f}")
            >>> 
            >>> # Gamma increases as T approaches 0
            >>> bs2 = BlackScholesAnalytic(100, 100, 0.01, 0.05, 0.2)
            >>> print(f"Gamma (T=0.01): {bs2.gamma():.6f}")  # Much larger
        """
        return normal_pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Vega: Sensitivity to volatility changes.
        
        Returns:
            float: Vega per 1% change in volatility
        
        Formula:
            ν = S * n(d1) * √T
        
        Notes:
            - Same for calls and puts
            - Always positive (option buyers benefit from volatility)
            - This is per 1% change (0.01), so multiply by σ_change/0.01
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> vega = bs.vega()
            >>> print(f"Vega: {vega:.2f}")  # Price change per 1% volatility increase
            >>> 
            >>> # 5% increase in volatility
            >>> vega_exposure = vega * 5
            >>> print(f"Price change from 5% σ increase: ${vega_exposure:.2f}")
        """
        return self.S * normal_pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type: str = 'call') -> float:
        """
        Theta: Time decay (dC/dT per day).
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Theta per day (not per year!)
        
        Formulas:
            Call Theta = -S*n(d1)*σ/(2√T) - r*K*e^(-rT)*N(d2)
            Put Theta = -S*n(d1)*σ/(2√T) + r*K*e^(-rT)*N(-d2)
        
        Notes:
            - Time decay per DAY (divide by 365 to get annual)
            - Call theta usually negative (decreases with time)
            - Put theta can be positive (early exercise value)
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> theta_call = bs.theta('call')
            >>> theta_put = bs.theta('put')
            >>> 
            >>> print(f"Call theta per day: ${theta_call:.4f}")
            >>> print(f"Put theta per day: ${theta_put:.4f}")
            >>> 
            >>> # Annual theta (multiply by 365)
            >>> print(f"Call theta annual: ${theta_call * 365:.2f}")
        """
        sqrt_T = np.sqrt(self.T)
        term1 = (self.S * normal_pdf(self.d1) * self.sigma) / (2 * sqrt_T)
        
        if option_type.lower() == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * normal_cdf(self.d2)
            theta = (-term1 - term2) / 365  # Convert to daily
        elif option_type.lower() == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * normal_cdf(-self.d2)
            theta = (-term1 + term2) / 365  # Convert to daily
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        
        return theta
    
    def rho(self, option_type: str = 'call') -> float:
        """
        Rho: Sensitivity to interest rate changes.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Rho per 1% change in interest rate
        
        Formulas:
            Call Rho = K*T*e^(-rT)*N(d2)
            Put Rho = -K*T*e^(-rT)*N(-d2)
        
        Notes:
            - Per 1% change (0.01), so multiply by rate_change/0.01
            - Less important for short-term options
            - More important for long-dated options
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> rho_call = bs.rho('call')
            >>> rho_put = bs.rho('put')
            >>> 
            >>> print(f"Call rho: {rho_call:.2f}")  # Per 1% rate increase
            >>> 
            >>> # Check: call rho should be positive, put rho negative
            >>> assert rho_call > 0 and rho_put < 0
        """
        discount = self.K * self.T * np.exp(-self.r * self.T)
        
        if option_type.lower() == 'call':
            return discount * normal_cdf(self.d2)
        elif option_type.lower() == 'put':
            return -discount * normal_cdf(-self.d2)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    def all_greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks at once.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'price': Option price
                - 'delta': First derivative
                - 'gamma': Second derivative
                - 'vega': Volatility sensitivity
                - 'theta': Time decay (per day)
                - 'rho': Rate sensitivity
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> greeks = bs.all_greeks('call')
            >>> 
            >>> for name, value in greeks.items():
            ...     print(f"{name}: {value:.6f}")
        """
        if option_type.lower() == 'call':
            price = self.call_price()
        elif option_type.lower() == 'put':
            price = self.put_price()
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        
        return {
            'price': price,
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type),
        }
    
    def sensitivity_analysis(
        self,
        stock_range: np.ndarray,
        greeks_to_compute: list = ['price', 'delta', 'gamma']
    ) -> Dict[str, np.ndarray]:
        """
        Compute Greeks over a range of stock prices.
        
        Parameters:
            stock_range (np.ndarray): Array of stock prices to evaluate
            greeks_to_compute (list): List of Greeks to compute
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of sensitivity arrays
        
        Examples:
            >>> bs = BlackScholesAnalytic(100, 100, 0.25, 0.05, 0.2)
            >>> stock_range = np.linspace(80, 120, 50)
            >>> sensitivities = bs.sensitivity_analysis(stock_range, 
            ...                                        ['delta', 'gamma'])
            >>> 
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(stock_range, sensitivities['delta'])
            >>> plt.xlabel('Stock Price')
            >>> plt.ylabel('Delta')
            >>> plt.show()
        """
        results = {}
        
        for greek in greeks_to_compute:
            values = []
            for S in stock_range:
                # Update parameters
                self.S = S
                self._recalculate_d1_d2()
                
                if greek == 'price':
                    values.append(self.call_price())
                elif greek == 'delta':
                    values.append(self.delta('call'))
                elif greek == 'gamma':
                    values.append(self.gamma())
                elif greek == 'vega':
                    values.append(self.vega())
                elif greek == 'theta':
                    values.append(self.theta('call'))
                elif greek == 'rho':
                    values.append(self.rho('call'))
            
            results[greek] = np.array(values)
        
        return results


class EuropeanOption:
    """
    European option with both analytical and numerical pricing.
    
    Provides a high-level interface for European option pricing with
    multiple calculation methods and risk analysis.
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0.0
    ):
        """
        Initialize European option.
        
        Parameters:
            S (float): Stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            q (float): Dividend yield (default 0)
        """
        if option_type not in ['call', 'put']:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q
        
        # Initialize BS pricer
        self.bs = BlackScholesAnalytic(S, K, T, r, sigma)
    
    def price(self) -> float:
        """Get option price."""
        if self.option_type == 'call':
            return self.bs.call_price()
        else:
            return self.bs.put_price()
    
    def delta(self) -> float:
        """Get delta."""
        return self.bs.delta(self.option_type)
    
    def gamma(self) -> float:
        """Get gamma."""
        return self.bs.gamma()
    
    def vega(self) -> float:
        """Get vega."""
        return self.bs.vega()
    
    def theta(self) -> float:
        """Get theta (per day)."""
        return self.bs.theta(self.option_type)
    
    def rho(self) -> float:
        """Get rho."""
        return self.bs.rho(self.option_type)
