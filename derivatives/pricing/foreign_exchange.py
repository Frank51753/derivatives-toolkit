"""
Foreign Exchange (FX) Options Pricing

Options on foreign exchange (currency pairs) with special considerations:
- Two interest rates (domestic and foreign)
- Spot and forward rates
- Garman-Kohlhagen model (generalized Black-Scholes)

Theory:
    FX Call: C = S*e^(-r_f*T)*N(d1) - K*e^(-r_d*T)*N(d2)
    
    where:
    d1 = [ln(F/K) + (σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    F = S*e^((r_d - r_f)T)  (forward rate)

References:
    Garman, M. B., & Kohlhagen, S. W. (1983). Foreign currency option values.
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats
from ..utils.math import normal_cdf, normal_pdf


class FXOptionPricer:
    """
    Price foreign exchange options using Garman-Kohlhagen model.
    
    Handles currency pairs with two interest rates.
    
    Examples:
        >>> # EUR/USD call option
        >>> fx = FXOptionPricer(
        ...     spot=1.20,              # EUR/USD spot rate
        ...     K=1.20,                 # Strike price
        ...     T=0.25,                 # 3 months
        ...     r_domestic=0.05,        # USD interest rate
        ...     r_foreign=0.03,         # EUR interest rate
        ...     sigma=0.12              # FX volatility
        ... )
        >>> call_price = fx.call_price()
        >>> put_price = fx.put_price()
    """
    
    def __init__(
        self,
        spot: float,
        K: float,
        T: float,
        r_domestic: float,
        r_foreign: float,
        sigma: float,
        currency_pair: str = 'EUR/USD'
    ):
        """
        Initialize FX option pricer.
        
        Parameters:
            spot (float): Spot exchange rate (foreign per domestic)
            K (float): Strike price
            T (float): Time to maturity (in years)
            r_domestic (float): Domestic interest rate
            r_foreign (float): Foreign interest rate
            sigma (float): FX volatility (annualized)
            currency_pair (str): Currency pair name for reference
        
        Raises:
            ValueError: If parameters are invalid
        """
        if spot <= 0 or K <= 0 or T <= 0:
            raise ValueError("spot, K, T must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        
        self.spot = spot
        self.K = K
        self.T = T
        self.r_d = r_domestic
        self.r_f = r_foreign
        self.sigma = sigma
        self.currency_pair = currency_pair
        
        # Calculate forward rate
        self.forward = spot * np.exp((r_domestic - r_foreign) * T)
        
        # Pre-calculate d1 and d2
        self._recalculate_d1_d2()
    
    def _recalculate_d1_d2(self):
        """Recalculate d1 and d2."""
        sqrt_T = np.sqrt(self.T)
        sigma_sqrt_T = self.sigma * sqrt_T
        
        self.d1 = (np.log(self.forward / self.K) + 0.5 * self.sigma**2 * self.T) / sigma_sqrt_T
        self.d2 = self.d1 - sigma_sqrt_T
    
    def call_price(self) -> float:
        """
        Price foreign currency call option.
        
        Formula:
            C = e^(-r_d*T) * [F*N(d1) - K*N(d2)]
        
        Returns:
            float: Call option price
        
        Examples:
            >>> fx = FXOptionPricer(1.20, 1.20, 0.25, 0.05, 0.03, 0.12)
            >>> price = fx.call_price()
            >>> print(f"EUR/USD call: {price:.4f}")
        """
        call = np.exp(-self.r_d * self.T) * (
            self.forward * normal_cdf(self.d1) - self.K * normal_cdf(self.d2)
        )
        return call
    
    def put_price(self) -> float:
        """
        Price foreign currency put option.
        
        Formula:
            P = e^(-r_d*T) * [K*N(-d2) - F*N(-d1)]
        
        Returns:
            float: Put option price
        """
        put = np.exp(-self.r_d * self.T) * (
            self.K * normal_cdf(-self.d2) - self.forward * normal_cdf(-self.d1)
        )
        return put
    
    def delta(self, option_type: str = 'call') -> float:
        """
        Delta: Sensitivity to spot rate changes.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Delta
        
        Notes:
            Call delta = e^(-r_f*T) * N(d1)
            Put delta = e^(-r_f*T) * (N(d1) - 1)
        """
        discount_f = np.exp(-self.r_f * self.T)
        
        if option_type == 'call':
            return discount_f * normal_cdf(self.d1)
        else:
            return discount_f * (normal_cdf(self.d1) - 1)
    
    def gamma(self) -> float:
        """
        Gamma: Convexity of option price.
        
        Returns:
            float: Gamma
        """
        discount_f = np.exp(-self.r_f * self.T)
        return discount_f * normal_pdf(self.d1) / (self.spot * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Vega: Sensitivity to volatility changes.
        
        Returns:
            float: Vega per 1% volatility change
        """
        discount_f = np.exp(-self.r_f * self.T)
        return discount_f * self.forward * normal_pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type: str = 'call') -> float:
        """
        Theta: Time decay (per day).
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Theta per day
        """
        discount_d = np.exp(-self.r_d * self.T)
        discount_f = np.exp(-self.r_f * self.T)
        sqrt_T = np.sqrt(self.T)
        
        term1 = discount_f * self.forward * normal_pdf(self.d1) * self.sigma / (2 * sqrt_T)
        
        if option_type == 'call':
            term2 = self.r_d * self.forward * normal_cdf(self.d1) * discount_d
            term3 = self.r_f * self.K * normal_cdf(self.d2) * discount_d
        else:
            term2 = -self.r_d * self.forward * normal_cdf(-self.d1) * discount_d
            term3 = -self.r_f * self.K * normal_cdf(-self.d2) * discount_d
        
        return (-term1 + term2 - term3) / 365  # Convert to daily
    
    def rho_domestic(self, option_type: str = 'call') -> float:
        """
        Rho: Sensitivity to domestic interest rate changes.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Rho per 1% rate change
        """
        discount_d = np.exp(-self.r_d * self.T)
        
        if option_type == 'call':
            return -self.T * self.K * discount_d * normal_cdf(self.d2)
        else:
            return self.T * self.K * discount_d * normal_cdf(-self.d2)
    
    def rho_foreign(self, option_type: str = 'call') -> float:
        """
        Rho_foreign: Sensitivity to foreign interest rate changes.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            float: Rho per 1% rate change
        """
        discount_f = np.exp(-self.r_f * self.T)
        
        if option_type == 'call':
            return self.T * self.forward * discount_f * normal_cdf(self.d1)
        else:
            return -self.T * self.forward * discount_f * normal_cdf(-self.d1)
    
    def all_greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks.
        
        Parameters:
            option_type (str): 'call' or 'put'
        
        Returns:
            Dict[str, float]: Dictionary of Greeks
        
        Examples:
            >>> fx = FXOptionPricer(1.20, 1.20, 0.25, 0.05, 0.03, 0.12)
            >>> greeks = fx.all_greeks('call')
            >>> for name, value in greeks.items():
            ...     print(f"{name}: {value:.6f}")
        """
        if option_type == 'call':
            price = self.call_price()
        else:
            price = self.put_price()
        
        return {
            'price': price,
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho_domestic': self.rho_domestic(option_type),
            'rho_foreign': self.rho_foreign(option_type),
        }


class CurrencyPortfolio:
    """
    Manages portfolio of FX options and forwards.
    
    Examples:
        >>> portfolio = CurrencyPortfolio(base_currency='USD')
        >>> portfolio.add_fx_position('EUR', quantity=1000000, spot_rate=1.20)
        >>> portfolio.add_fx_call('GBP', strike=1.35, notional=500000)
        >>> value = portfolio.value(market_rates={'EUR': 1.19, 'GBP': 1.36})
    """
    
    def __init__(self, base_currency: str = 'USD'):
        """Initialize currency portfolio."""
        self.base_currency = base_currency
        self.positions = {}  # Spot positions
        self.options = {}    # FX options
    
    def add_fx_position(self, currency: str, quantity: float, spot_rate: float):
        """Add spot FX position."""
        if currency not in self.positions:
            self.positions[currency] = {'quantity': 0, 'cost': 0}
        
        self.positions[currency]['quantity'] += quantity
        self.positions[currency]['cost'] += quantity * spot_rate
    
    def add_fx_call(
        self,
        currency: str,
        strike: float,
        notional: float,
        T: float,
        r_d: float,
        r_f: float,
        sigma: float,
        quantity: int = 1
    ):
        """Add FX call option."""
        option_key = f"{currency}_call_{strike}"
        self.options[option_key] = {
            'type': 'call',
            'currency': currency,
            'strike': strike,
            'notional': notional,
            'quantity': quantity,
            'pricer': FXOptionPricer(notional, strike, T, r_d, r_f, sigma, f"{currency}/USD")
        }
    
    def value(self, market_rates: Dict[str, float]) -> float:
        """Calculate portfolio value at market rates."""
        value = 0
        
        # Spot positions
        for currency, position in self.positions.items():
            value += position['quantity'] * market_rates[currency]
        
        # Options
        for option_key, option in self.options.items():
            if option['type'] == 'call':
                option_value = option['pricer'].call_price() * option['quantity']
                value += option_value
        
        return value
