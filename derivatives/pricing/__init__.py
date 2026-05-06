"""
Derivatives Pricing Module

This module provides implementations of various pricing methods including:
- Black-Scholes closed-form formula
- Binomial tree method
- Monte Carlo simulation
- Finite difference methods

Classes:
    BlackScholesAnalytic: Analytical Black-Scholes pricing
    OptionPricer: Generic option pricer with multiple methods
"""

from .black_scholes import BlackScholesAnalytic, EuropeanOption

__all__ = [
    'BlackScholesAnalytic',
    'EuropeanOption',
]
