"""
Stochastic process models for derivatives pricing.

This module provides implementations of various stochastic processes
used in quantitative finance, including Brownian motion, Itô processes,
and geometric Brownian motion.

Classes:
    BrownianMotion: Standard Brownian motion (Wiener process)
    GeometricBrownianMotion: GBM for stock price modeling
    OrnsteinUhlenbeck: Mean-reverting process
    CIRProcess: Cox-Ingersoll-Ross interest rate model
"""

from .brownian import BrownianMotion
from .ito_process import ItoProcess
from .geometric_brownian import GeometricBrownianMotion

__all__ = [
    'BrownianMotion',
    'ItoProcess',
    'GeometricBrownianMotion',
]
