"""
Analytics Module

This module provides tools for options analysis, risk management,
and sensitivity calculations.

Classes:
    GreeksCalculator: Comprehensive Greeks calculation
    OptionRiskAnalyzer: Risk and payoff analysis
"""

from .greeks import GreeksCalculator, OptionRiskAnalyzer, compute_greeks_vectorized

__all__ = [
    'GreeksCalculator',
    'OptionRiskAnalyzer',
    'compute_greeks_vectorized',
]
