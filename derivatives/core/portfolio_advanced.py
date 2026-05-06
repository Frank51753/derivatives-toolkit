"""
Advanced Portfolio Management

Manages complex portfolios with multiple derivatives,
calculates portfolio Greeks, monitors risk.

Features:
    - Multiple positions (stocks, calls, puts, forwards, etc.)
    - Portfolio Greeks aggregation
    - Value-at-Risk (VaR) calculation
    - Greeks sensitivity analysis
    - Rehedging optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """Represents a single position in portfolio."""
    instrument_type: str  # 'stock', 'call', 'put', 'forward', etc.
    quantity: float
    strike: Optional[float] = None
    maturity: Optional[float] = None
    Greeks: Dict[str, float] = field(default_factory=dict)
    cost_basis: float = 0.0
    current_price: float = 0.0


class AdvancedPortfolio:
    """
    Advanced portfolio manager with risk analytics.
    
    Examples:
        >>> portfolio = AdvancedPortfolio(base_currency='USD')
        >>> 
        >>> # Add positions
        >>> portfolio.add_position('stock', 'AAPL', quantity=100, price=150)
        >>> portfolio.add_position('call', 'AAPL_C100', strike=100, maturity=0.25, 
        ...                       delta=0.6, gamma=0.02, vega=10, quantity=10)
        >>> 
        >>> # Get portfolio stats
        >>> stats = portfolio.get_portfolio_stats()
        >>> print(f"Portfolio value: ${stats['total_value']:.2f}")
        >>> print(f"Delta: {stats['total_delta']:.2f}")
        >>> print(f"Vega: ${stats['total_vega']:.2f}")
    """
    
    def __init__(self, base_currency: str = 'USD', initial_cash: float = 0.0):
        """Initialize portfolio."""
        self.base_currency = base_currency
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.pnl_history = []
    
    def add_position(
        self,
        instrument_type: str,
        instrument_id: str,
        quantity: float,
        price: float = 0.0,
        strike: Optional[float] = None,
        maturity: Optional[float] = None,
        Greeks: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add position to portfolio.
        
        Parameters:
            instrument_type (str): Type of instrument
            instrument_id (str): Unique identifier
            quantity (float): Number of units
            price (float): Current price
            strike (Optional[float]): Strike for options
            maturity (Optional[float]): Time to maturity
            Greeks (Optional[Dict]): Greeks dictionary
        """
        position = Position(
            instrument_type=instrument_type,
            quantity=quantity,
            strike=strike,
            maturity=maturity,
            Greeks=Greeks or {},
            cost_basis=quantity * price,
            current_price=price
        )
        
        self.positions[instrument_id] = position
    
    def remove_position(self, instrument_id: str) -> None:
        """Remove position from portfolio."""
        if instrument_id in self.positions:
            del self.positions[instrument_id]
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update market prices of positions.
        
        Parameters:
            prices (Dict[str, float]): {instrument_id: price}
        """
        for instrument_id, price in prices.items():
            if instrument_id in self.positions:
                self.positions[instrument_id].current_price = price
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for pos in self.positions.values():
            if pos.instrument_type == 'stock':
                total += pos.quantity * pos.current_price
            else:  # Options
                total += pos.quantity * pos.current_price
        return total
    
    def get_portfolio_delta(self) -> float:
        """Calculate total portfolio delta."""
        total_delta = 0
        for pos in self.positions.values():
            if pos.instrument_type == 'stock':
                total_delta += pos.quantity  # Stock delta = 1
            else:
                delta = pos.Greeks.get('delta', 0)
                total_delta += pos.quantity * delta
        return total_delta
    
    def get_portfolio_gamma(self) -> float:
        """Calculate total portfolio gamma."""
        total_gamma = 0
        for pos in self.positions.values():
            gamma = pos.Greeks.get('gamma', 0)
            total_gamma += pos.quantity * gamma
        return total_gamma
    
    def get_portfolio_vega(self) -> float:
        """Calculate total portfolio vega."""
        total_vega = 0
        for pos in self.positions.values():
            vega = pos.Greeks.get('vega', 0)
            total_vega += pos.quantity * vega
        return total_vega
    
    def get_portfolio_theta(self) -> float:
        """Calculate total portfolio theta (daily)."""
        total_theta = 0
        for pos in self.positions.values():
            theta = pos.Greeks.get('theta', 0)
            total_theta += pos.quantity * theta
        return total_theta
    
    def get_portfolio_rho(self) -> float:
        """Calculate total portfolio rho."""
        total_rho = 0
        for pos in self.positions.values():
            rho = pos.Greeks.get('rho', 0)
            total_rho += pos.quantity * rho
        return total_rho
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get comprehensive portfolio statistics."""
        return {
            'total_value': self.get_portfolio_value(),
            'cash': self.cash,
            'total_delta': self.get_portfolio_delta(),
            'total_gamma': self.get_portfolio_gamma(),
            'total_vega': self.get_portfolio_vega(),
            'total_theta': self.get_portfolio_theta(),
            'total_rho': self.get_portfolio_rho(),
            'n_positions': len(self.positions)
        }
    
    def calculate_var(
        self,
        confidence: float = 0.95,
        time_horizon: float = 1.0
    ) -> float:
        """
        Calculate Value-at-Risk (VaR).
        
        Parameters:
            confidence (float): Confidence level (0.95 = 95%)
            time_horizon (float): Time horizon in years
        
        Returns:
            float: VaR amount
        
        Notes:
            Uses delta-normal approximation:
            VaR ≈ sqrt(time_horizon) * Φ^(-1)(confidence) * portfolio_delta * volatility * spot
        """
        from scipy.stats import norm
        
        # Simple approximation
        portfolio_delta = self.get_portfolio_delta()
        
        # Assume 20% volatility and $100 spot
        sigma = 0.20
        spot = 100
        
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        var = portfolio_delta * spot * sigma * np.sqrt(time_horizon) * z_score
        
        return abs(var)
    
    def calculate_pnl(self, previous_prices: Dict[str, float]) -> float:
        """
        Calculate daily P&L.
        
        Parameters:
            previous_prices (Dict[str, float]): Previous day prices
        
        Returns:
            float: Daily P&L
        """
        pnl = 0
        for instrument_id, pos in self.positions.items():
            if instrument_id in previous_prices:
                price_change = pos.current_price - previous_prices[instrument_id]
                pnl += pos.quantity * price_change
        
        return pnl
    
    def stress_test(
        self,
        spot_shock: float,
        vol_shock: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Stress test portfolio for spot and vol shocks.
        
        Parameters:
            spot_shock (float): Spot price change
            vol_shock (float): Volatility change
        
        Returns:
            Tuple[float, float, float]: (delta_pnl, gamma_pnl, vega_pnl)
        
        Notes:
            Uses Taylor expansion:
            ΔV ≈ delta*ΔS + (1/2)*gamma*ΔS² + vega*Δσ
        """
        delta = self.get_portfolio_delta()
        gamma = self.get_portfolio_gamma()
        vega = self.get_portfolio_vega()
        
        delta_pnl = delta * spot_shock
        gamma_pnl = 0.5 * gamma * spot_shock**2
        vega_pnl = vega * vol_shock
        
        return delta_pnl, gamma_pnl, vega_pnl
    
    def optimal_hedge_ratio(self) -> float:
        """
        Calculate optimal hedge ratio to achieve delta neutrality.
        
        Returns:
            float: Shares to buy/sell per unit of portfolio (negative = sell)
        """
        portfolio_delta = self.get_portfolio_delta()
        return -portfolio_delta
    
    def hedge_cost(self, stock_price: float) -> float:
        """
        Calculate cost of delta hedging.
        
        Parameters:
            stock_price (float): Current stock price
        
        Returns:
            float: Cost to establish hedge
        """
        hedge_ratio = self.optimal_hedge_ratio()
        return hedge_ratio * stock_price
