"""
Hedging Strategies and Analysis

Tools for constructing and analyzing hedging strategies,
delta hedging, gamma exposure management, and hedge ratio calculation.

Includes:
    - Delta hedging strategies
    - Gamma scalping
    - Vega hedging
    - Hedge ratio calculation
    - Hedging cost analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class HedgePosition:
    """Represents a hedge position."""
    instrument: str
    quantity: float
    delta: float
    cost: float
    purpose: str = ''


class DeltaHedger:
    """
    Delta hedging strategy and rebalancing analysis.
    
    Maintains delta-neutral position by rebalancing hedge ratio.
    
    Examples:
        >>> # Price European call option
        >>> from derivatives.pricing.black_scholes import BlackScholesAnalytic
        >>> option = BlackScholesAnalytic(S=100, K=100, T=1, r=0.05, sigma=0.2)
        >>> 
        >>> hedger = DeltaHedger(
        ...     short_call_qty=100,
        ...     option_delta=option.delta('call'),
        ...     option_gamma=option.gamma(),
        ...     option_vega=option.vega()
        ... )
        >>> 
        >>> # Rebalance as stock moves
        >>> cost = hedger.rebalance(new_stock_price=105)
    """
    
    def __init__(
        self,
        short_call_qty: int,
        option_delta: float,
        option_gamma: float = None,
        option_vega: float = None,
        initial_stock_price: float = 100
    ):
        """
        Initialize delta hedger.
        
        Parameters:
            short_call_qty (int): Number of short calls
            option_delta (float): Delta of each call
            option_gamma (float): Gamma of each call
            option_vega (float): Vega of each call
            initial_stock_price (float): Stock price at initiation
        """
        self.short_calls = short_call_qty
        self.option_delta = option_delta
        self.option_gamma = option_gamma or 0
        self.option_vega = option_vega or 0
        self.stock_price = initial_stock_price
        
        # Calculate initial hedge
        self.long_stock = short_call_qty * option_delta
        
        # Track rebalancing
        self.rebalance_history = [
            {
                'stock_price': initial_stock_price,
                'stock_position': self.long_stock,
                'delta': 0,
                'rebalance_cost': 0,
                'total_cost': 0
            }
        ]
    
    def rebalance(self, new_stock_price: float, new_delta: float) -> float:
        """
        Rebalance delta hedge at new stock price and delta.
        
        Parameters:
            new_stock_price (float): New stock price
            new_delta (float): New delta of option
        
        Returns:
            float: Cost of rebalancing (positive = we pay, negative = we receive)
        
        Examples:
            >>> cost = hedger.rebalance(new_stock_price=105, new_delta=0.65)
            >>> print(f"Rebalance cost: ${cost:.2f}")
        """
        # New required stock position
        new_stock_position = self.short_calls * new_delta
        
        # Shares to buy/sell
        shares_to_trade = new_stock_position - self.long_stock
        
        # Rebalance cost
        rebalance_cost = shares_to_trade * new_stock_price
        
        # Update positions
        self.long_stock = new_stock_position
        self.stock_price = new_stock_price
        self.option_delta = new_delta
        
        # Record
        total_cost = sum(r['rebalance_cost'] for r in self.rebalance_history) + rebalance_cost
        self.rebalance_history.append({
            'stock_price': new_stock_price,
            'stock_position': new_stock_position,
            'delta': new_delta,
            'rebalance_cost': rebalance_cost,
            'total_cost': total_cost
        })
        
        return rebalance_cost
    
    def portfolio_delta(self) -> float:
        """Calculate current portfolio delta."""
        return self.short_calls * (-self.option_delta) + self.long_stock
    
    def hedging_effectiveness(self) -> float:
        """
        Measure hedging effectiveness.
        
        Returns:
            float: Percentage of option delta hedged (0 to 1)
        """
        required_hedge = self.short_calls * self.option_delta
        return min(1.0, abs(self.long_stock) / required_hedge) if required_hedge != 0 else 1.0
    
    def get_rebalance_summary(self) -> Dict:
        """Get summary of rebalancing activity."""
        history = self.rebalance_history
        return {
            'n_rebalances': len(history) - 1,
            'total_cost': history[-1]['total_cost'],
            'avg_rebalance_cost': np.mean([r['rebalance_cost'] for r in history[1:]]),
            'max_rebalance_cost': max(abs(r['rebalance_cost']) for r in history[1:]),
            'current_delta': self.portfolio_delta()
        }


class PortfolioHedge:
    """
    Hedge analysis for a portfolio of derivatives.
    
    Examples:
        >>> # Portfolio with multiple positions
        >>> portfolio = PortfolioHedge()
        >>> portfolio.add_option('call', strike=100, delta=0.6, quantity=100)
        >>> portfolio.add_option('put', strike=100, delta=-0.4, quantity=50)
        >>> 
        >>> # Calculate hedge needed
        >>> hedge_ratio = portfolio.hedge_ratio()
        >>> print(f"Need to short {hedge_ratio} shares per dollar hedged")
    """
    
    def __init__(self):
        """Initialize portfolio."""
        self.positions = []
    
    def add_option(
        self,
        option_type: str,
        strike: float,
        delta: float,
        quantity: int,
        Greeks: Optional[Dict] = None
    ):
        """
        Add option position.
        
        Parameters:
            option_type (str): 'call' or 'put'
            strike (float): Strike price
            delta (float): Option delta
            quantity (int): Number of contracts
            Greeks (Optional[Dict]): Other Greeks (gamma, vega, theta, rho)
        """
        position = {
            'type': option_type,
            'strike': strike,
            'delta': delta,
            'quantity': quantity,
            'greeks': Greeks or {}
        }
        self.positions.append(position)
    
    def add_stock(self, quantity: float):
        """Add stock position."""
        self.positions.append({
            'type': 'stock',
            'delta': 1.0,
            'quantity': quantity,
            'greeks': {}
        })
    
    def portfolio_delta(self) -> float:
        """Calculate total portfolio delta."""
        total_delta = 0
        for pos in self.positions:
            if pos['type'] == 'stock':
                total_delta += pos['quantity']
            else:
                total_delta += pos['quantity'] * pos['delta']
        return total_delta
    
    def portfolio_gamma(self) -> float:
        """Calculate total portfolio gamma."""
        total_gamma = 0
        for pos in self.positions:
            gamma = pos['greeks'].get('gamma', 0)
            total_gamma += pos['quantity'] * gamma
        return total_gamma
    
    def portfolio_vega(self) -> float:
        """Calculate total portfolio vega."""
        total_vega = 0
        for pos in self.positions:
            vega = pos['greeks'].get('vega', 0)
            total_vega += pos['quantity'] * vega
        return total_vega
    
    def portfolio_theta(self) -> float:
        """Calculate total portfolio theta (daily decay)."""
        total_theta = 0
        for pos in self.positions:
            theta = pos['greeks'].get('theta', 0)
            total_theta += pos['quantity'] * theta
        return total_theta
    
    def hedge_ratio(self) -> float:
        """
        Calculate hedge ratio needed to achieve delta neutrality.
        
        Returns:
            float: Shares to sell per unit of portfolio (to hedge)
        """
        portfolio_delta = self.portfolio_delta()
        return -portfolio_delta if portfolio_delta != 0 else 0
    
    def hedge_cost(self, stock_price: float) -> float:
        """
        Calculate cost of hedging to delta-neutral.
        
        Parameters:
            stock_price (float): Current stock price
        
        Returns:
            float: Cost to establish delta hedge
        """
        hedge_ratio = self.hedge_ratio()
        return hedge_ratio * stock_price
    
    def get_greeks_summary(self) -> Dict[str, float]:
        """Get summary of all portfolio Greeks."""
        return {
            'delta': self.portfolio_delta(),
            'gamma': self.portfolio_gamma(),
            'vega': self.portfolio_vega(),
            'theta': self.portfolio_theta()
        }


class HedgingStrategies:
    """
    Pre-built hedging strategies.
    
    Provides common hedging strategies and their characteristics.
    """
    
    @staticmethod
    def long_call_hedge(
        stock_position: int,
        strike: float,
        call_delta: float
    ) -> Dict:
        """
        Protect stock position with long call (collar).
        
        Parameters:
            stock_position (int): Number of shares
            strike (float): Call strike (usually OTM)
            call_delta (float): Delta of call
        
        Returns:
            Dict: Strategy details
        
        Examples:
            >>> strategy = HedgingStrategies.long_call_hedge(
            ...     stock_position=1000,
            ...     strike=110,
            ...     call_delta=0.4
            ... )
            >>> print(f"Cost: ${strategy['cost']:.2f}")
            >>> print(f"Max loss: ${strategy['max_loss']:.2f}")
        """
        calls_needed = stock_position / (1 / call_delta) if call_delta != 0 else 0
        
        return {
            'strategy': 'Long Call Protection',
            'stock_qty': stock_position,
            'call_qty': calls_needed,
            'max_loss': (strike - 100) * stock_position,  # Assumes current = 100
            'max_gain': 'Unlimited',
            'breakeven': 100 + calls_needed  # Simplification
        }
    
    @staticmethod
    def put_spread_hedge(
        stock_position: int,
        long_put_strike: float,
        short_put_strike: float,
        long_put_price: float,
        short_put_price: float
    ) -> Dict:
        """
        Protect stock with put spread (reduces cost).
        
        Parameters:
            stock_position (int): Number of shares
            long_put_strike (float): Strike of long put
            short_put_strike (float): Strike of short put (lower)
            long_put_price (float): Price of long put
            short_put_price (float): Price of short put
        
        Returns:
            Dict: Strategy details
        """
        net_cost = (long_put_price - short_put_price) * stock_position
        max_loss = (long_put_strike - short_put_strike) * stock_position - net_cost
        
        return {
            'strategy': 'Put Spread Protection',
            'stock_qty': stock_position,
            'long_put_strike': long_put_strike,
            'short_put_strike': short_put_strike,
            'net_cost': net_cost,
            'max_loss': max_loss,
            'protection_starts': long_put_strike,
            'protection_ends': short_put_strike
        }
