"""
Forwards and Futures Pricing

Implements pricing of forward contracts and futures contracts,
with adjustments for carry costs and mark-to-market effects.

Theory:
    Forward Price: F = S * e^((r - q) * T)
    
    where:
    - S = spot price
    - r = risk-free rate
    - q = dividend/carry yield
    - T = time to maturity
    
    Futures = similar to forward but with daily settlement (mark-to-market)

References:
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
    Chance, D. M. (2002). A Chronology of Derivatives.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta


class ForwardContract:
    """
    Valuation of forward contracts.
    
    A forward contract is an agreement to buy/sell an asset at a future date
    at a predetermined price (forward price).
    
    Examples:
        >>> # Gold forward contract
        >>> forward = ForwardContract(
        ...     spot=1800,              # Spot price ($/oz)
        ...     T=0.5,                  # 6 months
        ...     r=0.05,                 # Risk-free rate
        ...     q=0.02,                 # Convenience yield
        ...     entry_date=datetime.now(),
        ...     forward_price=1836      # Agreed forward price
        ... )
        >>> value = forward.value()
        >>> print(f"Forward value: ${value:.2f}")
    """
    
    def __init__(
        self,
        spot: float,
        T: float,
        r: float,
        q: float = 0.0,
        entry_date: Optional[datetime] = None,
        forward_price: Optional[float] = None,
        asset_type: str = 'commodity'
    ):
        """
        Initialize forward contract.
        
        Parameters:
            spot (float): Current spot price of underlying
            T (float): Time to maturity (in years)
            r (float): Risk-free rate
            q (float): Convenience yield or dividend yield
            entry_date (Optional[datetime]): Date when forward was entered
            forward_price (Optional[float]): Price agreed at initiation
            asset_type (str): Type of asset ('commodity', 'stock', 'currency', 'bond')
        
        Raises:
            ValueError: If spot <= 0, T <= 0
        """
        if spot <= 0:
            raise ValueError(f"Spot price must be positive, got {spot}")
        if T <= 0:
            raise ValueError(f"Time to maturity T must be positive, got {T}")
        
        self.spot = spot
        self.T = T
        self.r = r
        self.q = q
        self.entry_date = entry_date or datetime.now()
        self.asset_type = asset_type
        
        # Calculate theoretical forward price
        self.theoretical_forward = self._calculate_forward_price()
        
        # Set agreed forward price
        self.forward_price = forward_price if forward_price is not None else self.theoretical_forward
    
    def _calculate_forward_price(self) -> float:
        """
        Calculate theoretical forward price.
        
        Formula:
            F = S * e^((r - q) * T)
        
        where q is convenience yield (for commodities) or dividend yield (for stocks).
        """
        return self.spot * np.exp((self.r - self.q) * self.T)
    
    def value(self, current_spot: Optional[float] = None) -> float:
        """
        Calculate value of forward contract.
        
        Parameters:
            current_spot (Optional[float]): Current spot price
                                          (if None, uses initial spot)
        
        Returns:
            float: Value of forward contract (to long position)
        
        Formula:
            Value = (F_new - F_agreed) * e^(-r*T)
            
            where F_new is the current forward price.
        
        Examples:
            >>> forward = ForwardContract(spot=100, T=1, r=0.05, forward_price=105)
            >>> print(forward.value())  # Value at initiation
            
            >>> print(forward.value(current_spot=110))  # Value after spot moves
        """
        if current_spot is None:
            current_spot = self.spot
        
        # Calculate current forward price
        current_forward = current_spot * np.exp((self.r - self.q) * self.T)
        
        # Value to long position
        value = (current_forward - self.forward_price) * np.exp(-self.r * self.T)
        
        return value
    
    def profit_loss(self, current_spot: Optional[float] = None) -> float:
        """
        Calculate profit/loss on forward contract.
        
        Parameters:
            current_spot (Optional[float]): Current spot price
        
        Returns:
            float: P&L at maturity (if current_spot provided, P&L if closed today)
        
        Examples:
            >>> forward = ForwardContract(spot=100, T=1, r=0.05, forward_price=105)
            >>> pnl = forward.profit_loss(current_spot=110)
            >>> print(f"Profit/Loss: ${pnl:.2f}")
        """
        if current_spot is None:
            # At maturity
            return current_spot - self.forward_price
        else:
            # If closed before maturity
            return self.value(current_spot)
    
    def fair_value(self) -> float:
        """
        Get fair forward price with no profit/loss at initiation.
        
        Returns:
            float: Fair forward price
        """
        return self.theoretical_forward
    
    def is_overpriced(self) -> bool:
        """Check if forward is overpriced (forward_price > theoretical)."""
        return self.forward_price > self.theoretical_forward
    
    def arbitrage_opportunity(self) -> Optional[Dict[str, str]]:
        """
        Identify arbitrage if forward is mispriced.
        
        Returns:
            Optional[Dict]: Arbitrage strategy or None if fairly priced
        
        Examples:
            >>> forward = ForwardContract(spot=100, T=1, r=0.05, forward_price=110)
            >>> arb = forward.arbitrage_opportunity()
            >>> if arb:
            ...     print(arb['strategy'])
            ...     print(f"Profit: ${arb['profit']:.2f}")
        """
        tolerance = 0.01  # 1 cent tolerance
        
        if self.forward_price > self.theoretical_forward + tolerance:
            # Overpriced: cash and carry
            profit = self.forward_price - self.theoretical_forward
            return {
                'strategy': 'Cash and Carry',
                'action': 'Buy spot, sell forward',
                'profit': profit,
                'pct_return': 100 * profit / self.spot
            }
        elif self.forward_price < self.theoretical_forward - tolerance:
            # Underpriced: reverse cash and carry
            profit = self.theoretical_forward - self.forward_price
            return {
                'strategy': 'Reverse Cash and Carry',
                'action': 'Short spot, buy forward',
                'profit': profit,
                'pct_return': 100 * profit / self.spot
            }
        
        return None


class FuturesContract:
    """
    Futures contract: Standardized forward with daily settlement.
    
    Similar to forwards but with important differences:
    - Standardized contracts
    - Mark-to-market daily (cash settlement)
    - Exchange-traded (lower credit risk)
    - Margin requirements
    
    Examples:
        >>> # S&P 500 futures
        >>> futures = FuturesContract(
        ...     spot=4500,
        ...     T=0.25,                  # 3 months
        ...     r=0.05,
        ...     multiplier=250,          # Contract multiplier
        ...     contract_name='ES (S&P 500)'
        ... )
        >>> print(futures.get_contract_specifications())
    """
    
    def __init__(
        self,
        spot: float,
        T: float,
        r: float,
        q: float = 0.0,
        multiplier: float = 1.0,
        tick_size: float = 0.01,
        contract_name: str = 'Unknown'
    ):
        """
        Initialize futures contract.
        
        Parameters:
            spot (float): Current spot price
            T (float): Time to maturity
            r (float): Risk-free rate
            q (float): Dividend/carry yield
            multiplier (float): Contract multiplier (size)
            tick_size (float): Minimum price movement
            contract_name (str): Name of contract
        """
        if spot <= 0 or T <= 0:
            raise ValueError("spot and T must be positive")
        
        self.spot = spot
        self.T = T
        self.r = r
        self.q = q
        self.multiplier = multiplier
        self.tick_size = tick_size
        self.contract_name = contract_name
        
        # Futures price (same as theoretical forward price)
        self.futures_price = spot * np.exp((r - q) * T)
        
        # Mark-to-market records
        self.mtm_history = [(datetime.now(), self.futures_price, 0)]  # (date, price, mtm_pnl)
    
    def mark_to_market(self, new_price: float) -> float:
        """
        Daily mark-to-market settlement.
        
        Parameters:
            new_price (float): New futures price
        
        Returns:
            float: Daily P&L (cash settlement)
        
        Examples:
            >>> futures = FuturesContract(spot=4500, T=0.25, r=0.05, multiplier=250)
            >>> daily_pnl = futures.mark_to_market(4510)
            >>> print(f"Daily P&L: ${daily_pnl:.2f}")
        """
        # Calculate daily P&L
        last_price = self.mtm_history[-1][1]
        daily_pnl = (new_price - last_price) * self.multiplier
        
        # Record
        self.mtm_history.append((datetime.now(), new_price, daily_pnl))
        self.futures_price = new_price
        
        return daily_pnl
    
    def cumulative_pnl(self) -> float:
        """Calculate cumulative P&L from mark-to-market."""
        return sum(pnl for _, _, pnl in self.mtm_history[1:])
    
    def contract_value(self) -> float:
        """Get notional value of one contract."""
        return self.spot * self.multiplier
    
    def get_contract_specifications(self) -> Dict[str, float]:
        """Get contract specifications."""
        return {
            'contract_name': self.contract_name,
            'spot_price': self.spot,
            'futures_price': self.futures_price,
            'multiplier': self.multiplier,
            'contract_value': self.contract_value(),
            'tick_size': self.tick_size,
            'minimum_tick_value': self.tick_size * self.multiplier,
            'time_to_maturity': self.T,
            'cumulative_pnl': self.cumulative_pnl()
        }


class CarryBasedPricer:
    """
    Price forwards/futures using Cost of Carry model.
    
    Formula:
        F = S * e^((r + u - y) * T)
    
    where:
        - r = risk-free rate
        - u = storage/convenience costs
        - y = income/yield
    
    Examples:
        >>> # Gold futures with storage cost
        >>> gold = CarryBasedPricer(
        ...     spot=1800,
        ...     r=0.05,
        ...     storage_cost=0.01,      # 1% annual storage
        ...     convenience_yield=0.02,  # 2% convenience yield
        ...     T=0.5
        ... )
        >>> fair_price = gold.fair_price()
    """
    
    def __init__(
        self,
        spot: float,
        r: float,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
        T: float = 1.0,
        cost_description: str = ''
    ):
        """
        Initialize carry-based pricer.
        
        Parameters:
            spot (float): Spot price
            r (float): Risk-free rate
            storage_cost (float): Storage/carrying cost (annualized)
            convenience_yield (float): Convenience yield (annualized)
            T (float): Time to maturity
            cost_description (str): Description of costs
        """
        self.spot = spot
        self.r = r
        self.storage_cost = storage_cost
        self.convenience_yield = convenience_yield
        self.T = T
        self.cost_description = cost_description
    
    def fair_price(self) -> float:
        """
        Calculate fair forward/futures price.
        
        Formula:
            F = S * e^((r + u - y) * T)
        """
        cost_of_carry = self.r + self.storage_cost - self.convenience_yield
        return self.spot * np.exp(cost_of_carry * self.T)
    
    def implied_carry(self, market_price: float) -> float:
        """
        Infer cost of carry from market price.
        
        Parameters:
            market_price (float): Observed futures price
        
        Returns:
            float: Implied cost of carry
        """
        return np.log(market_price / self.spot) / self.T
    
    def implied_convenience_yield(self, market_price: float) -> float:
        """
        Infer convenience yield from market price.
        
        Parameters:
            market_price (float): Observed futures price
        
        Returns:
            float: Implied convenience yield
        """
        implied_carry = self.implied_carry(market_price)
        return self.r + self.storage_cost - implied_carry
