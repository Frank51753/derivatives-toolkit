"""
Core financial instruments: Options and Portfolios
"""

class Option:
    """
    Base class for options
    
    Parameters
    ----------
    strike : float
        The strike price of the option
    maturity : float
        Time to maturity in years
    option_type : str, default='call'
        Type of option: 'call' or 'put'
    """
    
    def __init__(self, strike, maturity, option_type='call'):
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        if maturity <= 0:
            raise ValueError("Maturity must be positive")
        
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
    
    def payoff(self, spot_price):
        """
        Compute the payoff of the option at maturity
        
        Parameters
        ----------
        spot_price : float
            The spot price of the underlying asset
            
        Returns
        -------
        float
            The option payoff
        """
        if spot_price <= 0:
            raise ValueError("Spot price must be positive")
        
        if self.option_type == 'call':
            return max(spot_price - self.strike, 0)
        else:  # put
            return max(self.strike - spot_price, 0)
    
    def profit(self, spot_price, premium):
        """
        Compute the profit from the option position
        
        Parameters
        ----------
        spot_price : float
            The spot price at maturity
        premium : float
            The price paid for the option
            
        Returns
        -------
        float
            Profit = Payoff - Premium
        """
        return self.payoff(spot_price) - premium
    
    def is_in_the_money(self, spot_price):
        """Check if option is in the money"""
        if self.option_type == 'call':
            return spot_price > self.strike
        else:
            return spot_price < self.strike
    
    def intrinsic_value(self, spot_price):
        """Compute intrinsic value"""
        return self.payoff(spot_price)
    
    def __repr__(self):
        return f"{self.option_type.upper()} Option(K={self.strike}, T={self.maturity})"


class EuropeanOption(Option):
    """European option - can only be exercised at maturity"""
    pass


class AmericanOption(Option):
    """American option - can be exercised at any time before maturity"""
    pass


class Portfolio:
    """
    Portfolio class to track positions and compute portfolio value
    
    Parameters
    ----------
    cash : float
        Initial cash amount
    margin_requirement : float, default=0.5
        Minimum margin requirement (50% by default)
    """
    
    def __init__(self, cash, margin_requirement=0.5):
        if cash < 0:
            raise ValueError("Cash amount cannot be negative")
        if not (0 <= margin_requirement <= 1):
            raise ValueError("Margin requirement must be between 0 and 1")
        
        self.initial_cash = cash
        self.cash = cash
        self.positions = {}  # {asset: quantity}
        self.margin_requirement = margin_requirement
        self.trade_history = []
    
    def buy(self, asset, quantity, price):
        """
        Buy an asset
        
        Parameters
        ----------
        asset : str
            Asset name/ticker
        quantity : float
            Number of units to buy
        price : float
            Price per unit
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price <= 0:
            raise ValueError("Price must be positive")
        
        cost = quantity * price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}")
        
        self.cash -= cost
        self.positions[asset] = self.positions.get(asset, 0) + quantity
        self.trade_history.append({
            'action': 'BUY',
            'asset': asset,
            'quantity': quantity,
            'price': price,
            'cost': cost
        })
    
    def short(self, asset, quantity, price):
        """
        Short sell an asset
        
        Parameters
        ----------
        asset : str
            Asset name/ticker
        quantity : float
            Number of units to short
        price : float
            Price per unit
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price <= 0:
            raise ValueError("Price must be positive")
        
        proceeds = quantity * price
        self.cash += proceeds
        self.positions[asset] = self.positions.get(asset, 0) - quantity
        self.trade_history.append({
            'action': 'SHORT',
            'asset': asset,
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds
        })
    
    def check_margin(self, current_prices):
        """
        Check if portfolio meets margin requirements
        
        Parameters
        ----------
        current_prices : dict
            Dictionary of {asset: price}
        """
        equity = self.cash
        total_assets = self.cash
        
        for asset, quantity in self.positions.items():
            price = current_prices.get(asset, 0)
            equity += quantity * price
            total_assets += abs(quantity) * price
        
        margin_percent = (equity / total_assets) if total_assets > 0 else 1.0
        
        return {
            'cash': self.cash,
            'equity': equity,
            'total_assets': total_assets,
            'margin_percent': margin_percent,
            'meets_requirement': margin_percent >= self.margin_requirement
        }
    
    def value(self, current_prices):
        """
        Compute portfolio value
        
        Parameters
        ----------
        current_prices : dict
            Dictionary of {asset: price}
            
        Returns
        -------
        float
            Total portfolio value
        """
        total = self.cash
        for asset, quantity in self.positions.items():
            price = current_prices.get(asset, 0)
            total += quantity * price
        return total
    
    def __repr__(self):
        return f"Portfolio(cash=${self.cash:.2f}, positions={len(self.positions)})"
