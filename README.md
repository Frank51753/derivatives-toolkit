# Derivatives Toolkit

A Python library for derivatives pricing based on "A Course in Derivative Securities" by Kerry Back.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: Call Option Payoff
```python
from derivatives.core.instruments import Option

call = Option(strike=100, maturity=1, option_type='call')
print(call.payoff(105))  # Output: 5
print(call.payoff(95))   # Output: 0
```

### Example 2: Portfolio
```python
from derivatives.core.instruments import Portfolio

portfolio = Portfolio(cash=10000)
portfolio.buy('AAPL', 100, 50)
prices = {'AAPL': 55}
print(portfolio.value(prices))  # Output: 10500
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Features

- Option pricing (Call and Put)
- Portfolio management
- Margin calculations
- Greeks calculation (coming soon)
- Binomial tree pricing (coming soon)
