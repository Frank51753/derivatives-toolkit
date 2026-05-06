"""
Tests for Finite Difference Methods
"""

import pytest
import numpy as np
from derivatives.pricing.finite_difference import (
    ExplicitFD, ImplicitFD, CrankNicolson, compare_fd_methods
)
from derivatives.pricing.black_scholes import BlackScholesAnalytic


class TestFiniteDifferenceMethods:
    """Test FD solver methods."""
    
    @pytest.fixture
    def bs_params(self):
        """Standard parameters for comparison."""
        return {
            'S0': 100,
            'K': 100,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2,
            'n_S': 100,
            'n_t': 100
        }
    
    def test_explicit_fd_initialization(self, bs_params):
        """Test explicit FD initialization."""
        solver = ExplicitFD(**bs_params)
        
        assert solver.S_max == 200
        assert solver.T == 1.0
        assert solver.V.shape == (100, 100)
    
    def test_explicit_fd_solve(self, bs_params):
        """Test explicit FD solution."""
        solver = ExplicitFD(**bs_params)
        V = solver.solve()
        
        # Check shape
        assert V.shape == (100, 100)
        
        # Check option is non-negative
        assert np.all(V >= 0)
        
        # Check monotonicity in S
        assert np.all(np.diff(V[0, :]) >= -0.01)  # Allow small numerical errors
    
    def test_implicit_fd_solve(self, bs_params):
        """Test implicit FD solution."""
        solver = ImplicitFD(**bs_params)
        V = solver.solve()
        
        assert V.shape == (100, 100)
        assert np.all(V >= 0)
    
    def test_crank_nicolson_solve(self, bs_params):
        """Test Crank-Nicolson solution."""
        solver = CrankNicolson(**bs_params)
        V = solver.solve()
        
        assert V.shape == (100, 100)
        assert np.all(V >= 0)
    
    def test_fd_vs_black_scholes(self, bs_params):
        """Test FD solution matches Black-Scholes."""
        # Black-Scholes price
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        bs_price = bs.call_price()
        
        # FD price
        cn = CrankNicolson(**bs_params, n_S=150, n_t=150)
        cn.solve()
        fd_price = cn.option_price(100)
        
        # Should be close (within 5%)
        assert np.isclose(fd_price, bs_price, rtol=0.05)
    
    def test_fd_greeks(self):
        """Test Greeks calculation from FD solution."""
        solver = CrankNicolson(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150
        )
        solver.solve()
        
        # Calculate Greeks at spot = 100
        delta = solver.delta(100)
        gamma = solver.gamma(100)
        
        # Compare with Black-Scholes
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        bs_delta = bs.delta('call')
        bs_gamma = bs.gamma()
        
        # Should be close
        assert np.isclose(delta, bs_delta, rtol=0.1)
        assert np.isclose(gamma, bs_gamma, rtol=0.1)
    
    def test_compare_fd_methods(self):
        """Test comparison of FD methods."""
        prices = compare_fd_methods(n_S=80, n_t=80)
        
        # All methods should give similar answers
        bs_price = prices['Black-Scholes']
        
        for method in ['Explicit', 'Implicit', 'Crank-Nicolson']:
            assert method in prices
            assert abs(prices[method] - bs_price) / bs_price < 0.1
    
    def test_put_option_fd(self):
        """Test FD for put option."""
        solver = CrankNicolson(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150, option_type='put'
        )
        solver.solve()
        
        put_price = solver.option_price(100)
        
        # Compare with Black-Scholes
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        bs_put = bs.put_price()
        
        assert np.isclose(put_price, bs_put, rtol=0.05)
    
    def test_fd_convergence(self):
        """Test convergence with grid refinement."""
        # Coarse grid
        cn_coarse = CrankNicolson(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=50, n_t=50
        )
        cn_coarse.solve()
        price_coarse = cn_coarse.option_price(100)
        
        # Fine grid
        cn_fine = CrankNicolson(
            S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
            n_S=150, n_t=150
        )
        cn_fine.solve()
        price_fine = cn_fine.option_price(100)
        
        # Fine grid should be more accurate to BS
        bs = BlackScholesAnalytic(100, 100, 1.0, 0.05, 0.2)
        bs_price = bs.call_price()
        
        error_coarse = abs(price_coarse - bs_price)
        error_fine = abs(price_fine - bs_price)
        
        # Fine grid should have smaller error
        assert error_fine < error_coarse
