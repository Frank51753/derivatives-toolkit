"""
American Options Pricing via PDE

American options allow early exercise, which complicates the valuation.
The Black-Scholes PDE becomes an inequality (variational inequality).

Key difference from European:
    V(S,t) ≥ Intrinsic(S)  (option value >= intrinsic value)
    
The option value equals the maximum of:
    1. Intrinsic value (exercise now)
    2. Continuation value (hold and wait)

Method:
    Use implicit finite difference with early exercise check at each step.

References:
    Acworth, P., Broadie, M., & Glasserman, P. (1997). A comparison of some 
        Monte Carlo and quasi-Monte Carlo techniques for option pricing.
"""

import numpy as np
from typing import Tuple
from scipy import sparse
from scipy.sparse import linalg as splinalg


class AmericanOptionFD:
    """
    American option pricer using finite difference with early exercise.
    
    Examples:
        >>> american = AmericanOptionFD(
        ...     S_max=200, T=1.0, K=100, r=0.05, sigma=0.2,
        ...     option_type='put'
        ... )
        >>> price = american.solve()
        >>> print(f"American put price: ${price:.2f}")
    """
    
    def __init__(
        self,
        S_max: float,
        T: float,
        K: float,
        r: float,
        sigma: float,
        n_S: int = 150,
        n_t: int = 150,
        option_type: str = 'put'
    ):
        """
        Initialize American option solver.
        
        Parameters:
            S_max (float): Maximum stock price in grid
            T (float): Time to maturity
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility
            n_S (int): Number of space grid points
            n_t (int): Number of time grid points
            option_type (str): 'call' or 'put'
        """
        self.S_max = S_max
        self.T = T
        self.K = K
        self.r = r
        self.sigma = sigma
        self.n_S = n_S
        self.n_t = n_t
        self.option_type = option_type
        
        # Grid
        self.dS = S_max / (n_S - 1)
        self.dt = T / (n_t - 1)
        self.S = np.linspace(0, S_max, n_S)
        self.t = np.linspace(0, T, n_t)
        
        # Option value
        self.V = np.zeros((n_t, n_S))
        
        # Intrinsic value
        if option_type == 'call':
            self.intrinsic = np.maximum(self.S - K, 0)
        else:
            self.intrinsic = np.maximum(K - self.S, 0)
        
        # Initial condition
        self.V[-1, :] = self.intrinsic
    
    def solve(self) -> float:
        """
        Solve for American option using implicit FD with early exercise.
        
        Returns:
            float: American option price at S = S_0
        
        Algorithm:
            1. For each time step (backward):
               2. Set up implicit FD system
               3. Solve for tentative option values
               4. Enforce early exercise: V = max(V, intrinsic)
               5. Continue to next time step
        """
        alpha = self.sigma**2 * self.dt / (self.dS**2)
        beta = self.r * self.dt / self.dS
        
        # Backward in time
        for n in range(self.n_t - 2, -1, -1):
            # Boundary conditions
            if self.option_type == 'call':
                V_0 = 0
                V_max = self.S_max - self.K * np.exp(-self.r * (self.T - self.t[n]))
            else:
                V_0 = self.K * np.exp(-self.r * (self.T - self.t[n]))
                V_max = 0
            
            # Build tridiagonal system
            diag = np.zeros(self.n_S - 2)
            upper = np.zeros(self.n_S - 3)
            lower = np.zeros(self.n_S - 3)
            rhs = np.zeros(self.n_S - 2)
            
            for i in range(1, self.n_S - 1):
                i_idx = i - 1
                i_squared = i**2
                
                a = -(alpha * i_squared - beta * i) / 2
                b = 1 + alpha * i_squared + self.r * self.dt
                c = -(alpha * i_squared + beta * i) / 2
                
                diag[i_idx] = b
                if i_idx > 0:
                    lower[i_idx - 1] = a
                if i_idx < self.n_S - 3:
                    upper[i_idx] = c
                
                rhs[i_idx] = self.V[n+1, i]
            
            # Add boundaries to RHS
            rhs[0] -= a * V_0
            rhs[-1] -= c * V_max
            
            # Solve system
            A = sparse.diags([lower, diag, upper], [-1, 0, 1],
                            shape=(self.n_S-2, self.n_S-2))
            V_tentative = splinalg.spsolve(A.tocsr(), rhs)
            
            # Set interior values
            self.V[n, 1:-1] = V_tentative
            self.V[n, 0] = V_0
            self.V[n, -1] = V_max
            
            # **EARLY EXERCISE**: American option value = max(intrinsic, continuation)
            self.V[n, :] = np.maximum(self.V[n, :], self.intrinsic)
        
        # Interpolate option price at initial spot
        idx = int(np.round(self.S_max / 2 / self.dS))  # Assume initial spot = S_max/2
        return self.V[0, idx]
    
    def option_price(self, S: float) -> float:
        """Get option price at spot S."""
        idx = int(np.round(S / self.dS))
        idx = np.clip(idx, 0, self.n_S - 1)
        return self.V[0, idx]
    
    def early_exercise_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the optimal early exercise boundary.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, exercise_prices)
        
        The optimal exercise boundary is where V(S,t) = intrinsic(S).
        """
        exercise_prices = []
        
        for n in range(self.n_t):
            # Find where V = intrinsic (within tolerance)
            diff = np.abs(self.V[n, :] - self.intrinsic)
            idx = np.argmin(diff)
            
            if diff[idx] < 1.0:  # Within $1
                exercise_prices.append(self.S[idx])
            else:
                exercise_prices.append(np.nan)
        
        return self.t, np.array(exercise_prices)
    
    def american_premium(self) -> float:
        """
        Calculate the early exercise premium (American - European).
        
        Returns:
            float: Premium value
        """
        from .black_scholes import BlackScholesAnalytic
        
        S0 = self.S_max / 2
        idx = int(np.round(S0 / self.dS))
        american_price = self.V[0, idx]
        
        bs = BlackScholesAnalytic(S0, self.K, self.T, self.r, self.sigma)
        if self.option_type == 'call':
            european_price = bs.call_price()
        else:
            european_price = bs.put_price()
        
        return american_price - european_price
