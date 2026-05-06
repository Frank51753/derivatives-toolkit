"""
Finite Difference Methods for Option Pricing

Solves the Black-Scholes PDE numerically:
∂C/∂t + (1/2)σ²S²(∂²C/∂S²) + r*S(∂C/∂S) - r*C = 0

Methods:
    - Explicit finite difference (simple, can be unstable)
    - Implicit finite difference (stable, requires matrix solution)
    - Crank-Nicolson (better accuracy, stable)

Advantages:
    - Handles American options (early exercise)
    - Handles dividends easily
    - Good for exotic options
    - Provides Greeks via differentiation

References:
    Wilmott, P., Howison, S., & Dewynne, J. (1995). The Mathematics of Financial Derivatives.
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg


class FiniteDifferenceSolver:
    """
    Base class for finite difference solvers.
    
    Solves Black-Scholes PDE on a grid:
    - S: Stock price grid (0 to S_max)
    - t: Time grid (0 to T)
    """
    
    def __init__(
        self,
        S_max: float,
        T: float,
        K: float,
        r: float,
        sigma: float,
        n_S: int = 100,
        n_t: int = 100,
        option_type: str = 'call'
    ):
        """
        Initialize FD solver.
        
        Parameters:
            S_max (float): Maximum stock price in grid
            T (float): Time to maturity
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility
            n_S (int): Number of stock price grid points
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
        
        # Grid setup
        self.dS = S_max / (n_S - 1)
        self.dt = T / (n_t - 1)
        
        self.S = np.linspace(0, S_max, n_S)
        self.t = np.linspace(0, T, n_t)
        
        # Option values grid
        self.V = np.zeros((n_t, n_S))
        
        # Set boundary conditions
        self._set_boundary_conditions()
    
    def _set_boundary_conditions(self):
        """Set initial and boundary conditions."""
        # Initial condition at t=T (maturity)
        if self.option_type == 'call':
            self.V[-1, :] = np.maximum(self.S - self.K, 0)
        else:
            self.V[-1, :] = np.maximum(self.K - self.S, 0)
        
        # Boundary at S=0
        if self.option_type == 'call':
            self.V[:, 0] = 0
        else:
            self.V[:, 0] = self.K * np.exp(-self.r * (self.T - self.t))
        
        # Boundary at S=S_max
        if self.option_type == 'call':
            self.V[:, -1] = self.S_max - self.K * np.exp(-self.r * (self.T - self.t))
        else:
            self.V[:, -1] = 0
    
    def solve(self) -> np.ndarray:
        """Solve the PDE. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def option_price(self, S: float) -> float:
        """
        Get option price at spot price S.
        
        Parameters:
            S (float): Spot price
        
        Returns:
            float: Option price
        """
        # Interpolate on grid
        idx = int(np.round(S / self.dS))
        idx = np.clip(idx, 0, self.n_S - 1)
        return self.V[0, idx]
    
    def delta(self, S: float) -> float:
        """
        Calculate delta by finite difference.
        
        Parameters:
            S (float): Spot price
        
        Returns:
            float: Delta
        """
        idx = int(np.round(S / self.dS))
        idx = np.clip(idx, 1, self.n_S - 2)
        
        # Central difference
        return (self.V[0, idx+1] - self.V[0, idx-1]) / (2 * self.dS)
    
    def gamma(self, S: float) -> float:
        """
        Calculate gamma by finite difference.
        
        Parameters:
            S (float): Spot price
        
        Returns:
            float: Gamma
        """
        idx = int(np.round(S / self.dS))
        idx = np.clip(idx, 1, self.n_S - 2)
        
        # Second central difference
        return (self.V[0, idx+1] - 2*self.V[0, idx] + self.V[0, idx-1]) / (self.dS**2)
    
    def get_price_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full price surface.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (S_grid, t_grid, V_grid)
        """
        return self.S, self.t, self.V


class ExplicitFD(FiniteDifferenceSolver):
    """
    Explicit Finite Difference Method.
    
    Simple forward in time, central in space (FTCS) scheme.
    Recurrence relation:
        V_{i,j}^{n+1} = a*V_{i,j-1}^n + b*V_{i,j}^n + c*V_{i,j+1}^n
    
    where:
        a = [α*i² - β*i] / 2
        b = 1 - α*i² - r*dt
        c = [α*i² + β*i] / 2
        
        α = σ²*dt / dS²
        β = r*dt / dS
    
    Stability requires: α ≤ 1/2
    
    Disadvantages:
        - Can be unstable (needs small dt)
        - Slow convergence
    
    Advantages:
        - Simple to implement
        - No matrix inversion needed
    """
    
    def solve(self) -> np.ndarray:
        """
        Solve using explicit method.
        
        Returns:
            np.ndarray: Option value grid
        """
        alpha = self.sigma**2 * self.dt / (self.dS**2)
        beta = self.r * self.dt / self.dS
        
        # Check stability
        if alpha > 0.5:
            print(f"Warning: alpha = {alpha} > 0.5, solution may be unstable")
        
        # Coefficients
        # For interior points: a*V_{i-1} + b*V_i + c*V_{i+1}
        
        # Backward in time (from T to 0)
        for n in range(self.n_t - 2, -1, -1):
            for i in range(1, self.n_S - 1):
                # Index squared for stability
                i_squared = i**2
                
                a = (alpha * i_squared - beta * i) / 2
                b = 1 - alpha * i_squared - self.r * self.dt
                c = (alpha * i_squared + beta * i) / 2
                
                self.V[n, i] = a * self.V[n+1, i-1] + b * self.V[n+1, i] + c * self.V[n+1, i+1]
            
            # Maintain boundary conditions
            if self.option_type == 'put':
                self.V[n, 0] = self.K * np.exp(-self.r * (self.T - self.t[n]))
        
        return self.V


class ImplicitFD(FiniteDifferenceSolver):
    """
    Implicit Finite Difference Method (Backward Time Central Space - BTCS).
    
    Recurrence relation (backward in time):
        a*V_{i,j-1}^{n} + b*V_{i,j}^{n} + c*V_{i,j+1}^{n} = V_{i,j}^{n+1}
    
    Advantages:
        - Unconditionally stable (works for all dt, dS)
        - Better accuracy than explicit
    
    Disadvantages:
        - Requires solving tridiagonal system at each time step
        - More complex implementation
    
    Examples:
        >>> solver = ImplicitFD(S_max=200, T=1.0, K=100, r=0.05, sigma=0.2)
        >>> V = solver.solve()
        >>> price = solver.option_price(100)
    """
    
    def solve(self) -> np.ndarray:
        """
        Solve using implicit method.
        
        Returns:
            np.ndarray: Option value grid
        """
        alpha = self.sigma**2 * self.dt / (self.dS**2)
        beta = self.r * self.dt / self.dS
        
        # Backward in time
        for n in range(self.n_t - 2, -1, -1):
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
            
            # Add boundary conditions to RHS
            rhs[0] -= a * self.V[n, 0]
            rhs[-1] -= c * self.V[n, -1]
            
            # Solve tridiagonal system
            A = sparse.diags([lower, diag, upper], [-1, 0, 1], 
                            shape=(self.n_S-2, self.n_S-2))
            V_interior = splinalg.spsolve(A.tocsr(), rhs)
            
            # Update interior points
            self.V[n, 1:-1] = V_interior
            
            # Maintain boundary conditions
            if self.option_type == 'put':
                self.V[n, 0] = self.K * np.exp(-self.r * (self.T - self.t[n]))
        
        return self.V


class CrankNicolson(FiniteDifferenceSolver):
    """
    Crank-Nicolson Method: Average of explicit and implicit methods.
    
    Characteristics:
        - O(dt², dS²) accuracy (better than explicit or implicit alone)
        - Unconditionally stable
        - Gold standard for PDE solving
    
    Recurrence:
        Solves weighted average of forward and backward schemes.
    
    Advantages:
        - Better accuracy
        - Stable
        - Widely used in industry
    
    Examples:
        >>> solver = CrankNicolson(S_max=200, T=1.0, K=100, r=0.05, sigma=0.2)
        >>> V = solver.solve()
        >>> price = solver.option_price(100)
    """
    
    def solve(self) -> np.ndarray:
        """
        Solve using Crank-Nicolson method.
        
        Returns:
            np.ndarray: Option value grid
        """
        alpha = self.sigma**2 * self.dt / (self.dS**2)
        beta = self.r * self.dt / self.dS
        
        # Backward in time
        for n in range(self.n_t - 2, -1, -1):
            # Build tridiagonal system
            diag_l = np.zeros(self.n_S - 2)
            upper_l = np.zeros(self.n_S - 3)
            lower_l = np.zeros(self.n_S - 3)
            rhs = np.zeros(self.n_S - 2)
            
            for i in range(1, self.n_S - 1):
                i_idx = i - 1
                i_squared = i**2
                
                # Left side (implicit) coefficients
                a_l = -(alpha * i_squared - beta * i) / 4
                b_l = 1 + alpha * i_squared / 2 + self.r * self.dt / 2
                c_l = -(alpha * i_squared + beta * i) / 4
                
                diag_l[i_idx] = b_l
                if i_idx > 0:
                    lower_l[i_idx - 1] = a_l
                if i_idx < self.n_S - 3:
                    upper_l[i_idx] = c_l
                
                # Right side (explicit) coefficients
                a_r = (alpha * i_squared - beta * i) / 4
                b_r = 1 - alpha * i_squared / 2 - self.r * self.dt / 2
                c_r = (alpha * i_squared + beta * i) / 4
                
                rhs[i_idx] = a_r * self.V[n+1, i-1] + b_r * self.V[n+1, i] + c_r * self.V[n+1, i+1]
            
            # Add boundary conditions
            rhs[0] -= a_l * self.V[n, 0]
            rhs[-1] -= c_l * self.V[n, -1]
            
            # Solve tridiagonal system
            A = sparse.diags([lower_l, diag_l, upper_l], [-1, 0, 1],
                            shape=(self.n_S-2, self.n_S-2))
            V_interior = splinalg.spsolve(A.tocsr(), rhs)
            
            # Update
            self.V[n, 1:-1] = V_interior
            
            # Maintain boundary conditions
            if self.option_type == 'put':
                self.V[n, 0] = self.K * np.exp(-self.r * (self.T - self.t[n]))
        
        return self.V


def compare_fd_methods(
    S_max: float = 200,
    T: float = 1.0,
    K: float = 100,
    r: float = 0.05,
    sigma: float = 0.2,
    n_S: int = 100,
    n_t: int = 100,
    spot: float = 100
) -> Dict[str, float]:
    """
    Compare different FD methods.
    
    Returns:
        Dict[str, float]: Option prices from different methods
    
    Examples:
        >>> prices = compare_fd_methods()
        >>> for method, price in prices.items():
        ...     print(f"{method}: ${price:.2f}")
    """
    results = {}
    
    # Explicit
    explicit = ExplicitFD(S_max, T, K, r, sigma, n_S, n_t)
    explicit.solve()
    results['Explicit'] = explicit.option_price(spot)
    
    # Implicit
    implicit = ImplicitFD(S_max, T, K, r, sigma, n_S, n_t)
    implicit.solve()
    results['Implicit'] = implicit.option_price(spot)
    
    # Crank-Nicolson
    cn = CrankNicolson(S_max, T, K, r, sigma, n_S, n_t)
    cn.solve()
    results['Crank-Nicolson'] = cn.option_price(spot)
    
    # Compare with Black-Scholes
    from .black_scholes import BlackScholesAnalytic
    bs = BlackScholesAnalytic(spot, K, T, r, sigma)
    results['Black-Scholes'] = bs.call_price()
    
    return results
