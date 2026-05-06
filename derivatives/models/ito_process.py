"""
Itô Process Implementation

This module implements Itô processes and the Itô lemma for
solving stochastic differential equations.

Theory:
    An Itô process is defined by the SDE:
    dX(t) = μ(X,t)dt + σ(X,t)dW(t)
    
    where:
    - μ is the drift coefficient
    - σ is the volatility coefficient
    - dW is the Brownian increment
    
    Itô Lemma (stochastic chain rule):
    For a function f(X,t), df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ∂f/∂x dW

References:
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
    Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from .brownian import BrownianMotion


class ItoProcess:
    """
    Generic Itô process solver.
    
    Solves stochastic differential equations of the form:
        dX(t) = μ(X,t) dt + σ(X,t) dW(t)
    
    using Euler-Maruyama discretization scheme.
    
    Attributes:
        X0 (float): Initial value at t=0
        T (float): Time to maturity
        n_steps (int): Number of time steps
        n_paths (int): Number of sample paths
        drift_func (Callable): Function μ(x, t) -> scalar
        volatility_func (Callable): Function σ(x, t) -> scalar
    
    Examples:
        >>> # Define simple drift and volatility
        >>> def drift(x, t): return 0.05 * x
        >>> def vol(x, t): return 0.2 * x
        >>> 
        >>> # Create process (geometric Brownian motion)
        >>> ito = ItoProcess(X0=100, T=1.0, n_steps=252, n_paths=1000,
        ...                  drift_func=drift, volatility_func=vol)
        >>> paths = ito.solve()
        >>> print(paths.shape)
        (253, 1000)
    """
    
    def __init__(
        self,
        X0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        drift_func: Callable[[float, float], float],
        volatility_func: Callable[[float, float], float],
        random_seed: Optional[int] = None
    ):
        """
        Initialize Itô process.
        
        Parameters:
            X0 (float): Initial value
            T (float): Time to maturity
            n_steps (int): Number of time steps
            n_paths (int): Number of paths to simulate
            drift_func (Callable): Function μ(x, t) returning drift coefficient
            volatility_func (Callable): Function σ(x, t) returning volatility coefficient
            random_seed (Optional[int]): Seed for reproducibility
        
        Raises:
            ValueError: If T <= 0, n_steps <= 0, n_paths <= 0, or X0 < 0
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        if X0 < 0:
            raise ValueError(f"Initial value X0 must be non-negative, got {X0}")
        
        self.X0 = X0
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.drift_func = drift_func
        self.volatility_func = volatility_func
        self.random_seed = random_seed
    
    def solve(self) -> np.ndarray:
        """
        Solve the Itô process using Euler-Maruyama scheme.
        
        Returns:
            np.ndarray: Array of shape (n_steps+1, n_paths) containing
                       the solution paths X(t).
        
        Notes:
            The Euler-Maruyama discretization is:
            X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)ΔW_n
            
            This is a first-order weak scheme with error O(Δt).
        
        Mathematical Background:
            For the general SDE: dX = μ(X,t)dt + σ(X,t)dW
            
            The Euler-Maruyama approximation converges to the true solution
            with strong error O(√Δt) and weak error O(Δt).
        
        Examples:
            >>> # Ornstein-Uhlenbeck process
            >>> def drift(x, t): return -0.5 * x  # Mean reversion
            >>> def vol(x, t): return 0.1
            >>> 
            >>> ito = ItoProcess(X0=1.0, T=5.0, n_steps=1000, n_paths=500,
            ...                  drift_func=drift, volatility_func=vol)
            >>> paths = ito.solve()
            >>> # Paths should revert to mean 0
            >>> print(np.mean(paths[-1, :]))  # Close to 0
        """
        # Generate Brownian increments
        bm = BrownianMotion(T=self.T, n_steps=self.n_steps, n_paths=self.n_paths,
                           random_seed=self.random_seed)
        dW = bm.generate_increments()
        
        # Initialize paths
        X = np.zeros((self.n_steps + 1, self.n_paths))
        X[0, :] = self.X0
        
        # Time grid
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        # Euler-Maruyama iteration
        for i in range(self.n_steps):
            X_current = X[i, :]
            t_current = t[i]
            
            # Calculate drift and volatility at current state
            mu = self.drift_func(X_current, t_current)
            sigma = self.volatility_func(X_current, t_current)
            
            # Update step
            X[i + 1, :] = X_current + mu * self.dt + sigma * dW[i, :]
        
        return X
    
    def solve_with_control_variate(
        self,
        control_process: 'ItoProcess'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using Euler-Maruyama with control variate variance reduction.
        
        Parameters:
            control_process (ItoProcess): Another process used for variance reduction
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (paths_main, paths_control)
        
        Notes:
            Control variates reduce variance by using:
            Y = X - β(C - E[C]) where C is the control and β is optimal coefficient
        
        Examples:
            >>> # Main process
            >>> main_drift = lambda x, t: 0.05 * x
            >>> main_vol = lambda x, t: 0.2 * x
            >>> main = ItoProcess(100, 1.0, 252, 1000, main_drift, main_vol)
            >>> 
            >>> # Control process (simpler version)
            >>> control_drift = lambda x, t: 0.05 * x
            >>> control_vol = lambda x, t: 0.2 * x
            >>> control = ItoProcess(100, 1.0, 252, 1000, control_drift, control_vol)
            >>> 
            >>> main_paths, control_paths = main.solve_with_control_variate(control)
        """
        bm = BrownianMotion(T=self.T, n_steps=self.n_steps, n_paths=self.n_paths,
                           random_seed=self.random_seed)
        dW = bm.generate_increments()
        
        # Solve main process
        X_main = np.zeros((self.n_steps + 1, self.n_paths))
        X_main[0, :] = self.X0
        
        # Solve control process
        X_control = np.zeros((self.n_steps + 1, self.n_paths))
        X_control[0, :] = control_process.X0
        
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        for i in range(self.n_steps):
            # Main process
            mu_main = self.drift_func(X_main[i, :], t[i])
            sigma_main = self.volatility_func(X_main[i, :], t[i])
            X_main[i + 1, :] = X_main[i, :] + mu_main * self.dt + sigma_main * dW[i, :]
            
            # Control process
            mu_control = control_process.drift_func(X_control[i, :], t[i])
            sigma_control = control_process.volatility_func(X_control[i, :], t[i])
            X_control[i + 1, :] = X_control[i, :] + mu_control * self.dt + sigma_control * dW[i, :]
        
        return X_main, X_control
    
    @staticmethod
    def ito_lemma(
        X_paths: np.ndarray,
        func: Callable[[float], float],
        func_dx: Callable[[float], float],
        func_dxx: Callable[[float], float],
        sigma: Callable[[float], float],
        dt: float
    ) -> np.ndarray:
        """
        Apply Itô lemma to transform process.
        
        For Y = f(X), computes:
        dY = (df/dx * dX + 1/2 * d²f/dx² * σ² * dt)
        
        Parameters:
            X_paths (np.ndarray): Paths of X, shape (n_steps+1, n_paths)
            func (Callable): Function f(x)
            func_dx (Callable): First derivative f'(x)
            func_dxx (Callable): Second derivative f''(x)
            sigma (Callable): Volatility function σ(x)
            dt (float): Time step
        
        Returns:
            np.ndarray: Paths of Y = f(X), shape (n_steps+1, n_paths)
        
        Mathematical Formula (Itô Lemma):
            df(X) = (∂f/∂x μ + 1/2 ∂²f/∂x² σ²)dt + ∂f/∂x σ dW
        
        Examples:
            >>> # If X ~ GBM, compute Y = ln(X)
            >>> X_paths = ...  # GBM paths
            >>> func = lambda x: np.log(x)
            >>> func_dx = lambda x: 1/x
            >>> func_dxx = lambda x: -1/x**2
            >>> sigma = lambda x: 0.2*x
            >>> dt = 1/252
            >>> 
            >>> Y_paths = ItoProcess.ito_lemma(X_paths, func, func_dx, func_dxx, sigma, dt)
        """
        Y_paths = np.zeros_like(X_paths)
        Y_paths[0, :] = func(X_paths[0, :])
        
        for i in range(X_paths.shape[0] - 1):
            X_current = X_paths[i, :]
            
            # Calculate derivatives at current X
            f_prime = func_dx(X_current)
            f_double_prime = func_dxx(X_current)
            sig = sigma(X_current)
            
            # Increments (approximation)
            dX = X_paths[i + 1, :] - X_paths[i, :]
            
            # Itô correction term: 1/2 f''(X) σ² dt
            ito_correction = 0.5 * f_double_prime * sig**2 * dt
            
            # Apply Itô lemma
            Y_paths[i + 1, :] = Y_paths[i, :] + f_prime * dX + ito_correction
        
        return Y_paths
