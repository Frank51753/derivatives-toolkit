"""
Brownian Motion (Wiener Process) Implementation

This module implements standard Brownian motion and related processes
for use in stochastic differential equation solutions.

Theory:
    A standard Brownian motion W(t) satisfies:
    1. W(0) = 0
    2. W(t) ~ N(0, t)
    3. Independent increments: dW(t) ~ N(0, dt)
    4. Continuous paths (almost surely)

References:
    Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.).
    Chapter 2: Properties of Stock Options
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


class BrownianMotion:
    """
    Standard Brownian motion (Wiener process) simulator.
    
    A Brownian motion is a continuous-time stochastic process with:
    - Independent increments
    - Stationary increments that are normally distributed
    - Continuous paths
    
    This class generates sample paths for numerical solutions to stochastic
    differential equations and Monte Carlo simulations.
    
    Attributes:
        dt (float): Time step size
        n_steps (int): Number of time steps
        n_paths (int): Number of sample paths to simulate
        T (float): Total time to maturity
        random_seed (Optional[int]): Random seed for reproducibility
    
    Examples:
        >>> # Create Brownian motion object
        >>> bm = BrownianMotion(T=1.0, n_steps=252, n_paths=10000)
        >>> paths = bm.generate_paths()
        >>> print(paths.shape)
        (10001, 10000)  # (time_steps, paths)
        
        >>> # Check increments are independent
        >>> dW = np.diff(paths, axis=0)
        >>> print(np.cov(dW[:,0], dW[:,1]))  # Should be close to 0
    """
    
    def __init__(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Brownian motion object.
        
        Parameters:
            T (float): Total time horizon (in years, default 1.0)
            n_steps (int): Number of discrete time steps (default 252 for daily)
            n_paths (int): Number of independent sample paths (default 1000)
            random_seed (Optional[int]): Seed for reproducibility
        
        Raises:
            ValueError: If T <= 0, n_steps <= 0, or n_paths <= 0
        """
        if T <= 0:
            raise ValueError(f"Time horizon T must be positive, got {T}")
        if n_steps <= 0:
            raise ValueError(f"Number of steps must be positive, got {n_steps}")
        if n_paths <= 0:
            raise ValueError(f"Number of paths must be positive, got {n_paths}")
        
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_paths(self) -> np.ndarray:
        """
        Generate standard Brownian motion sample paths.
        
        Returns:
            np.ndarray: Array of shape (n_steps+1, n_paths) containing
                       sample paths where axis 0 is time and axis 1 is paths.
                       W(0) = 0 in the first row.
        
        Notes:
            Uses cumulative sum of random normal increments.
            dW(t) ~ N(0, sqrt(dt))
            
        Mathematical Implementation:
            W(t) = Σ_{i=1}^{t/dt} dW_i where dW_i ~ N(0, dt)
        
        Examples:
            >>> bm = BrownianMotion(T=1.0, n_steps=252, n_paths=10)
            >>> paths = bm.generate_paths()
            >>> # Check initial condition
            >>> assert np.allclose(paths[0, :], 0)
            >>> # Check final variance
            >>> assert np.isclose(np.var(paths[-1, :]), 1.0, atol=0.1)
        """
        # Generate random increments dW ~ N(0, sqrt(dt))
        dW = np.random.standard_normal((self.n_steps, self.n_paths)) * np.sqrt(self.dt)
        
        # Add initial condition W(0) = 0
        W = np.vstack([np.zeros(self.n_paths), dW])
        
        # Cumulative sum gives the Brownian path
        paths = np.cumsum(W, axis=0)
        
        return paths
    
    def generate_increments(self) -> np.ndarray:
        """
        Generate Brownian motion increments (dW).
        
        Returns:
            np.ndarray: Array of shape (n_steps, n_paths) of increments
                       where dW ~ N(0, dt)
        
        Notes:
            These increments are the building blocks for Itô processes.
            Use these when you need explicit control over increments.
        
        Examples:
            >>> bm = BrownianMotion(T=1.0, n_steps=252, n_paths=10)
            >>> dW = bm.generate_increments()
            >>> # Check mean is close to 0
            >>> assert np.abs(np.mean(dW)) < 0.05
            >>> # Check variance is close to dt
            >>> assert np.isclose(np.var(dW), bm.dt, atol=0.01)
        """
        dW = np.random.standard_normal((self.n_steps, self.n_paths)) * np.sqrt(self.dt)
        return dW
    
    def simulate_paths_advanced(
        self,
        correlation: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Generate correlated Brownian motions (multi-dimensional).
        
        Parameters:
            correlation (Optional[np.ndarray]): Correlation matrix of shape (n, n)
                                              where n is the number of processes.
                                              If None, returns single independent path.
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: 
                If correlation is None: single path array of shape (n_steps+1, n_paths)
                If correlation is provided: tuple of correlated paths
        
        Notes:
            Uses Cholesky decomposition to generate correlated increments:
            dW^corr = L @ dW where L = cholesky(correlation_matrix)
        
        Raises:
            ValueError: If correlation matrix is not positive definite
        
        Examples:
            >>> bm = BrownianMotion(T=1.0, n_steps=100, n_paths=1000)
            >>> corr = np.array([[1.0, 0.7], [0.7, 1.0]])
            >>> paths = bm.simulate_paths_advanced(corr)
            >>> # paths[0] and paths[1] are correlated
        """
        if correlation is None:
            return self.generate_paths()
        
        # Validate correlation matrix
        correlation = np.asarray(correlation, dtype=float)
        n_dim = correlation.shape[0]
        
        if correlation.shape != (n_dim, n_dim):
            raise ValueError(f"Correlation matrix must be square, got {correlation.shape}")
        
        # Check eigenvalues for positive definiteness
        eigenvalues = np.linalg.eigvals(correlation)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix is not positive definite")
        
        try:
            L = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite")
        
        # Generate independent increments
        dW_indep = np.random.standard_normal((self.n_steps, n_dim, self.n_paths)) * np.sqrt(self.dt)
        
        # Apply Cholesky factor to create correlation
        dW_corr = np.zeros_like(dW_indep)
        for i in range(self.n_steps):
            # Apply correlation: L @ Z where Z ~ N(0, I)
            dW_corr[i] = L @ dW_indep[i]
        
        # Generate paths for each dimension
        paths = tuple(
            np.vstack([np.zeros(self.n_paths), np.cumsum(dW_corr[:, d, :], axis=0)])
            for d in range(n_dim)
        )
        
        return paths
    
    @staticmethod
    def verify_properties(
        paths: np.ndarray,
        T: float = 1.0
    ) -> dict:
        """
        Verify properties of generated Brownian motion paths.
        
        Parameters:
            paths (np.ndarray): Generated paths of shape (n_steps+1, n_paths)
            T (float): Total time horizon
        
        Returns:
            dict: Dictionary containing verification results:
                - 'initial_value': Should be 0
                - 'mean_final': Mean of W(T), should be ~0
                - 'var_final': Variance of W(T), should be ~T
                - 'increments_independent': Bool for independence test
                - 'normality_test': Jarque-Bera p-value
        
        Examples:
            >>> bm = BrownianMotion(T=1.0, n_steps=252, n_paths=1000)
            >>> paths = bm.generate_paths()
            >>> props = BrownianMotion.verify_properties(paths, T=1.0)
            >>> print(f"Initial value: {props['initial_value']}")
            >>> print(f"Final variance: {props['var_final']:.2f}")
        """
        from scipy import stats
        
        results = {}
        
        # Check initial condition
        results['initial_value'] = np.mean(np.abs(paths[0, :]))
        
        # Check final time distribution
        W_T = paths[-1, :]
        results['mean_final'] = np.mean(W_T)
        results['var_final'] = np.var(W_T, ddof=1)
        results['expected_var'] = T
        
        # Check independence of increments
        increments = np.diff(paths, axis=0)
        corr_consecutive = np.corrcoef(increments[:-1].flatten(), increments[1:].flatten())[0, 1]
        results['increments_independent'] = np.abs(corr_consecutive) < 0.05
        
        # Normality test on increments
        jb_stat, jb_pval = stats.jarque_bera(increments.flatten())
        results['normality_test'] = jb_pval
        
        return results


class BridgeBrownianMotion(BrownianMotion):
    """
    Brownian Bridge: Brownian motion conditioned to return to 0 at time T.
    
    A Brownian bridge from (0,0) to (T,0) is a Brownian motion path
    constrained to start and end at specific values.
    
    Mathematical Definition:
        B(t) = W(t) - (t/T)*W(T)  for 0 ≤ t ≤ T
    
    where W(t) is standard Brownian motion.
    
    This is useful for generating sample paths with boundary conditions.
    """
    
    def generate_paths(self) -> np.ndarray:
        """
        Generate Brownian bridge paths from (0,0) to (T,0).
        
        Returns:
            np.ndarray: Array of shape (n_steps+1, n_paths) where:
                       B(0) = 0 and B(T) = 0
        
        Examples:
            >>> bridge = BridgeBrownianMotion(T=1.0, n_steps=252, n_paths=100)
            >>> paths = bridge.generate_paths()
            >>> assert np.allclose(paths[0, :], 0)
            >>> assert np.allclose(paths[-1, :], 0)
        """
        # Generate standard Brownian motion
        W = super().generate_paths()
        
        # Apply bridge transformation: B(t) = W(t) - (t/T)*W(T)
        times = np.linspace(0, self.T, self.n_steps + 1)
        bridge = W - (times[:, np.newaxis] / self.T) * W[-1, :]
        
        return bridge
