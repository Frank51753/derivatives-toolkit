"""
Setup configuration for derivatives-toolkit package.

A comprehensive Python library for derivatives pricing and risk management,
based on Kerry Back's "A Course in Derivative Securities".

Features:
    - European option pricing (Black-Scholes)
    - Stochastic process simulation (Brownian motion, GBM, Itô processes)
    - Greeks calculation and risk analysis
    - Monte Carlo simulation (planned)
    - Exotic options pricing (planned)
    - Fixed income instruments (planned)

Installation:
    pip install -e .
    
For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A Python library for derivatives pricing and risk management"

setup(
    name="derivatives-toolkit",
    version="0.3.0",
    author="Frank",
    author_email="jungle51753@gmail.com",
    description="Comprehensive Python library for derivatives pricing and risk management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YourUsername/derivatives-toolkit",
    project_urls={
        "Documentation": "https://github.com/YourUsername/derivatives-toolkit",
        "Source Code": "https://github.com/YourUsername/derivatives-toolkit",
        "Issue Tracker": "https://github.com/YourUsername/derivatives-toolkit/issues",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=[
        # Numerical computing
        "numpy>=1.20.0",           # Numerical arrays and operations
        "scipy>=1.7.0",            # Scientific computing (integration, optimization, distributions)
        
        # Data manipulation
        "pandas>=1.3.0",           # Data frames and time series
        
        # Visualization
        "matplotlib>=3.4.0",       # Plotting and visualization
        
        # Machine learning (for volatility models)
        "scikit-learn>=0.24.0",    # Machine learning utilities
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            # Testing
            "pytest>=6.2.0",                  # Testing framework
            "pytest-cov>=2.12.0",             # Code coverage for pytest
            
            # Code quality
            "black>=21.0",                    # Code formatter
            "flake8>=3.9.0",                  # Linter
            "isort>=5.9.0",                   # Import sorter
            "mypy>=0.910",                    # Static type checker
            
            # Documentation
            "sphinx>=4.0.0",                  # Documentation generator
            "sphinx-rtd-theme>=1.0.0",        # ReadTheDocs theme for Sphinx
            
            # Jupyter notebooks
            "jupyter>=1.0.0",                 # Jupyter notebook server
            "ipython>=7.0.0",                 # IPython kernel
            "notebook>=6.0.0",                # Jupyter notebook interface
        ],
        
        "scientific": [
            # Additional scientific tools
            "sympy>=1.9",                     # Symbolic mathematics
            "statsmodels>=0.12.0",            # Statistical modeling
        ],
        
        "visualization": [
            "seaborn>=0.11.0",                # Statistical data visualization
            "plotly>=5.0.0",                  # Interactive plots
        ],
        
        "all": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "notebook>=6.0.0",
            "sympy>=1.9",
            "statsmodels>=0.12.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    
    # Package data
    include_package_data=True,
    
    # Entry points (for command-line tools, if any)
    entry_points={
        "console_scripts": [
            # Future CLI tools could go here
            # "derivatives-cli=derivatives.cli:main",
        ],
    },
    
    # Additional metadata
    keywords=[
        "derivatives",
        "options",
        "pricing",
        "quantitative-finance",
        "risk-management",
        "black-scholes",
        "monte-carlo",
        "stochastic-calculus",
        "financial-engineering",
    ],
    zip_safe=False,
)
