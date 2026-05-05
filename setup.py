from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="derivatives-toolkit",
    version="0.1.0",
    author="Frank",
    author_email="jungle51753@gmail.com",
    description="A Python library for derivatives pricing and risk analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Frank5753/derivatives-toolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
)
