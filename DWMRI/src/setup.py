#!/usr/bin/env python3
"""
Setup script for DWMRI processing package.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dwmri-processing",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DWMRI processing tools including MDS2S, P2S, and DRCNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dwmri-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "PyYAML>=5.4.0",
        "munch>=2.5.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "mds2s=mds2s.run:main",
        ],
    },
)
