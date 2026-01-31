"""
DRCNet-S2S: DRCNet run with the S2S (Self2Self) framework for DWMRI denoising.

One volume at a time with Bernoulli-masked pixels; denoising is done per volume.
Data layout: (Z, Vols, X, Y).
"""

__version__ = "0.1.0"
