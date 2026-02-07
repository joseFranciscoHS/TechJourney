"""
DRCNet-S2S: MD-S2S style (spatial–angular hybrid) DWMRI denoising.

DRCNet with J-invariance at the pixel level: Bernoulli masks occlude pixels
across all volumes; the network predicts occluded pixels from visible ones.
Combines angular redundancy (Patch2Self) with spatial redundancy (Self2Self).

Data layout: (Vols, X, Y, Z). See J_invariance_DWMRI_denoising_report.md.
"""

__version__ = "0.1.0"
