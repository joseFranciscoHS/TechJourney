"""
P2S: Patch2Self denoising pipeline for full DWMRI reconstruction.

Uses DIPY's classical Patch2Self method to denoise diffusion-weighted volumes
(bvals > b0_threshold) while leaving b0 volumes intact.
"""

from p2s.run import main

__version__ = "0.1.0"
__all__ = ["main"]
