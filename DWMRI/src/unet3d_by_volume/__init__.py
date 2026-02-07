"""
unet3d_by_volume: 3D U-Net for DWMRI by-volume denoising.

By-volume paradigm: X-1 noisy volumes as input, predict the denoised jth volume
(the one excluded from input). For X total volumes, input has X-1 channels.
"""

__version__ = "0.1.0"
