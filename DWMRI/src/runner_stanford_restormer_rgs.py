"""
Stanford HARDI: Restormer hybrid RGS few-volume train + reconstruct.

Thin wrapper around ``restormer_hybrid_rgs.run_stanford_fewvol`` (see that module
for CLI flags: ``--force-train``, ``--skip-train``, ``--skip-reconstruct``,
``--no-wandb``, ``--no-images``).

Run from ``DWMRI/src`` (same layout as ``runner_stanford.py`` for DRCNet)::

    python runner_stanford_restormer_rgs.py
    python runner_stanford_restormer_rgs.py --no-wandb --skip-train
"""

from restormer_hybrid_rgs.run_stanford_fewvol import main

main()
