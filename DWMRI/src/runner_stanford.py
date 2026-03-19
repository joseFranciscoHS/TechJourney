# Noise sweep: train and validate over different noise levels and distributions (same-condition).
# Uses DRCNet-hybrid; each run gets its own checkpoint/metrics dir and wandb run name/tags.
from drcnet_hybrid.run import main
# Noise sweep: Restormer-hybrid; same conditions as DRCNet sweep.
# from restormer_hybrid.run import main

main(
    "stanford",
    train=True,
    reconstruct=True,
    generate_images=True,
)
