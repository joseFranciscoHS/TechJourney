# Noise sweep: train and validate over different noise levels and distributions (same-condition).
# Uses DRCNet-hybrid; each run gets its own checkpoint/metrics dir and wandb run name/tags.
from drcnet_hybrid_rgs.run import main

# Noise sweep: Restormer-hybrid; same conditions as DRCNet sweep.
# from restormer_hybrid.run import main


NOISE_SWEEP = [
    # (0.05, "rician"),
    (0.1, "rician"),
    # (0.15, "rician"),
    # (0.05, "gaussian"),
    # (0.1, "gaussian"),
    # (0.15, "gaussian"),
    # (0.05, "noncentral_chi"),
    # (0.1, "noncentral_chi"),
    # (0.15, "noncentral_chi"),
]

for sigma, nt in NOISE_SWEEP:
    print(f"--- Noise sweep: sigma={sigma}, type={nt} ---")
    try:
        main(
            "dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            noise_sigma=sigma,
            noise_type=nt,
        )
    except Exception as e:
        print(f"Condition sigma={sigma} type={nt} failed: {e}")
        continue
print("Noise sweep finished.")

"""
main(
    "dbrain",
    train=True,
    reconstruct=True,
    generate_images=True,
    noise_sigma=0.1,
    noise_type="rician",
)
"""
