#!/usr/bin/env python3
"""
Comparison script for testing MultiScaleDetailNet vs DenoiserNet.
This script allows easy switching between architectures and loss functions.
"""

import logging
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import main


def run_comparison():
    """Run comparison between different model configurations"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== DWMRI Model Comparison ===")

    # Configuration 1: Original DenoiserNet with L1Loss
    logger.info("\n--- Configuration 1: Original DenoiserNet + L1Loss ---")
    try:
        main(
            dataset="dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            use_multiscale_model=False,
            use_edge_aware_loss=False,
            use_mixed_precision=False,
            accumulation_steps=1,
        )
        logger.info("Configuration 1 completed successfully")
    except Exception as e:
        logger.error(f"Configuration 1 failed: {e}")

    # Configuration 2: MultiScaleDetailNet with EdgeAwareLoss
    logger.info("\n--- Configuration 2: MultiScaleDetailNet + EdgeAwareLoss ---")
    try:
        main(
            dataset="dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            use_multiscale_model=True,
            use_edge_aware_loss=True,
            use_mixed_precision=True,
            accumulation_steps=1,
        )
        logger.info("Configuration 2 completed successfully")
    except Exception as e:
        logger.error(f"Configuration 2 failed: {e}")

    # Configuration 3: MultiScaleDetailNet with L1Loss (for comparison)
    logger.info("\n--- Configuration 3: MultiScaleDetailNet + L1Loss ---")
    try:
        main(
            dataset="dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            use_multiscale_model=True,
            use_edge_aware_loss=False,
            use_mixed_precision=True,
            accumulation_steps=1,
        )
        logger.info("Configuration 3 completed successfully")
    except Exception as e:
        logger.error(f"Configuration 3 failed: {e}")

    logger.info("\n=== Comparison completed ===")


def run_single_config():
    """Run a single configuration for testing"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== Running MultiScaleDetailNet with EdgeAwareLoss ===")

    main(
        dataset="dbrain",
        train=True,
        reconstruct=True,
        generate_images=True,
        use_multiscale_model=True,
        use_edge_aware_loss=True,
        use_mixed_precision=True,
        accumulation_steps=1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DWMRI Model Comparison")
    parser.add_argument(
        "--mode",
        choices=["comparison", "single"],
        default="single",
        help="Run mode: comparison (all configs) or single (MultiScaleDetailNet only)",
    )

    args = parser.parse_args()

    if args.mode == "comparison":
        run_comparison()
    else:
        run_single_config()
