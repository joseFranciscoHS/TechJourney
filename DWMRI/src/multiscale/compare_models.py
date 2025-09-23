#!/usr/bin/env python3
"""
Configuration testing script for MultiScaleDetailNet.
This script allows testing different loss functions and training configurations.
"""

import logging
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import main


def run_loss_comparison():
    """Run comparison between different loss functions"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== MultiScaleDetailNet Loss Function Comparison ===")

    # Configuration 1: MultiScaleDetailNet with L1Loss
    logger.info("\n--- Configuration 1: MultiScaleDetailNet + L1Loss ---")
    try:
        main(
            dataset="dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            use_edge_aware_loss=False,
            use_mixed_precision=True,
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
            use_edge_aware_loss=True,
            use_mixed_precision=True,
            accumulation_steps=1,
        )
        logger.info("Configuration 2 completed successfully")
    except Exception as e:
        logger.error(f"Configuration 2 failed: {e}")

    # Configuration 3: MultiScaleDetailNet with EdgeAwareLoss + Gradient Accumulation
    logger.info(
        "\n--- Configuration 3: MultiScaleDetailNet + EdgeAwareLoss + Gradient Accumulation ---"
    )
    try:
        main(
            dataset="dbrain",
            train=True,
            reconstruct=True,
            generate_images=True,
            use_edge_aware_loss=True,
            use_mixed_precision=True,
            accumulation_steps=4,
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
        use_edge_aware_loss=True,
        use_mixed_precision=True,
        accumulation_steps=1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MultiScaleDetailNet Configuration Testing"
    )
    parser.add_argument(
        "--mode",
        choices=["comparison", "single"],
        default="single",
        help="Run mode: comparison (all configs) or single (EdgeAwareLoss only)",
    )

    args = parser.parse_args()

    if args.mode == "comparison":
        run_loss_comparison()
    else:
        run_single_config()
