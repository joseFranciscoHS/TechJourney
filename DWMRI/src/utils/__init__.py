"""
Utility functions for DWMRI processing.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the entire repository.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path (default: None, uses timestamped file)
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    logging.info(
        f"Logging configured - level: {logging.getLevelName(log_level)}, file: {log_file_path}"
    )
    return log_file_path
