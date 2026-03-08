import logging

import yaml
from munch import munchify


def noise_path_segment(noise_type, noise_sigma):
    """
    Build a path segment that identifies noise distribution and level.
    Used in checkpoint_dir, loss_dir, metrics_dir, images_dir so runs
    with different noise settings do not overwrite each other.

    Examples: noise_rician_sigma_0.1, noise_gaussian_sigma_0.15, noise_ncchi_sigma_0.1
    """
    nt = (noise_type or "rician").lower().strip()
    alias = "ncchi" if nt == "noncentral_chi" else nt
    return f"noise_{alias}_sigma_{noise_sigma}"


def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info("Configuration file loaded successfully")
        logging.debug(f"Raw config keys: {list(config.keys()) if config else 'None'}")

        munchified_config = munchify(config)
        logging.info("Configuration converted to Munch object")
        return munchified_config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        raise
