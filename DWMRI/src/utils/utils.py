import logging

import yaml
from munch import munchify


def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info("Configuration file loaded successfully")
        logging.debug(
            f"Raw config keys: {list(config.keys()) if config else 'None'}"
        )

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
        raise
