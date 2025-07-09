import os

from utils.utils import load_config


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    settings = load_config(config_path)
    print(f"Loaded configuration: {settings}")
    
    # TODO: Add your main execution logic here
    # Example:
    # from .model import YourModel
    # from .data import YourDataLoader
    # from .fit import train_model
    
    # model = YourModel(settings.model)
    # data_loader = YourDataLoader(settings.data)
    # train_model(model, data_loader, settings.train)


if __name__ == "__main__":
    main()
    