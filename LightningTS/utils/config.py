import yaml

from typing import Any

class ConfigUtils:

    @staticmethod
    def load_config(config_dir: str) -> dict[str, Any]:
        try:
            with open(config_dir, "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            raise e
        return config

    @staticmethod
    def save_config(config, output_dir: str) -> None:
        try:
            with open(output_dir, "w") as file:
                yaml.dump(config, file)
        except Exception as e:
            raise e