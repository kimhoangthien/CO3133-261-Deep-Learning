from pathlib import Path
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("assignment", "task", "experiment_name", "dataset", "model", "training"):
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    for key in ("assignment", "experiment_name"):
        value = config[key]
        if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError(f"{key} must be a simple identifier")
    return config


def output_dir(config, kind):
    path = Path(config[kind]["root"]) / config["assignment"] / config["experiment_name"]
    path.mkdir(parents=True, exist_ok=True)
    return path
