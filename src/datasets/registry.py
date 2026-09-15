from .fashion_mnist import build_fashion_mnist

DATASETS = {"fashion_mnist": build_fashion_mnist}


def build_dataset(config):
    name = config["dataset"]["name"]
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS)}")
    return DATASETS[name](config)
