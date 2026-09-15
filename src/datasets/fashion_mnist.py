import hashlib
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.core.seed import seed_worker


def build_fashion_mnist(config):
    options = config["dataset"]
    ratio = options.get("val_ratio", 0.1)
    if not 0 < ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    steps = [transforms.ToTensor()]
    if options.get("normalization", True):
        # Fixed scaling avoids estimating statistics from validation or test data.
        steps.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(steps)
    common = dict(root=options.get("root", "data"), download=options.get("download", True), transform=transform)
    training = datasets.FashionMNIST(train=True, **common)
    test = datasets.FashionMNIST(train=False, **common)
    seed = config.get("seed", 42)
    indices = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(len(training) * ratio)
    if not 0 < n_val < len(training):
        raise ValueError("Split must contain training and validation samples")
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    splits = {"train": Subset(training, train_indices), "val": Subset(training, val_indices), "test": test}
    limit = options.get("max_samples")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("max_samples must be a positive integer")
        splits = {key: Subset(value, range(min(limit, len(value)))) for key, value in splits.items()}
    loaders = {key: DataLoader(value, batch_size=options.get("batch_size", 64),
                              num_workers=options.get("num_workers", 0), shuffle=key == "train",
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed)) for key, value in splits.items()}
    metadata = {"name": "fashion_mnist", "task": "classification", "num_classes": 10,
                "class_names": list(training.classes), "input_shape": [1, 28, 28],
                "split_seed": seed, "val_ratio": ratio,
                "split_sha256": hashlib.sha256(str(indices).encode()).hexdigest(),
                "sizes": {key: len(value) for key, value in splits.items()},
                "max_samples": limit, "normalization": options.get("normalization", True)}
    return {"loaders": loaders, "metadata": metadata}
