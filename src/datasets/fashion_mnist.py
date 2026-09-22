"""Fashion-MNIST data loading, preprocessing, and reproducible splitting."""

import hashlib
import json
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.core.seed import seed_worker


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_train_transform(config: Mapping[str, Any]) -> transforms.Compose:
    """Build the preprocessing pipeline used by the training subset."""
    return _build_base_transform(config)


def build_eval_transform(config: Mapping[str, Any]) -> transforms.Compose:
    """Build deterministic preprocessing for validation and test data."""
    return _build_base_transform(config)


def _build_base_transform(config: Mapping[str, Any]) -> transforms.Compose:
    options = config["dataset"]
    normalization = options.get("normalization", True)
    if not isinstance(normalization, bool):
        raise TypeError("normalization must be a boolean")

    steps = [transforms.ToTensor()]
    if normalization:
        # Fixed scaling avoids estimating statistics from held-out data.
        steps.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(steps)


def _validate_integer(
    value: Any, name: str, *, allow_zero: bool = False
) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _limit_splits(
    splits: dict[str, Dataset], limit: int | None
) -> dict[str, Dataset]:
    if limit is None:
        return splits
    _validate_integer(limit, "max_samples")
    return {
        name: Subset(dataset, range(min(limit, len(dataset))))
        for name, dataset in splits.items()
    }


def build_fashion_mnist(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create Fashion-MNIST loaders and reproducibility metadata."""
    options = config["dataset"]
    ratio = options.get("val_ratio", 0.1)
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0 < ratio < 1
    ):
        raise ValueError("val_ratio must be a number between zero and one")

    seed = _validate_integer(config.get("seed", 42), "seed", allow_zero=True)
    batch_size = _validate_integer(options.get("batch_size", 64), "batch_size")
    num_workers = _validate_integer(
        options.get("num_workers", 0), "num_workers", allow_zero=True
    )

    common = {
        "root": options.get("root", "data"),
        "download": options.get("download", True),
    }
    train_source = datasets.FashionMNIST(
        train=True, transform=build_train_transform(config), **common
    )
    validation_source = datasets.FashionMNIST(
        train=True, transform=build_eval_transform(config), **common
    )
    test_source = datasets.FashionMNIST(
        train=False, transform=build_eval_transform(config), **common
    )

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=split_generator).tolist()
    validation_size = int(len(train_source) * ratio)
    if not 0 < validation_size < len(train_source):
        raise ValueError("split must contain both training and validation samples")

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    splits: dict[str, Dataset] = {
        "train": Subset(train_source, train_indices),
        "val": Subset(validation_source, validation_indices),
        "test": test_source,
    }
    limit = options.get("max_samples")
    splits = _limit_splits(splits, limit)

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=name == "train",
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
        for name, dataset in splits.items()
    }

    split_payload = {"train": train_indices, "val": validation_indices}
    split_hash = hashlib.sha256(
        json.dumps(split_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "name": "fashion_mnist",
        "task": "classification",
        "num_classes": len(FASHION_MNIST_CLASSES),
        "class_names": FASHION_MNIST_CLASSES.copy(),
        "input_shape": [1, 28, 28],
        "split_seed": seed,
        "val_ratio": ratio,
        "split_sha256": split_hash,
        "sizes": {name: len(dataset) for name, dataset in splits.items()},
        "max_samples": limit,
        "normalization": options.get("normalization", True),
        "train_transform": repr(train_source.transform),
        "eval_transform": repr(validation_source.transform),
    }
    return {"loaders": loaders, "metadata": metadata}
