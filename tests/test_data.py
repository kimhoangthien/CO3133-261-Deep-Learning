import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.datasets import FashionMNIST
from src.datasets.fashion_mnist import (
    FASHION_MNIST_CLASSES,
    build_eval_transform,
    build_train_transform,
)
from src.datasets.registry import build_dataset


def test_loaders_and_split(config, fake_fashion):
    a, b = build_dataset(config), build_dataset(config)
    train, val, test = [a["loaders"][key] for key in ("train", "val", "test")]
    assert [len(x.dataset) for x in (train,val,test)] == [90,10,20]
    assert val.dataset.indices == b["loaders"]["val"].dataset.indices
    assert train.dataset.indices == b["loaders"]["train"].dataset.indices
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert len(set(train.dataset.indices + val.dataset.indices)) == 100
    assert train.dataset.dataset.train and not test.dataset.train
    images, labels = next(iter(train))
    assert images.shape == (8,1,28,28) and labels.dtype == torch.int64
    assert labels.shape == (images.size(0),)
    assert a["metadata"]["num_classes"] == 10
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    config["seed"] += 1
    assert build_dataset(config)["metadata"]["split_sha256"] != a["metadata"]["split_sha256"]


def test_normalization_and_invalid_ratio(config, fake_fashion):
    config["dataset"]["normalization"] = False
    raw = build_dataset(config)["loaders"]["test"].dataset[0][0]
    config["dataset"]["normalization"] = True
    scaled = build_dataset(config)["loaders"]["test"].dataset[0][0]
    assert torch.allclose(scaled, (raw-0.5)/0.5)
    config["dataset"]["val_ratio"] = 0
    with pytest.raises(ValueError):
        build_dataset(config)


def test_class_names_match_torchvision():
    # The names are hard-coded here but label every confusion matrix and error
    # analysis, so silent drift from the real dataset order must fail loudly.
    assert FASHION_MNIST_CLASSES == list(FashionMNIST.classes)


def test_evaluation_preprocessing_is_deterministic(config):
    # Augmentation belongs to the training pipeline only; validation and test
    # must return the same tensor every time they see the same image. The image
    # is asymmetric so that a flip or crop cannot pass unnoticed, and repeating
    # the call keeps a random transform from escaping on a lucky draw.
    image = Image.fromarray(np.arange(784, dtype=np.uint8).reshape(28, 28))
    transform = build_eval_transform(config)
    expected = transform(image)
    assert all(torch.equal(transform(image), expected) for _ in range(8))


def test_normalization_flag_must_be_boolean(config):
    config["dataset"]["normalization"] = 1
    for builder in (build_train_transform, build_eval_transform):
        with pytest.raises(TypeError):
            builder(config)


def test_metadata_records_preprocessing_and_sizes(config, fake_fashion):
    metadata = build_dataset(config)["metadata"]
    assert metadata["class_names"] == FASHION_MNIST_CLASSES
    assert metadata["sizes"] == {"train": 90, "val": 10, "test": 20}
    assert metadata["split_seed"] == config["seed"] and metadata["max_samples"] is None
    # Evaluation refuses a checkpoint whose preprocessing no longer matches.
    assert "Normalize" in metadata["train_transform"]
    assert metadata["eval_transform"] == metadata["train_transform"]


def test_max_samples_limits_every_split(config, fake_fashion):
    config["dataset"]["max_samples"] = 5
    metadata = build_dataset(config)["metadata"]
    assert metadata["sizes"] == {"train": 5, "val": 5, "test": 5}
    assert metadata["max_samples"] == 5


@pytest.mark.parametrize("field, value", [
    ("batch_size", 0),
    ("num_workers", -1),
    ("max_samples", 0),
    ("val_ratio", 1),
    ("val_ratio", True),
])
def test_invalid_dataset_options_are_rejected(config, fake_fashion, field, value):
    config["dataset"][field] = value
    with pytest.raises(ValueError):
        build_dataset(config)


def test_invalid_seed_is_rejected(config, fake_fashion):
    config["seed"] = -1
    with pytest.raises(ValueError):
        build_dataset(config)
