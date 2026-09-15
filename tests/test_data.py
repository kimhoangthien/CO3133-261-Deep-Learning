import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from src.datasets.registry import build_dataset


def test_loaders_and_split(config, fake_fashion):
    a, b = build_dataset(config), build_dataset(config)
    train, val, test = [a["loaders"][key] for key in ("train", "val", "test")]
    assert [len(x.dataset) for x in (train,val,test)] == [90,10,20]
    assert val.dataset.indices == b["loaders"]["val"].dataset.indices
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert len(set(train.dataset.indices + val.dataset.indices)) == 100
    assert train.dataset.dataset.train and not test.dataset.train
    images, labels = next(iter(train))
    assert images.shape == (8,1,28,28) and labels.dtype == torch.int64
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
