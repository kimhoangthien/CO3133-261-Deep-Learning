import pytest
import torch
from src.core.config import load_config


@pytest.fixture
def config(tmp_path):
    value = load_config("configs/a1/linear.yaml")
    value["device"] = "cpu"
    value["training"]["epochs"] = 1
    value["dataset"].update(batch_size=8, num_workers=0)
    for kind in ("checkpoint", "logging", "results"):
        value[kind]["root"] = str(tmp_path / kind)
    return value


@pytest.fixture
def fake_fashion(monkeypatch):
    class FakeFashion:
        classes = [str(i) for i in range(10)]
        def __init__(self, root, train, download, transform):
            self.train = train
            self.transform = transform
        def __len__(self):
            return 100 if self.train else 20
        def __getitem__(self, index):
            import numpy as np
            from PIL import Image
            image = Image.fromarray(np.full((28, 28), index % 256, dtype=np.uint8))
            return self.transform(image), index % 10
    monkeypatch.setattr("src.datasets.fashion_mnist.datasets.FashionMNIST", FakeFashion)
