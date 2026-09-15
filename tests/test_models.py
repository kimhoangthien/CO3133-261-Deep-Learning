import pytest
import torch
from src.models.registry import build_model


def test_linear_logits_and_optimizer(config):
    model = build_model(config, {"input_shape": [1,28,28], "num_classes":10})
    inputs = torch.randn(4,1,28,28)
    before = model.layers[1].weight.detach().clone()
    output = model(inputs)
    assert output.shape == (4,10)
    assert torch.allclose(output, torch.nn.functional.linear(inputs.flatten(1), model.layers[1].weight, model.layers[1].bias))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([0,1,2,3]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.layers[1].weight)


def test_todo_model(config):
    config["model"]["name"] = "cnn"
    with pytest.raises(NotImplementedError):
        build_model(config, {})
