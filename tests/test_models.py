import pytest
import torch
from src.models.registry import MODELS, build_model

# Discovered from the registry so new A1 models are covered once registered.
A1_MODELS = sorted(name for assignment, name in MODELS if assignment == "a1")
METADATA = {"input_shape": [1, 28, 28], "num_classes": 10}


@pytest.mark.parametrize("name", A1_MODELS)
def test_model_contract(config, name):
    torch.manual_seed(0)
    config["model"] = {"name": name}
    model = build_model(config, METADATA)
    inputs = torch.randn(4, 1, 28, 28)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    output = model(inputs)
    assert output.shape == (4, METADATA["num_classes"])
    # Raw logits: a model applying softmax itself would make every row sum to one.
    assert not torch.allclose(output.sum(1), torch.ones(4))
    loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_linear_is_a_plain_affine_map(config):
    config["model"] = {"name": "linear"}
    model = build_model(config, METADATA)
    inputs = torch.randn(4, 1, 28, 28)
    weight, bias = model.layers[1].weight, model.layers[1].bias
    assert torch.allclose(model(inputs), torch.nn.functional.linear(inputs.flatten(1), weight, bias))


def test_unknown_model_is_rejected(config):
    config["model"] = {"name": "not_a_model"}
    with pytest.raises(NotImplementedError):
        build_model(config, METADATA)
