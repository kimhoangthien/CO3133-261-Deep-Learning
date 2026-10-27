import pytest
import torch
from src.core.config import load_config
from src.models.a1.rnn import RNNClassifier
from src.models.registry import MODELS, build_model

# Discovered from the registry so new A1 models are covered once registered.
A1_MODELS = sorted(name for assignment, name in MODELS if assignment == "a1")
METADATA = {"input_shape": [1, 28, 28], "num_classes": 10}

# TODO(A1-M2, Khoa): Add CNN forward tests for [B, 10], finite raw logits and CrossEntropyLoss/backward, plus configurable class count and invalid arguments.
# TODO(A1-M2, Antoine): Test token/projection/attention shapes and Transformer forward [B, 10], finite raw logits, CrossEntropyLoss/backward and invalid dimensions.
# TODO(A1-M2-INTEGRATION): Once all models are ready, extend registry-driven coverage using valid agreed parameters for each model; retain Linear/MLP coverage.


@pytest.mark.parametrize("name", A1_MODELS)
def test_model_contract(config, name):
    torch.manual_seed(0)
    config["model"] = {"name": name}
    model = build_model(config, METADATA)
    inputs = torch.randn(4, 1, 28, 28)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    output = model(inputs)
    assert output.shape == (4, METADATA["num_classes"])
    assert torch.isfinite(output).all()
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


def test_rnn_uses_each_image_row_as_a_timestep():
    model = RNNClassifier(input_shape=METADATA["input_shape"], num_classes=10)
    inputs = torch.randn(3, 1, 28, 28)
    observed_sequences = []
    hook = model.recurrent.register_forward_pre_hook(
        lambda _module, arguments: observed_sequences.append(arguments[0].detach())
    )
    try:
        model(inputs)
    finally:
        hook.remove()

    assert len(observed_sequences) == 1
    assert observed_sequences[0].shape == (3, 28, 28)
    assert torch.equal(observed_sequences[0], inputs[:, 0, :, :])


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
@pytest.mark.parametrize("batch_size", [8, 1])
def test_rnn_forward_loss_and_backward(config, recurrent_type, batch_size):
    config["model"] = {
        "name": "rnn",
        "parameters": {
            "recurrent_type": recurrent_type,
            "input_size": 28,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
    }
    model = build_model(config, METADATA)
    inputs = torch.randn(batch_size, 1, 28, 28)
    labels = torch.arange(batch_size) % METADATA["num_classes"]

    logits = model(inputs)
    assert logits.shape == (batch_size, METADATA["num_classes"])
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits.sum(1), torch.ones(batch_size))

    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_rnn_config_builds_model():
    rnn_config = load_config("configs/a1/rnn.yaml")
    model = build_model(rnn_config, METADATA)

    assert isinstance(model, RNNClassifier)
    assert isinstance(model.recurrent, torch.nn.LSTM)
    assert model.input_size == rnn_config["model"]["parameters"]["input_size"]
    assert model.hidden_size == rnn_config["model"]["parameters"]["hidden_size"]
    assert model.num_layers == rnn_config["model"]["parameters"]["num_layers"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"input_size": 27},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"recurrent_type": "rnn"},
        {"dropout": 1.0},
        {"num_layers": 1, "dropout": 0.1},
    ],
)
def test_rnn_rejects_invalid_recurrent_arguments(parameters):
    with pytest.raises(ValueError):
        RNNClassifier(
            input_shape=METADATA["input_shape"],
            num_classes=METADATA["num_classes"],
            **parameters,
        )


@pytest.mark.parametrize(
    ("input_shape", "num_classes"),
    [([1, 28], 10), ([3, 28, 28], 10), ([1, 28, 28], 1)],
)
def test_rnn_rejects_invalid_input_shape_or_class_count(input_shape, num_classes):
    with pytest.raises(ValueError):
        RNNClassifier(input_shape=input_shape, num_classes=num_classes)


def test_rnn_rejects_input_with_unexpected_channel_or_spatial_shape():
    model = RNNClassifier(input_shape=METADATA["input_shape"], num_classes=10)

    with pytest.raises(ValueError, match="inputs must have shape"):
        model(torch.randn(1, 3, 28, 28))
    with pytest.raises(ValueError, match="inputs must have shape"):
        model(torch.randn(1, 1, 27, 28))


def test_unknown_model_is_rejected(config):
    config["model"] = {"name": "not_a_model"}
    with pytest.raises(NotImplementedError):
        build_model(config, METADATA)
