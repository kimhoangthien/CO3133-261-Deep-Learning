from .a1.linear import LinearClassifier
from .a1.mlp import MLP
from .a1.rnn import RNNClassifier


def build_linear(config, metadata):
    return LinearClassifier(metadata["input_shape"], metadata["num_classes"], **config["model"].get("parameters", {}))

def build_mlp(config, metadata):
    parameters = config["model"].get("parameters", {})
    return MLP(
        input_shape=metadata["input_shape"],
        hidden_size=parameters.get("hidden_size", 128),
        num_classes=metadata["num_classes"],
    )

def build_rnn(config, metadata):
    return RNNClassifier(
        input_shape=metadata["input_shape"],
        num_classes=metadata["num_classes"],
        **config["model"].get("parameters", {}),
    )


# TODO(A1-M2-INTEGRATION): Add CNN and Transformer factories after their owner implementations are reviewed.
MODELS = {
    ("a1", "linear"): build_linear,
    ("a1", "mlp"): build_mlp,
    ("a1", "rnn"): build_rnn,
}


def build_model(config, metadata):
    key = (config["assignment"], config["model"]["name"])
    if key not in MODELS:
        raise NotImplementedError(f"Model {key} is not implemented. A1 CNN/Transformer are TODO.")
    return MODELS[key](config, metadata)
