from .a1.linear import LinearClassifier
from .a1.mlp import MLP
from .a1.cnn import CNNClassifier
from .a1.rnn import RNNClassifier


def build_linear(config, metadata):
    return LinearClassifier(metadata["input_shape"], metadata["num_classes"], **config["model"].get("parameters", {}))

def build_mlp(config, metadata):
    parameters = config["model"].get("parameters", {})
    return MLP(
        input_shape=metadata["input_shape"],
        hidden_size=parameters.get("hidden_size", 256),
        num_classes=metadata["num_classes"],
    )

def build_cnn(config, metadata):
    parameters = config["model"].get("parameters", {})
    return CNNClassifier(
        input_shape=metadata["input_shape"],
        num_classes=metadata["num_classes"],
        base_channels=parameters.get("base_channels", 32),
        fc_size=parameters.get("fc_size", 128),
        dropout_rate=parameters.get("dropout_rate", 0.5)
    )


def build_rnn(config, metadata):
    return RNNClassifier(
        input_shape=metadata["input_shape"],
        num_classes=metadata["num_classes"],
        **config["model"].get("parameters", {}),
    )


# TODO(A1-M2-INTEGRATION): Add a Transformer factory after its owner implementation is reviewed.
MODELS = {
    ("a1", "linear"): build_linear,
    ("a1", "mlp"): build_mlp,
    ("a1", "cnn"): build_cnn,
    ("a1", "rnn"): build_rnn,
}


def build_model(config, metadata):
    key = (config["assignment"], config["model"]["name"])
    if key not in MODELS:
        raise NotImplementedError(f"Model {key} is not implemented. A1 Transformer is TODO.")
    return MODELS[key](config, metadata)
