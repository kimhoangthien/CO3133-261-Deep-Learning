from .a1.linear import LinearClassifier
from .a1.mlp import MLP
from .a1.cnn import CNNClassifier


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

# TODO(A1-M2-INTEGRATION): After owner implementations/tests are reviewed, add cnn/rnn/transformer factories and registrations that consume every agreed model.parameters key.
MODELS = {
    ("a1", "linear"): build_linear,
    ("a1", "mlp"): build_mlp,
    ("a1", "cnn"): build_cnn
}


def build_model(config, metadata):
    key = (config["assignment"], config["model"]["name"])
    if key not in MODELS:
        raise NotImplementedError(f"Model {key} is not implemented. A1 CNN/RNN/Transformer are TODO.")
    return MODELS[key](config, metadata)
