from .a1.linear import LinearClassifier


def build_linear(config, metadata):
    return LinearClassifier(metadata["input_shape"], metadata["num_classes"], **config["model"].get("parameters", {}))


MODELS = {("a1", "linear"): build_linear}


def build_model(config, metadata):
    key = (config["assignment"], config["model"]["name"])
    if key not in MODELS:
        raise NotImplementedError(f"Model {key} is not implemented. A1 MLP/CNN/RNN/Transformer are TODO.")
    return MODELS[key](config, metadata)
