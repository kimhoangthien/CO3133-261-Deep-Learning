from pathlib import Path
import json
import logging
import math
import pytest
import torch
from src.train import run as train
from src.evaluate import run as evaluate
from src.core.checkpoint import load_checkpoint
from src.models.registry import build_model
from src.tasks.classification.metrics import classification_metrics
from src.core.config import load_config
from src.datasets.registry import build_dataset
from src.tasks.classification.trainer import fit, train_one_epoch
from src.tasks.classification.evaluator import validate
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize("model_name,parameters", [("linear", 7850), ("mlp", 101770)])
def test_pipeline_and_checkpoint(config, fake_fashion, model_name, parameters):
    model_config = load_config(f"configs/a1/{model_name}.yaml")
    config["model"] = model_config["model"]
    config["experiment_name"] = model_name
    result = train(config)
    assert len(result["history"]) == 1
    row = result["history"][0]
    assert row["epoch"] == 1
    for phase in ("train", "val"):
        assert math.isfinite(row[f"{phase}_loss"])
        assert 0 <= row[f"{phase}_accuracy"] <= 1
    history_path = Path(config["results"]["root"]) / f"a1/{model_name}/training.json"
    assert json.loads(history_path.read_text())["history"] == result["history"]
    # Closed log handles must allow Windows to remove completed-run logs.
    log_path = Path(config["logging"]["root"]) / f"a1/{model_name}/train.log"
    log_path.unlink()
    saved = load_checkpoint(result["checkpoint"])
    assert saved["epoch"] == 1 and saved["optimizer_state"]["state"]
    assert saved["config"] == config
    assert saved["best_metric"] == row["val_accuracy"]
    model = build_model(config, saved["metadata"])
    load_checkpoint(result["checkpoint"], model)
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in saved["model_state"].items())
    metrics = evaluate(config, result["checkpoint"])["metrics"]
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["macro_f1"] <= 1
    assert metrics["num_parameters"] == parameters and metrics["num_samples"] == 20
    assert Path(result["checkpoint"]).exists()
    config["dataset"]["val_ratio"] = 0.2
    with pytest.raises(ValueError, match="dataset/split"):
        evaluate(config, result["checkpoint"])


def test_metrics_known_values():
    result = classification_metrics(torch.tensor([0,0,1,1]), torch.tensor([0,1,1,1]), 2)
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2/3+4/5)/2)
    assert result["confusion_matrix"] == [[1,1],[0,2]]


@pytest.mark.parametrize("model_name", ["linear", "mlp"])
def test_training_updates_parameters_and_validation_is_read_only(model_name):
    torch.manual_seed(42)
    config = load_config(f"configs/a1/{model_name}.yaml")
    model = build_model(config, {"input_shape": [1, 28, 28], "num_classes": 10})
    inputs, labels = torch.randn(8, 1, 28, 28), torch.arange(8)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=8)
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(inputs)
    assert logits.shape == (8, 10)
    assert torch.isfinite(criterion(logits, labels))
    states = []
    hook = model.register_forward_pre_hook(
        lambda module, args: states.append((module.training, torch.is_grad_enabled())))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.eval()
    trained = train_one_epoch(model, loader, torch.optim.Adam(model.parameters()), criterion, "cpu")
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
    after = {key: value.clone() for key, value in model.state_dict().items()}
    gradients = [p.grad.clone() for p in model.parameters()]
    validated = validate(model, loader, torch.device("cpu"), 10, criterion)
    hook.remove()
    assert states == [(True, True), (False, False)]
    assert all(torch.equal(after[key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(old, p.grad) for old, p in zip(gradients, model.parameters()))
    for result in (trained, validated):
        assert math.isfinite(result["loss"])
        assert 0 <= result["accuracy"] <= 1


def test_epoch_metrics_weight_uneven_batches():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    inputs = torch.tensor([[4., 0.], [0., 4.], [4., 0.]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    criterion = torch.nn.CrossEntropyLoss()
    expected = criterion(model(inputs), labels).item()
    # Freeze updates so both epochs can be compared with one full-dataset loss.
    trained = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0), criterion, "cpu")
    validated = validate(model, loader, torch.device("cpu"), 2, criterion)
    for result in (trained, validated):
        assert result["loss"] == pytest.approx(expected)
        assert result["accuracy"] == pytest.approx(2 / 3)


def test_best_checkpoint_keeps_first_maximum(config, fake_fashion, monkeypatch):
    config["training"]["epochs"] = 4
    bundle = build_dataset(config)
    model = build_model(config, bundle["metadata"])
    scores = iter([0.5, 0.8, 0.8, 0.6])
    snapshots = []

    def controlled_validation(model, *args):
        snapshots.append({key: value.clone() for key, value in model.state_dict().items()})
        return {"accuracy": next(scores), "loss": 1.0}

    monkeypatch.setattr("src.tasks.classification.trainer.validate", controlled_validation)
    result = fit(model, bundle, config, "cpu", logging.getLogger(__name__))
    saved = load_checkpoint(result["checkpoint"])
    assert saved["epoch"] == 2
    assert saved["best_metric"] == result["best_validation_accuracy"] == 0.8
    assert all(torch.equal(value, snapshots[1][key]) for key, value in saved["model_state"].items())


def test_linear_and_mlp_share_comparison_settings():
    linear, mlp = (load_config(f"configs/a1/{name}.yaml") for name in ("linear", "mlp"))
    for key in ("assignment", "task", "seed", "device", "dataset", "training"):
        assert linear[key] == mlp[key]
