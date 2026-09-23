"""Regression checks for the shared, architecture-independent evaluation protocol."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.classification.evaluator import evaluate, evaluate_bundle
from src.tasks.classification.metrics import classification_metrics, count_parameters


def test_macro_f1_includes_all_ten_classes():
    result = classification_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]), 10)
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 10)
    assert len(result["confusion_matrix"]) == 10


def test_evaluation_predictions_loss_and_total_parameters():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    model.weight.requires_grad_(False)
    inputs = torch.tensor([[4., 0.], [0., 4.], [4., 0.]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    criterion = torch.nn.CrossEntropyLoss()
    result = evaluate(model, loader, "cpu", 2, criterion, collect_predictions=True)
    assert result["predictions"] == [0, 1, 0]
    assert result["targets"] == labels.tolist()
    assert result["loss"] == pytest.approx(criterion(inputs, labels).item())
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["num_parameters"] == count_parameters(model) == 4
    assert result["num_samples"] == 3
    assert result["inference_seconds"] >= 0
    assert result["inference_ms_per_sample"] == pytest.approx(1000 * result["inference_seconds"] / 3)
    bundle = {"loaders": {"test": loader}, "metadata": {"num_classes": 2}}
    bundled = evaluate_bundle(model, bundle, {}, torch.device("cpu"))
    assert bundled["loss"] == pytest.approx(result["loss"])
    assert "predictions" not in bundled and "targets" not in bundled


def test_empty_evaluation_is_rejected():
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty loader"):
        evaluate(torch.nn.Identity(), loader, "cpu", 2)
