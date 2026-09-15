from pathlib import Path
import pytest
import torch
from src.train import run as train
from src.evaluate import run as evaluate
from src.core.checkpoint import load_checkpoint
from src.models.registry import build_model
from src.tasks.classification.metrics import classification_metrics


def test_pipeline_and_checkpoint(config, fake_fashion):
    result = train(config)
    assert len(result["history"]) == 1
    # Closed log handles must allow Windows to remove completed-run logs.
    log_path = Path(config["logging"]["root"]) / "a1/linear/train.log"
    log_path.unlink()
    saved = load_checkpoint(result["checkpoint"])
    assert saved["epoch"] == 1 and saved["optimizer_state"]["state"]
    model = build_model(config, saved["metadata"])
    load_checkpoint(result["checkpoint"], model)
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in saved["model_state"].items())
    metrics = evaluate(config, result["checkpoint"])["metrics"]
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["macro_f1"] <= 1
    assert metrics["num_parameters"] == 7850 and metrics["num_samples"] == 20
    assert Path(result["checkpoint"]).exists()
    config["dataset"]["val_ratio"] = 0.2
    with pytest.raises(ValueError, match="dataset/split"):
        evaluate(config, result["checkpoint"])


def test_metrics_known_values():
    result = classification_metrics(torch.tensor([0,0,1,1]), torch.tensor([0,1,1,1]), 2)
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2/3+4/5)/2)
    assert result["confusion_matrix"] == [[1,1],[0,2]]
