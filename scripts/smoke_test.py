"""Real Fashion-MNIST integration check. First run downloads the dataset."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_config
from src.train import run as train
from src.evaluate import run as evaluate

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("configs/a1/linear.yaml", "configs/a1/mlp.yaml")


def check(name, directory):
    config = load_config(ROOT / name)
    config["device"] = "cpu"
    config["dataset"].update(num_workers=0, max_samples=128, batch_size=32)
    config["training"]["epochs"] = 1
    for key in ("checkpoint", "logging", "results"):
        config[key]["root"] = str(Path(directory) / key)
    trained = train(config)
    evaluated = evaluate(config, trained["checkpoint"])
    metrics = evaluated["metrics"]
    assert metrics["num_samples"] == 128
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["macro_f1"] <= 1
    assert Path(trained["checkpoint"]).exists()
    return {"model": config["model"]["name"], "parameters": metrics["num_parameters"],
            "split": evaluated["dataset"]["split_sha256"]}


def main():
    with TemporaryDirectory(prefix="co3133-smoke-") as directory:
        checked = [check(name, directory) for name in CONFIGS]
    # The main comparison is only fair when every model sees the same split.
    assert len({result["split"] for result in checked}) == 1, "Models disagree on the dataset split"
    for result in checked:
        print(f"  {result['model']}: {result['parameters']} parameters, split {result['split'][:12]}")
    print(f"PASS: real Fashion-MNIST train/validation/checkpoint/test pipeline "
          f"for {len(checked)} models (128 samples per split).")


if __name__ == "__main__":
    main()
