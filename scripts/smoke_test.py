"""Real Fashion-MNIST integration check. First run downloads the dataset."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_config
from src.train import run as train
from src.evaluate import run as evaluate


def main():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/a1/linear.yaml")
    config["device"] = "cpu"
    config["dataset"].update(num_workers=0, max_samples=128, batch_size=32)
    config["training"]["epochs"] = 1
    with TemporaryDirectory(prefix="co3133-smoke-") as directory:
        for key in ("checkpoint", "logging", "results"):
            config[key]["root"] = str(Path(directory) / key)
        trained = train(config)
        evaluated = evaluate(config, trained["checkpoint"])
        assert evaluated["metrics"]["num_samples"] == 128
    print("PASS: real Fashion-MNIST train/validation/checkpoint/test pipeline (128 samples per split).")


if __name__ == "__main__":
    main()
