import argparse
from src.core.config import load_config, output_dir
from src.core.checkpoint import load_checkpoint
from src.core.seed import seed_everything
from src.core.device import select_device
from src.core.logger import close_logger, make_logger, write_json
from src.datasets.registry import build_dataset
from src.models.registry import build_model
from src.tasks.registry import get_task


def run(config, checkpoint_path):
    seed_everything(config.get("seed", 42))
    device = select_device(config.get("device", "auto"))
    logger = make_logger(output_dir(config, "logging") / "evaluate.log")
    try:
        logger.info("Selected device: %s", device)
        saved = load_checkpoint(checkpoint_path)
        for key in ("assignment", "task", "model", "seed"):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Checkpoint/config mismatch: {key}")
        bundle = build_dataset(config)
        if bundle["metadata"] != saved["metadata"]:
            raise ValueError("Checkpoint dataset/split does not match evaluation configuration")
        model = build_model(config, bundle["metadata"]).to(device)
        model.load_state_dict(saved["model_state"])
        metrics = get_task(config["task"])["evaluate"](model, bundle, config, device)
        result = {"metrics": metrics, "config": config, "dataset": bundle["metadata"],
                  "checkpoint": str(checkpoint_path), "checkpoint_epoch": saved["epoch"], "device": str(device)}
        write_json(output_dir(config, "results") / "evaluation.json", result)
        logger.info("Evaluation: %s", metrics)
        return result
    finally:
        close_logger(logger)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a course checkpoint on the official test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
