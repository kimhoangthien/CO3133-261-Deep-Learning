import argparse
from src.core.config import load_config, output_dir
from src.core.seed import seed_everything
from src.core.device import select_device
from src.core.logger import close_logger, make_logger
from src.datasets.registry import build_dataset
from src.models.registry import build_model
from src.tasks.registry import get_task


def run(config):
    seed_everything(config.get("seed", 42))
    device = select_device(config.get("device", "auto"))
    logger = make_logger(output_dir(config, "logging") / "train.log")
    try:
        logger.info("Selected device: %s", device)
        task = get_task(config["task"])
        bundle = build_dataset(config)
        model = build_model(config, bundle["metadata"]).to(device)
        return task["train"](model, bundle, config, device, logger)
    finally:
        close_logger(logger)


def main():
    parser = argparse.ArgumentParser(description="Train a configured course experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
