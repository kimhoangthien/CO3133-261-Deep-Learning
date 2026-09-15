from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric, config, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_metric": best_metric, "config": config,
                "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    return checkpoint
