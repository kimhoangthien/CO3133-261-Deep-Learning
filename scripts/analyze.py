"""Post-training analysis for any A1 model: curves, confusion matrix, examples.

Reads a finished run and writes figures plus an error taxonomy. It never
trains and never selects a checkpoint, so it can be re-run on any model
without touching the comparison protocol.

    python scripts/analyze.py --config configs/a1/transformer.yaml \
        --checkpoint checkpoints/a1/transformer/best.pt
"""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.checkpoint import load_checkpoint
from src.core.config import load_config, output_dir
from src.core.device import select_device
from src.core.seed import seed_everything
from src.datasets.registry import build_dataset
from src.models.registry import build_model


def collect(config, checkpoint_path, device):
    """One inference pass over the official test split, kept per sample."""
    saved = load_checkpoint(checkpoint_path)
    # Same guards as src/evaluate.py: figures must describe the reported run.
    for key in ("assignment", "task", "model", "seed"):
        if saved["config"].get(key) != config.get(key):
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    bundle = build_dataset(config)
    if bundle["metadata"] != saved["metadata"]:
        raise ValueError("Checkpoint dataset/split does not match this configuration")
    model = build_model(config, bundle["metadata"]).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    targets, probabilities = [], []
    with torch.inference_mode():
        for inputs, labels in bundle["loaders"]["test"]:
            probabilities.append(model(inputs.to(device)).softmax(1).cpu())
            targets.append(labels)
    return bundle, torch.cat(targets), torch.cat(probabilities)


def plot_curves(history, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [entry["epoch"] for entry in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot(epochs, [entry[f"{split}_{metric}"] for entry in history],
                      marker="o", label=split)
        axis.set_xlabel("epoch"), axis.set_ylabel(metric), axis.legend(), axis.grid(alpha=0.3)
    figure.tight_layout(), figure.savefig(path, dpi=150), plt.close(figure)


def plot_confusion(matrix, names, path):
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("predicted"), axis.set_ylabel("true")
    threshold = matrix.max() / 2
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=7,
                      color="white" if matrix[i, j] > threshold else "black")
    figure.colorbar(image), figure.tight_layout(), figure.savefig(path, dpi=150), plt.close(figure)


def plot_examples(dataset, names, groups, path, count):
    figure, axes = plt.subplots(len(groups), count, figsize=(1.5 * count, 1.9 * len(groups)))
    axes = axes.reshape(len(groups), count)
    for row, (title, indices) in enumerate(groups.items()):
        axes[row, 0].set_ylabel(title, fontsize=8)
        for column in range(count):
            axis = axes[row, column]
            axis.set_xticks([]), axis.set_yticks([])
            if column >= len(indices):
                axis.axis("off")
                continue
            index, true, predicted, confidence = indices[column]
            image = dataset[index][0]
            # Undo the fixed 0.5/0.5 normalization for display only.
            axis.imshow(image.squeeze(0) * 0.5 + 0.5, cmap="gray")
            axis.set_title(f"{names[predicted]}\n({names[true]}) {confidence:.2f}", fontsize=6)
    figure.tight_layout(), figure.savefig(path, dpi=150), plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Analyze a finished A1 run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    device = select_device(config.get("device", "auto"))
    results = output_dir(config, "results")
    out = Path(args.out) if args.out else results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    bundle, targets, probabilities = collect(config, args.checkpoint, device)
    names = bundle["metadata"]["class_names"]
    classes = len(names)
    top = probabilities.topk(2, dim=1)
    predictions, confidence = top.indices[:, 0], top.values[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]
    correct = predictions == targets
    matrix = torch.bincount(targets * classes + predictions, minlength=classes**2).reshape(classes, classes)

    history_path = results / "training.json"
    if history_path.exists():
        plot_curves(json.loads(history_path.read_text())["history"], out / "curves.png")
    plot_confusion(matrix.numpy(), names, out / "confusion_matrix.png")

    def pick(mask, score, descending):
        order = torch.where(mask)[0][score[mask].argsort(descending=descending)][:args.examples]
        return [(int(i), int(targets[i]), int(predictions[i]), float(confidence[i])) for i in order]

    plot_examples(bundle["loaders"]["test"].dataset, names, {
        "confident\ncorrect": pick(correct, confidence, True),
        "correct but\nuncertain": pick(correct, margin, False),
        "confident\nfailures": pick(~correct, confidence, True),
    }, out / "examples.png", args.examples)

    support = matrix.sum(1)
    confusions = [{"true": names[i], "predicted": names[j], "count": int(matrix[i, j]),
                   "rate_of_true_class": round(float(matrix[i, j] / support[i]), 4)}
                  for i in range(classes) for j in range(classes) if i != j and matrix[i, j]]
    confusions.sort(key=lambda row: -row["count"])
    taxonomy = {
        "model": config["model"]["name"],
        "checkpoint": str(args.checkpoint),
        "accuracy": round(float(correct.float().mean()), 4),
        "per_class": [{"class": names[i], "support": int(support[i]),
                       "recall": round(float(matrix[i, i] / support[i]), 4),
                       "precision": round(float(matrix[i, i] / matrix[:, i].sum()), 4) if matrix[:, i].sum() else 0.0}
                      for i in range(classes)],
        "top_confusions": confusions[:args.pairs],
        "mean_confidence_correct": round(float(confidence[correct].mean()), 4),
        "mean_confidence_wrong": round(float(confidence[~correct].mean()), 4) if (~correct).any() else None,
    }
    (out / "error_analysis.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")

    print(f"Wrote curves, confusion matrix, examples and error taxonomy to {out}")
    for row in taxonomy["top_confusions"][:5]:
        print(f"  {row['true']} -> {row['predicted']}: {row['count']} ({row['rate_of_true_class']:.1%} of the class)")


if __name__ == "__main__":
    main()
