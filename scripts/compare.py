"""Aggregate finished runs into the required five-model comparison table.

Reads results/<assignment>/<experiment>/{training,evaluation}.json, so it
reports only what was actually run and never retrains or re-evaluates.
Refuses to emit a table whose rows do not share one dataset split, which
is the fairness condition the comparison depends on.

    python scripts/compare.py --assignment a1
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = [("model", "Model"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
           ("num_parameters", "Parameters"), ("training_seconds", "Train (s)"),
           ("inference_ms_per_sample", "Inference (ms/sample)"),
           ("best_validation_accuracy", "Best val acc"), ("epochs", "Epochs")]


def read_run(directory):
    training_path, evaluation_path = directory / "training.json", directory / "evaluation.json"
    if not (training_path.exists() and evaluation_path.exists()):
        return None
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    return {"model": evaluation["config"]["model"]["name"],
            "experiment": directory.name,
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "num_parameters": metrics["num_parameters"],
            "inference_ms_per_sample": metrics["inference_ms_per_sample"],
            "training_seconds": training["training_seconds"],
            "best_validation_accuracy": training["best_validation_accuracy"],
            "epochs": len(training["history"]),
            "split": evaluation["dataset"]["split_sha256"],
            "device": evaluation.get("device", "?")}


def format_cell(key, value):
    if key in ("accuracy", "macro_f1", "best_validation_accuracy"):
        return f"{value:.4f}"
    if key == "num_parameters":
        return f"{value:,}"
    if key == "training_seconds":
        return f"{value:.0f}"
    if key == "inference_ms_per_sample":
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Build the model comparison table")
    parser.add_argument("--assignment", default="a1")
    parser.add_argument("--results", default="results")
    parser.add_argument("--order", nargs="*",
                        default=["linear", "mlp", "cnn", "rnn", "transformer"])
    args = parser.parse_args()

    root = Path(args.results) / args.assignment
    runs = [run for run in (read_run(d) for d in sorted(root.iterdir()) if d.is_dir()) if run]
    if not runs:
        raise SystemExit(f"No finished run under {root}: train and evaluate first")
    rank = {name: index for index, name in enumerate(args.order)}
    runs.sort(key=lambda run: (rank.get(run["model"], len(rank)), run["model"]))

    splits = {run["split"] for run in runs}
    if len(splits) > 1:
        raise SystemExit("Runs disagree on the dataset split; the comparison would not be fair:\n"
                         + "\n".join(f"  {run['model']}: {run['split'][:12]}" for run in runs))

    header = [label for _, label in COLUMNS]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for run in runs:
        lines.append("| " + " | ".join(format_cell(key, run[key]) for key, _ in COLUMNS) + " |")
    missing = [name for name in args.order if name not in {run["model"] for run in runs}]
    table = "\n".join(lines)
    note = (f"\nAll rows share dataset split {splits.pop()[:12]} on {runs[0]['device']}."
            + (f"\nStill missing: {', '.join(missing)}." if missing else ""))

    (root / "comparison.md").write_text(table + note + "\n", encoding="utf-8")
    (root / "comparison.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(table + note)
    print(f"\nWrote {root / 'comparison.md'} and {root / 'comparison.json'}")


if __name__ == "__main__":
    main()
