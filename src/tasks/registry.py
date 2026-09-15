from .classification.trainer import fit
from .classification.evaluator import evaluate_bundle

TASKS = {"classification": {"train": fit, "evaluate": evaluate_bundle}}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}")
    return TASKS[name]
