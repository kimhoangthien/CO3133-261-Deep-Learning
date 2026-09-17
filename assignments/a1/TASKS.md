# Assignment 1 task allocation

Operational counterpart to the externally maintained `CO3133_A1_Timeline.xlsx`.
Dates below are in 2026, GMT+7. The workbook was not provided or edited.
Assignments indicate responsibility, not completion.

## Milestones and dependencies

| Date | Work |
| --- | --- |
| 17–18 Sep | A1-001: shared interface/source-base agreement |
| 18–20 Sep | Parallel foundations: Kim A1-002–004; Khoa A1-005–006; Antoine A1-007–008 |
| 21 Sep | A1-009: integrate reviewed M1 branches into `dev` |
| 21–22 Sep | A1-010: end-to-end Linear + MLP verification; A1-011: A1 page and M1 documentation |
| 23 Sep | A1-012: final QA and M1 submission preparation |

**M1 Draft: 23 September, 23:59 GMT+7 (25% of A1).** Requires EDA,
Dataset/DataLoader, training/validation, runnable Linear and MLP. CNN is optional.
**M2 Final: 21 October, 23:59 GMT+7 (75% of A1).** Requires Linear, MLP, CNN,
LSTM or GRU, Transformer, full comparison, report, slides, YouTube presentation,
Assignment 1 page and checkpoints.

A1-001 precedes parallel A1-002–008, followed by A1-009 → A1-010 → A1-011 →
A1-012. A1-001 and A1-009–012 are shared responsibilities. Source interfaces
have been inspected; team agreement and milestone acceptance remain team actions.

## M1 ownership

| Member / role | Branch | Tasks and primary files |
| --- | --- | --- |
| Hoàng Thiên Kim — 2352660; Data & Reproducibility | `feature/a1-data-linear` | A1-002: Fashion-MNIST and deterministic train/validation/test handling (`src/datasets/fashion_mnist.py`, `src/datasets/registry.py` if needed). A1-003: class distribution, input size, imbalance and representative samples (`assignments/a1/notebooks/eda.ipynb`, to create). A1-004: Linear/Softmax (`src/models/a1/linear.py`, `configs/a1/linear.yaml`). Own dataset metadata, split reproducibility, relevant dataset configuration and seed consistency. |
| Phạm Hồ Minh Khoa — 2352585; Training & CNN Models | `feature/a1-training-mlp` | A1-005: generic training/validation, train/validation loss and accuracy, optimizer/criterion integration, best checkpoint (`src/tasks/classification/trainer.py`, `tests/test_training.py`). A1-006: MLP (`src/models/a1/mlp.py`, to create; `configs/a1/mlp.yaml`). Reuse `src/core/checkpoint.py`. |
| Antoine Ansou Wu — 22660057; Evaluation & Transformer | `feature/a1-evaluation` | A1-007: accuracy, macro-F1, parameter count and basic inference timing (`src/tasks/classification/evaluator.py`, `metrics.py` in that directory). A1-008: data, forward/loss, optimizer-step, training smoke and checkpoint save/load checks (`scripts/smoke_test.py`, `tests/test_data.py`, `tests/test_models.py`). Coordinate training/checkpoint test acceptance with Khoa, who edits `tests/test_training.py`. |

Kim coordinates dataset/seed settings across A1 configs; each model owner edits
their own config. Coordinate changes to `src/core/seed.py` as shared infrastructure.
Do not edit another member's trainer/evaluator files without an integration need.

## Existing interface agreement

- `build_dataset(config)` in `src/datasets/registry.py` returns a bundle with
  `loaders` (`train`, `val`, `test`) and `metadata` (input shape, class count/names,
  seed, split hash, sizes and preprocessing).
- `build_model(config, metadata)` in `src/models/registry.py` returns a PyTorch
  module. Classification models accept `[N, 1, 28, 28]` images and return raw
  `[N, num_classes]` logits; labels are integer class indices for CrossEntropyLoss.
  Future sequence/token conversion belongs inside the model or its agreed adapter.
- `fit(model, bundle, config, device, logger)` owns Adam/CrossEntropyLoss and calls
  `train_one_epoch(model, loader, optimizer, criterion, device)`. Validation uses
  the shared evaluator; history records train/validation loss and accuracy.
- `evaluate(model, loader, device, num_classes, criterion=None)` is model-independent;
  `validate` aliases it. `evaluate_bundle(model, bundle, config, device)` selects
  the official test loader. Timing covers synchronized forward passes, including
  the first pass, excluding data loading/transfer; it is only basic timing.
- `src/train.py` and `src/evaluate.py` dispatch through the existing task registry.
  No new abstraction or interface change is needed for the planned MLP.

Inspected baseline: Fashion-MNIST, Linear, shared trainer/evaluator, metrics,
checkpoint utility and six tests already exist. MLP implementation/registration
and EDA notebook are missing; MLP config is a TODO. Current smoke coverage is
Linear only. Existing code does not mean the owner's task has been accepted.

## Integration and fair comparison

Shared integration files: `src/train.py`, `src/evaluate.py`,
`src/models/registry.py`, `src/tasks/registry.py`, `README.md`, `AI_USAGE.md`, and
`docs/assignments/a1/index.html`. Coordinate shared core utilities as well.
Change these together at integration when needed, rather than assigning them to
one feature branch. Khoa supplies the MLP factory contract; the team adds its
registry entry at A1-009. Antoine extends the smoke check to both models then.
Ownership documentation is synchronized during this setup; no model wiring is changed.

All models use the same split/test set, metrics and evaluation protocol. Preserve
seed 42, validation ratio 0.1 and fixed normalization 0.5/0.5 unless the team
agrees and records a common change. Checkpoint selection uses strictly improving
validation accuracy (first winner on ties), never test metrics. Preserve config,
dataset metadata and split hash in checkpoints. Do not duplicate model-specific
training loops or hard-code Linear/MLP behavior into the trainer/evaluator.

Merge only reviewed, tested feature work into `dev`; delete feature branches after
merge. Run both Linear and MLP through the same train/evaluate entry points before
M1 acceptance. After final QA, promote `dev` to `main` and tag `a1-draft`.
No integration merges or submission tags are part of this setup.

## M2 plan — document only until M1 Draft is completed

| Future branch (from `dev`) | Owner | Scope |
| --- | --- | --- |
| `feature/a1-rnn` | Kim | LSTM/GRU, image-to-sequence representation, representation/inductive-bias analysis, final reproducibility audit |
| `feature/a1-cnn` | Khoa | CNN/tuning, training behavior, training-time and parameter comparison |
| `feature/a1-transformer` | Antoine | Transformer, evaluation, confusion matrix, correct/incorrect examples, difficult cases, error taxonomy and qualitative analysis |

Do not create these branches yet. At A1 Final only, create frozen snapshot branch
`a1` and immutable tag `a1-final`. Do not create A2/A3 branches during current work.

## Next actions

- Kim: on `feature/a1-data-linear`, audit existing splits/metadata/Linear and
  create the EDA notebook with actual distributions and representative samples.
- Khoa: on `feature/a1-training-mlp`, implement the MLP and update its config;
  verify the existing trainer and checkpoint flow without replacing them.
- Antoine: on `feature/a1-evaluation`, audit metrics/timing and existing tests;
  prepare model-independent coverage and the integrated Linear + MLP smoke check.
