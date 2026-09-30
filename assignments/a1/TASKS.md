# Assignment 1 task allocation

CO3133 – Deep Learning and Its Applications · Semester 261 · Group G-08

Updated 30 September 2026. Operational counterpart to the externally maintained
`CO3133_A1_Timeline.xlsx`; that workbook was not provided or edited. Dates use GMT+7.
`dev` is the single source of truth for development. See [START_HERE.md](START_HERE.md)
for member entry points and branch synchronization instructions.

## M1 Draft — Completed

M1 was completed and merged, as confirmed for this preparation task. Deadline:
23 September 2026, 23:59 (25% of A1). Preserve the working Fashion-MNIST pipeline,
Linear, MLP, trainer, evaluator, metrics, EDA and M1 tests.

The following allocation and schedule are retained as historical records:

| IDs | Owner / historical branch | Delivered scope | Original schedule |
| --- | --- | --- | --- |
| A1-001 | Shared / `dev` | Interface and source-base agreement | 17–18 Sep |
| A1-002–004 | Hoàng Thiên Kim, 2352660 / `feature/a1-data-linear` | Fashion-MNIST, deterministic splits/metadata, EDA, Linear and config | 18–20 Sep |
| A1-005–006 | Phạm Hồ Minh Khoa, 2352585 / `feature/a1-training-mlp` | Generic training/validation, history, best checkpoint, MLP and config | 18–20 Sep |
| A1-007–008 | Antoine Ansou Wu, 2660057 / `feature/a1-evaluation` | Accuracy, Macro-F1, parameter count, inference timing, pipeline/model/data tests | 18–20 Sep |
| A1-009 | Shared / `dev` | Integrate M1 branches | 21 Sep |
| A1-010–011 | Shared / `dev` | Linear + MLP verification, A1 page and documentation | 21–22 Sep |
| A1-012 | Shared / `dev` → `main` | M1 QA and submission preparation | 23 Sep |

Historical verification is in [VERIFICATION.md](../../VERIFICATION.md) and
[AI_USAGE.md](../../AI_USAGE.md). Their 23 September observations include an earlier
unexplained MLP nonfinite run and EDA/report follow-ups. M1 completion does not
establish that the earlier failure's cause was diagnosed; carry these provenance
questions into M2 reproducibility and final QA without rewriting the baselines.

## Post-M1 audit

- Linear and MLP are implemented and registered. Existing tests cover their
  raw-logit contract, training, evaluation, checkpoints and shared protocol.
- The task registry and CLI entry points already support generic classification;
  no structural change is required for M2.
- Advanced configs already existed. Their common settings now match Linear/MLP,
  including explicit `dataset.download: true`; architecture choices remain comments.
- CNN/RNN/Transformer now have interface-only skeletons. Construction/forward
  deliberately raise `NotImplementedError`; no layers or model logic are supplied.
  They remain unregistered, so M1 imports and execution stay usable.
- `START_HERE.md` was absent and is now the M2 entry guide. The future error-analysis
  notebook remains absent until five-model predictions/results are available.
- No legacy M1 TODO markers were found in the audited base.

## M2 Final — In Progress

Deadline: **21 October 2026, 23:59 GMT+7** (75% of A1). Five mandatory models:
**Linear, MLP, CNN, one LSTM OR GRU, Transformer**. Implementing both recurrent
types is optional extension work, not a mandatory requirement.

| Member | Existing branch | Primary files | Analysis responsibility |
| --- | --- | --- | --- |
| Hoàng Thiên Kim — 2352660 | `feature/a1-rnn` | `src/models/a1/rnn.py`, `configs/a1/rnn.yaml` | Reproducibility |
| Phạm Hồ Minh Khoa — 2352585 | `feature/a1-cnn` | `src/models/a1/cnn.py`, `configs/a1/cnn.yaml` | Training behavior |
| Antoine Ansou Wu — 2660057 | `feature/a1-transformer` | `src/models/a1/transformer.py`, `configs/a1/transformer.yaml` | Later `assignments/a1/notebooks/error_analysis.ipynb` |

All task dependencies below refer to A1-M2 IDs. Shared/All tasks are coordinated
on `dev`; members must not independently complete shared integration markers on
their feature branches. Coordinate edits to shared test files as well.

### A1-M2-001 — M2 interface/config agreement

- Owner: Shared. Branch: `dev`. Status: In Progress (preparation documented; team review pending).
- Dependencies: completed M1 Draft.
- Expected output: agreed model interfaces, ownership, config schema and common protocol.
- Definition of Done: team reviews this plan and the skeletons; each owner specifies
  consumed `model.parameters` and shape contracts; baseline behavior is preserved.

### A1-M2-002 — Implement LSTM/GRU model

- Owner: Kim. Branch: `feature/a1-rnn`. Status: Not Started.
- Dependencies: A1-M2-001.
- Expected output: `src/models/a1/rnn.py` with one LSTM OR GRU classifier.
- Definition of Done: validate arguments; document image `[B, 1, 28, 28]` to row
  sequence `[B, 28, 28]` (preferred), timestep meaning, input size and hidden
  representation; classify a final/aggregated recurrent representation into raw
  `[B, 10]` logits compatible with shared CrossEntropyLoss and trainer/evaluator.

### A1-M2-003 — RNN config and model tests

- Owner: Kim. Branch: `feature/a1-rnn`. Status: Not Started.
- Dependencies: A1-M2-002.
- Expected output: completed `configs/a1/rnn.yaml` and RNN tests in `tests/test_models.py`.
- Definition of Done: every model parameter is consumed/validated; chosen recurrent
  type, input/hidden sizes, layers and optional dropout are explained; tests cover
  sequence conversion, forward `[B, 10]`, finite logits and CrossEntropyLoss/backward.

### A1-M2-004 — Implement CNN model

- Owner: Khoa. Branch: `feature/a1-cnn`. Status: Not Started.
- Dependencies: A1-M2-001.
- Expected output: `src/models/a1/cnn.py` with a group-designed CNN.
- Definition of Done: document convolution, activation, feature maps, pooling or
  justified spatial reduction, flatten/global aggregation and classification head;
  support configurable classes and raw `[B, 10]` logits through the generic pipeline.
  Use an architecture appropriate for Fashion-MNIST, not a pretrained main model.

### A1-M2-005 — CNN config and model tests

- Owner: Khoa. Branch: `feature/a1-cnn`. Status: Not Started.
- Dependencies: A1-M2-004.
- Expected output: completed `configs/a1/cnn.yaml` and CNN tests in `tests/test_models.py`.
- Definition of Done: consumed channel/classifier/dropout settings are justified;
  forward, `[B, 10]`, finite logits and CrossEntropyLoss/backward tests pass;
  shared training-history output remains available for later curve analysis.

### A1-M2-006 — Implement Transformer model

- Owner: Antoine. Branch: `feature/a1-transformer`. Status: Not Started.
- Dependencies: A1-M2-001.
- Expected output: `src/models/a1/transformer.py` with a project-built classifier.
- Definition of Done: define row/column/patch tokens, sequence length, embedding
  dimension, projection, positional encoding, encoder/self-attention input/output
  shapes, aggregation and classifier; validate shapes and return raw `[B, 10]`
  logits with shared trainer/evaluator compatibility. No pretrained ViT main model.

### A1-M2-007 — Transformer config and model tests

- Owner: Antoine. Branch: `feature/a1-transformer`. Status: Not Started.
- Dependencies: A1-M2-006.
- Expected output: completed `configs/a1/transformer.yaml` and Transformer tests in `tests/test_models.py`.
- Definition of Done: token strategy, `d_model`, `nhead`, encoder layers, feedforward
  dimension and dropout are consumed and justified; token/attention shapes,
  forward `[B, 10]`, finite logits and CrossEntropyLoss/backward are tested.

### A1-M2-008 — Integrate all five models

- Owner: Shared. Branch: `dev`. Status: Waiting for implementations.
- Dependencies: A1-M2-003, A1-M2-005, A1-M2-007.
- Expected output: reviewed merges, factories/registry entries and five-model integration coverage.
- Definition of Done: all five real configs run through the generic trainer and
  evaluator; model, checkpoint and smoke tests pass with common split metadata;
  M1 remains functional. Coordinate shared registries/tests and prediction export
  using the existing evaluator's `collect_predictions` support for later analysis.

### A1-M2-009 — Run controlled five-model experiments

- Owner: Shared. Branch: `dev`. Status: Waiting for integration.
- Dependencies: A1-M2-008.
- Expected output: full-data Linear, MLP, CNN, RNN and Transformer runs and the artifact set below.
- Definition of Done: all five use the common protocol, with no debug sample limit;
  retain real checkpoints, configs, histories, metrics, predictions/targets linked
  to sample identities, split hashes and timing/hardware records. No invented results.

### A1-M2-010 — Training behavior analysis

- Owner: Khoa. Branch: `feature/a1-cnn` (merge reviewed analysis into `dev`). Status: Waiting for experiments.
- Dependencies: A1-M2-009.
- Expected output: five-model training/validation loss and accuracy curves with analysis.
- Definition of Done: interpret convergence, generalization gap, stability and
  training cost using saved histories; label axes/run identities and explain tuning decisions.

### A1-M2-011 — Error analysis

- Owner: Antoine. Branch: `feature/a1-transformer` (merge reviewed analysis into `dev`). Status: Waiting for all five outputs.
- Dependencies: A1-M2-009; predictions/results from every model must exist first.
- Expected output: `assignments/a1/notebooks/error_analysis.ipynb` and referenced figures.
- Definition of Done: reproducible confusion matrices, correct/incorrect examples,
  difficult cases, class-pair confusions and cross-model error patterns from real
  predictions on the same samples. Do not implement this analysis during preparation.

### A1-M2-012 — Reproducibility audit

- Owner: Kim. Branch: `feature/a1-rnn` (coordinate shared documentation on `dev`). Status: Planned; final audit waits for experiments.
- Dependencies: A1-M2-001 for documentation preparation; A1-M2-009 for final audit.
- Expected output: environment/version record, exact commands, artifact manifest and rerun evidence.
- Definition of Done: document seed, split hash, preprocessing, hardware, dependency
  versions, config, commit, checkpoint selection and timing; reproduce checkpoint
  loading/evaluation and check the training recipe from a clean checkout. Record
  numerical tolerances/hardware limits and investigate or explicitly track the
  historical MLP nonfinite-run provenance without claiming an unsupported diagnosis.

### A1-M2-013 — Full comparison and interpretation

- Owner: All. Branch: `dev` (coordinated). Status: Waiting for analysis.
- Dependencies: A1-M2-009, A1-M2-010, A1-M2-011, A1-M2-012.
- Expected output: sourced five-model comparison and representation/inductive-bias discussion.
- Definition of Done: compare quality, parameters and compute; explain flattening,
  spatial locality, recurrent order and token/self-attention representations;
  discuss limitations using observed evidence, without premature final conclusions.

### A1-M2-014 — Update A1 GitHub Page

- Owner: All. Branch: `dev` (coordinated). Status: Waiting for final outputs.
- Dependencies: A1-M2-013; final report/presentation links from A1-M2-015–016.
- Expected output: `docs/assignments/a1/index.html` with final artifacts and results.
- Definition of Done: accurate five-model results, figures, checkpoint/report/slides/video
  links and AI disclosure are reviewed; local and published links are verified at final QA.

### A1-M2-015 — Final report

- Owner: All. Branch: `dev` (coordinated). Status: Waiting for analysis.
- Dependencies: A1-M2-013.
- Expected output: final report with sections allocated below.
- Definition of Done: explain all five architectures, protocol, metrics, curves,
  error analysis, reproducibility, limitations and conclusions; verify every claim
  against retained outputs and reconcile historical EDA documentation.

### A1-M2-016 — Slides and YouTube presentation

- Owner: All. Branch: `dev` (coordinated). Status: Waiting for report.
- Dependencies: A1-M2-015.
- Expected output: final slides and accessible YouTube video link.
- Definition of Done: all members review the presentation, cover methods/results/
  analysis and contributions, and verify slides/video access from the final webpage.

### A1-M2-017 — Final QA, main promotion, a1-final tag

- Owner: Shared. Branch: `dev` → `main`; final snapshot `a1`. Status: Waiting for deliverables.
- Dependencies: A1-M2-012, A1-M2-014, A1-M2-015, A1-M2-016.
- Expected output: reviewed final submission, stable `main`, frozen `a1`, immutable `a1-final` tag.
- Definition of Done: repeat final reproducibility check, tests, artifact/link checks
  and team approval; promote verified `dev` to `main`, then create the final snapshot/tag.
  This is later work; this preparation does not publish the final assignment or tag it.

## Shared interface and final experiment contract

`build_dataset(config)` returns `loaders` (`train`, `val`, `test`) and `metadata`.
`build_model(config, metadata)` returns an `nn.Module`; factories take input shape
and number of classes from metadata. Models receive `[B, 1, 28, 28]` images and
return `[B, num_classes]` raw logits. Integer class targets and CrossEntropyLoss
belong to the shared trainer. Sequence/token conversion stays inside each model.

Use the existing config schema in `configs/a1/linear.yaml` and `mlp.yaml`:
`assignment`, `task`, `experiment_name`, `seed`, `device`, `dataset`, `model`,
`training`, `checkpoint`, `logging`, `results`. Add architecture settings only under
`model.parameters`, with explicit consuming code. Skeleton constructor signatures
must be extended by their owners when those settings are chosen. Do not add unused
keys or arbitrary/null hyperparameter values to make a config appear complete.

All five final models must share:

- Fashion-MNIST, the same train/validation/test indices and split hash; default
  54,000/6,000/10,000, seed 42 and validation ratio 0.1. Keep the official test set
  held out from tuning and checkpoint selection.
- Dataset root `data`, batch size 64, workers 0 and fixed normalization mean/std
  0.5/0.5. Retain the common `device: auto` config but record and control the actual
  device/hardware for comparisons. No `max_samples` limit in final experiments.
- The existing generic trainer/evaluator and Accuracy/Macro-F1 implementations.
  Macro-F1 averages all ten classes, with zero for undefined class F1.
- Parameter count = all model parameters, including frozen parameters.
- Checkpoint selection = highest validation accuracy, keeping the first winner
  on ties; save config, dataset metadata and split hash with the checkpoint.
- The same timing convention: `training_seconds` measures the fit loop, including
  validation and checkpoint writes; `inference_seconds` sums synchronized forward
  passes after device transfer, including the first pass, excluding loading/transfer;
  `inference_ms_per_sample` uses the evaluated sample count. Record hardware/batch
  size. Agree repeated/warmed-up measurements for final compute analysis, keeping
  them separately labeled and applying the same procedure to every model.

Existing shared training values (10 epochs, Adam, learning rate 0.001, weight decay
0) are retained for preparation. Document and agree any later protocol deviations
before controlled runs. Architecture-specific hyperparameters need not be identical;
fairness means controlled data, selection and measurement protocols. Tune on validation only.

Shared integration files include model/task registries, train/evaluate entry points,
shared tests/smoke script, README, AI log and A1 webpage. Reuse working M1 code.
No full experiments, final metrics or final conclusions are part of this setup.

## Required final outputs

For **each of Linear, MLP, CNN, RNN and Transformer**, retain Accuracy, Macro-F1,
parameter count, training time, inference time, checkpoint and training history.
Use unique experiment names for retained runs to avoid overwriting outputs.
The current output layout is `<root>/a1/<experiment_name>/`, including `best.pt`,
`training.json` and `evaluation.json` under their configured roots.

Final analysis must include training/validation curves, confusion matrices, correct
and incorrect prediction examples, and representation/inductive-bias discussion.
The existing evaluator can collect predictions, but the standard evaluation entry
point does not export them; coordinate a reproducible analysis/export step in
A1-M2-008 before A1-M2-009. Keep sample identities aligned across all five models.

Generated checkpoints/results are ignored by Git. Publish real artifacts through
an agreed submission location with an accessible manifest; do not rely on local
ignored files or invent metric placeholders. Complete report, assignment webpage,
slides, YouTube video and the final reproducibility check.

## Final report ownership

| Owner | Sections |
| --- | --- |
| Kim | Dataset/split/preprocessing updates if needed; Linear; RNN; reproducibility |
| Khoa | Training setup; MLP; CNN; training behavior |
| Antoine | Metrics/evaluation; Transformer; quantitative comparison support; confusion matrix; error analysis |
| All | Five-model comparison; representation/inductive-bias discussion; limitations; conclusion; slides; video; final webpage |
