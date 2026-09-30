# Assignment 1 — M2 development start

CO3133 · Semester 261 · G-08. M1 Draft is completed and merged. M2 Final is in
progress, due **21 October 2026, 23:59 GMT+7**. [TASKS.md](TASKS.md) defines the
17 M2 tasks, dependencies, acceptance criteria and final experiment contract.

`dev` is the single source of truth. These feature branches already exist;
do not recreate them. Work from the synchronized M1 base and preserve Linear/MLP.
The three new model files are intentionally nonfunctional interface skeletons,
not registered implementations. No CNN/RNN/Transformer training can run yet.

## Kim — Hoàng Thiên Kim, 2352660

- Branch: `feature/a1-rnn`.
- Primary files: `src/models/a1/rnn.py`, `configs/a1/rnn.yaml`.
- Tasks: A1-M2-002–003; reproducibility A1-M2-012.
- Implement **one LSTM OR GRU**, not both as a requirement. Prefer image rows:
  `[B, 1, 28, 28]` → `[B, 28, 28]`, 28 timesteps with 28 input features each.
  Explain timestep meaning, hidden representation and final/aggregated state.
- Prepare environment/command documentation; final reproducibility audit follows
  all five experiments. Report: dataset/split/preprocessing updates, Linear, RNN,
  reproducibility.

```sh
git grep "TODO(A1-M2, Kim)"
```

## Khoa — Phạm Hồ Minh Khoa, 2352585

- Branch: `feature/a1-cnn`.
- Primary files: `src/models/a1/cnn.py`, `configs/a1/cnn.yaml`.
- Tasks: A1-M2-004–005; training analysis A1-M2-010.
- Design a small Fashion-MNIST CNN with convolution, activation, documented feature
  maps, pooling/justified spatial reduction and a classification head. The mandatory
  model must be group-designed, not a pretrained backbone.
- Preserve shared histories for later training/validation curve analysis across
  all five models. Report: training setup, MLP, CNN, training behavior.

```sh
git grep "TODO(A1-M2, Khoa)"
```

## Antoine — Antoine Ansou Wu, 2660057

- Branch: `feature/a1-transformer`.
- Primary files: `src/models/a1/transformer.py`, `configs/a1/transformer.yaml`.
- Tasks: A1-M2-006–007; error analysis A1-M2-011.
- Define row/column/patch tokens, token projection, positional encoding,
  Transformer encoder/self-attention and sequence aggregation. Document sequence
  length, embedding dimension and attention shapes. Do not use a pretrained ViT
  as the mandatory implementation.
- Later file: `assignments/a1/notebooks/error_analysis.ipynb`. Create it **after
  predictions/results from all five models exist**. Cover confusion matrices,
  correct/incorrect predictions, difficult cases, class-pair confusions and
  cross-model error patterns.
- Report: metrics/evaluation, Transformer, quantitative comparison support,
  confusion matrix and error analysis.

```sh
git grep "TODO(A1-M2, Antoine)"
```

## Working agreement

Each member should implement only TODOs assigned to them. Coordinate ownership
of shared test-file edits to avoid conflicting changes. Owner tests can directly
instantiate their completed class before shared registry integration.

```sh
git grep "TODO(A1-M2-INTEGRATION)"
```

Do not complete shared integration TODOs independently in a feature branch unless
explicitly coordinated. These cover registry/factory wiring and five-model generic
trainer/evaluator tests. Shared tasks also include controlled experiments, prediction
export, comparisons, report, webpage, slides/video and final QA (see TASKS.md).

Use `# TODO(A1-M2, Kim):`, `# TODO(A1-M2, Khoa):`,
`# TODO(A1-M2, Antoine):` and `# TODO(A1-M2-INTEGRATION):` with concrete actions.
Remove completed TODOs; do not replace them with DONE comments. Reserve
`# FIXME(A1):` for confirmed bugs and `# NOTE(A1):` for interface/design constraints.

All models accept image batches `[B, 1, 28, 28]` and return raw `[B, 10]` logits.
Do not apply softmax before shared CrossEntropyLoss or add model-specific training
loops. Extend skeleton constructor arguments explicitly when selecting architecture
settings; keep config keys under existing `model.parameters` and consume every key.
The common configs follow Linear/MLP; architecture choices are intentionally left
as TODO comments with no invented values.

All five final models use the same Fashion-MNIST split/seed, normalization,
evaluation, Accuracy/Macro-F1, parameter-count and timing conventions, and documented
validation-based checkpoint rule. See [the full contract](TASKS.md#shared-interface-and-final-experiment-contract).
Architecture-specific hyperparameters may differ. Final evidence includes metrics,
timings, checkpoints, histories, curves, confusion matrices and example predictions.

## Safe branch workflow

Start from a clean working tree. If `git status` shows uncommitted changes, stop;
do not stash, discard, reset or auto-commit them. Substitute your existing branch
name below (Kim example):

```sh
git status
git fetch origin
git checkout feature/a1-rnn
git pull --no-rebase origin feature/a1-rnn
git merge origin/dev
git push origin feature/a1-rnn
```

Use the equivalent existing branch for Khoa or Antoine. Do not force push or
recreate branches. Resolve conflicts only when the intended result is unambiguous;
otherwise coordinate with the affected owner. Implement and test your assigned
work, then submit it for review and merge into `dev`. Pull/merge current `dev`
before starting later analysis. Final promotion to `main`, snapshot `a1` and
`a1-final` tag happen only after all final deliverables and QA are complete.

## Validation before integration

Activate the project Python environment from the repository root, then run:

```sh
python -m compileall src
pytest
```

Keep existing M1 tests passing. Add model-specific forward/shape/finite-logit/loss
tests when implementing your model, then coordinate registry and shared pipeline
coverage. Do not weaken tests or implement another member's model to make checks pass.

All members jointly own five-model comparison, representation/inductive-bias
discussion, limitations, conclusion, slides, video and final webpage. Record
meaningful AI assistance and actual verification in the shared AI log.
