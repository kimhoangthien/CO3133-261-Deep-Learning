 # Verification - 2026-09-15

## Initial inspection

Existing branch: `dev`, tracking `origin/dev`. HEAD contained only a two-line
README; the working tree had that README deleted and no source files.
No existing implementations, imports, duplicate architectures or files requiring
migration were present. README was restored and expanded. No source files were
moved or removed. No Git branch, commit, merge, push or remote settings changed.

## Verified environment

Windows, CPython 3.12; project `.venv` created with system-site-packages to reuse
installed NumPy/PyYAML/pytest. Installed torch 2.14.0 and torchvision 0.29.0.
Existing NumPy 2.4.3, PyYAML 6.0.3 and pytest 9.0.2 were used.
`python -m pip install -r requirements.txt` resolved all requirements as satisfied.
The initial PATH Python was MSYS Python 3.11 without pip. Commands below used
`.venv/Scripts/python.exe`. Windows sandbox temporary-directory restrictions
required elevated environment setup and test execution.

## Results

- `python -m pytest -q --tb=short`: **6 passed in 5.33 seconds** after the final
  Python change. Tests cover offline dataset construction, split determinism and
  disjointness, official-test selection, normalization, dimensions, class count,
  raw logits, loss/backprop/optimizer update, TODO model errors, training,
  checkpoint roundtrip, log handle closure, evaluation, split mismatch rejection,
  and known-value accuracy/macro-F1/confusion matrix.
- `python scripts/smoke_test.py`: **PASS**, real downloaded Fashion-MNIST,
  128 examples per split, one CPU epoch, validation, checkpoint save/load and
  official-test subset evaluation. Temporary outputs cleaned successfully.
- Shared CLI training and evaluation: **PASS**, using
  `results/cli-verification.yaml` (one epoch, 128 examples per split, CPU), with
  checkpoint `checkpoints/a1/cli-verification/best.pt`. These ignored artifacts
  are infrastructure checks only; the default ten-epoch full-data run was not run.
- Both entry-point `--help` commands: **PASS**.
- Every `src` module imported successfully.
- Static HTML validation: **30 local navigation, stylesheet and fragment links
  resolved across all four pages**, using relative paths under `docs`.
- Source review: no task/data-specific logic in `src/core`; no duplicate `src/data`
  or `src/engine`; A2/A3 have only extension placeholders.
- `git check-ignore`: data, checkpoints, logs, results and environment excluded.
  `git ls-files` contained only the original README before staging; generated
  assets are not tracked. `git diff --check` passed after whitespace cleanup.

## Fix found during verification

Windows could not remove a smoke-test directory while logs remained open.
Training/evaluation now close their logger handlers in `finally` blocks. The
pipeline test checks that a completed training log can be deleted on Windows;
the smoke test was rerun successfully.

## Limits and manual follow-up

The in-app Browser runtime reported unavailable; visual desktop/mobile browser
QA was not performed. Responsive CSS and static links were checked in source.
No full experiments, GPU/MPS runs, multi-worker runs, remote publication or
student/reviewer approvals were performed. Fill real group identity, contributions,
report/slides and YouTube links, and review AI_USAGE.md before submission.


## Status evidence addendum - 2026-09-23

This addendum summarizes existing local evidence inspected during the landing-page
and AI-disclosure refresh. The original 15 September checks above remain historical.
No training or Python test suite was rerun for this documentation-only update.

- Implementation inspected at `5445324`: Linear and MLP, shared training/evaluation,
  dataset preprocessing and metadata, and guards against nonfinite values.
- Local source: `results/a1/integration-review/integration-report.md` records
  **34 passed**, real Fashion-MNIST smoke checks for both models, and execution
  of all 11 EDA code cells without errors. Its descriptions of uncommitted guard
  changes refer to review time; those changes are now in commit `5445324`.
- Both `results/a1/linear/evaluation.json` and `results/a1/mlp/evaluation.json`
  record full CPU evaluation on 10,000 test images, seed 42, 54,000/6,000 training
  and validation sizes, no sample limit, identical preprocessing and split hash,
  ten configured training epochs and selected checkpoint epoch 10.

| Model | Test accuracy | Macro-F1 | Parameters |
| --- | ---: | ---: | ---: |
| Linear | 84.06% | 0.83934 | 7,850 |
| MLP | 87.59% | 0.87549 | 101,770 |

These are single-seed baseline results. Checkpoints, result JSON files, the review
report and executed EDA copy are local ignored artifacts, not published evidence.
The team must include them through the submission channel and complete review.
The initial MLP run's NaN losses/nonfinite checkpoint remain unexplained according
to the review; guards and successful subsequent runs do not resolve that cause.
The review therefore does not establish M1 submission readiness.

Documentation checks: baseline values and protocol checked against local JSON;
local HTML navigation/stylesheets/fragments checked for valid targets;
`git diff --check` passed. Browser visual QA was not performed for this text update.
