# CO3133 — Deep Learning and Its Applications

Ho Chi Minh City University of Technology – VNU-HCM
Faculty of Computer Science and Engineering
Semester **261** · Instructor **Lê Thành Sách**

One repository grows from **shared foundation → A1 → A2 → A3**.
Group name, student names/IDs, contributions and GitHub profiles are pending.
Repository: https://github.com/kimhoangthien/CO3133-261-Deep-Learning

## Architecture and current status

- `src/core`: task-independent YAML, seeds, devices, logs and checkpoints.
- `src/datasets`: dataset factories; Fashion-MNIST implemented.
- `src/models`: small factory registry with assignment-specific models.
- `src/tasks`: task-owned batch handling, objectives, training and evaluation.
- `src/train.py`, `src/evaluate.py`: common entry points dispatch to a task.
- `assignments`: notes, notebooks and analysis; reusable code stays in `src`.
- `docs`: plain HTML/CSS GitHub Pages website; no build step or JavaScript needed.

**Implemented:** Linear raw-logit classifier, Adam/CrossEntropy training, validation,
best-accuracy checkpoint, official-test evaluation, accuracy, macro-F1 (all ten
classes, zero for undefined class F1), parameter count, confusion-matrix data,
training history and timing. No full assignment comparison has been completed.
MLP/CNN/LSTM-or-GRU/Transformer configs are explicit TODOs and fail clearly.
A2/A3 are placeholders, with no speculative implementations.

See [PROJECT_TREE.md](PROJECT_TREE.md) for the complete source tree and
[VERIFICATION.md](VERIFICATION.md) for checks actually performed.

## Environment and installation

Use a standard CPython installation with PyTorch support (verified here with
Python 3.12). Run commands from the repository root. On this Windows machine,
`python` initially resolves to an MSYS installation without pip; select the
standard Python interpreter or use its full path to create the environment.

```sh
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

If activation is unavailable, run `.venv\Scripts\python.exe` directly on Windows.
Dependencies are unpinned; record your environment with `python -m pip freeze`
when finalizing experiments. Install an appropriate PyTorch build for your GPU
if GPU acceleration is needed; the CPU path is sufficient for verification.

## Dataset and reproducibility

Fashion-MNIST is downloaded automatically to ignored `data/` on first use.
The official 60,000 training images are split into 54,000 training and 6,000
validation examples by default; the official 10,000 test images remain separate.
All models must keep seed, split and preprocessing fixed for comparison.
Normalization uses fixed mean/std 0.5/0.5, without fitting statistics on held-out data.
MNIST is permitted only for debugging; CIFAR-10 is optional, neither is implemented.

Seeds cover Python, NumPy, PyTorch, CUDA and worker initialization. Device `auto`
selects CUDA, then MPS, then CPU. cuDNN deterministic settings reduce variation;
bitwise equivalence across hardware/library versions is not promised.
Checkpoints record model/optimizer states, epoch, best validation accuracy, full
config, dataset metadata and split hash. Evaluation rejects incompatible model,
seed, preprocessing or split metadata. The test set never selects the checkpoint.

## Commands

```sh
python scripts/smoke_test.py
python -m pytest
python -m src.train --config configs/a1/linear.yaml
python -m src.evaluate --config configs/a1/linear.yaml --checkpoint checkpoints/a1/linear/best.pt
```

The smoke test downloads real Fashion-MNIST, then uses 128 samples per split for
one CPU epoch. It removes its temporary output files. First-run download time
is additional; unit tests use a mocked torchvision dataset and need no network.
Smoke metrics are not meaningful performance results.

Output paths are `<root>/<assignment>/<experiment_name>/`:

- `checkpoints/a1/linear/best.pt`: first checkpoint attaining the best validation accuracy.
- `logs/a1/linear/`: selected device, epoch metrics and evaluation log.
- `results/a1/linear/training.json`: history, elapsed fit time and provenance.
- `results/a1/linear/evaluation.json`: test metrics, confusion matrix and provenance.

Use a distinct `experiment_name` for each retained run; repeating a name overwrites
its checkpoint/results and appends logs. `dataset.max_samples` is an optional
positive debug limit per split; omit it for real experiments. Timing measures
synchronized forward passes after device transfer, including the first forward
pass; it excludes data loading and is only a basic timing check. Fit time includes
validation and checkpoint writing. Record hardware and use repeated warmed-up
measurements for the final compute-cost comparison.

## Remaining Assignment 1 work

Implement MLP, CNN, LSTM/GRU and Transformer; add EDA under
`assignments/a1/notebooks/eda.ipynb` when analysis begins; run controlled full-data
experiments; plot training/validation curves and confusion matrices; collect
correct/incorrect examples; analyze representation, inductive bias and limitations;
finish the report/slides, presentation, website and team-reviewed AI records.

## Extending the same foundation

**A2:** add the selected dataset factory in `src/datasets`, models in
`src/models/a2`, and a task package in `src/tasks` only when needed. Register
factories and task train/evaluate functions, then add a config in `configs/a2`.
Task functions receive model, dataset bundle, config and device; they own the
loss, metrics and batch unpacking. Existing A1 code need not be restructured.

**A3:** add paired-data loading, encoders/fusion models in `src/models/a3`, and
appropriate task behavior. Structured dictionaries and modality-specific outputs
belong to that task; shared seed/device/config/log/checkpoint utilities remain
unchanged. No multimodal functionality is claimed at this stage.

## Git workflow

- `main`: latest verified stable version of the whole course project; the tested
  initial foundation should ultimately be reviewed and incorporated here.
- `dev`: continuous development across A1, then A2, then A3. Finalized, verified
  work flows from `dev` into `main` at each milestone.
- `a1`, `a2`, `a3`: create only when each assignment is finalized, as snapshots;
  they are not daily development branches.
- Optional submission tags: `a1-draft`, `a1-final`, `a2-proposal`, `a2-draft`,
  `a2-final`, `a3-proposal`, `a3-draft`, `a3-final`.
- Temporary feature branches are optional for larger/parallel changes.

Initialization preserves the existing `dev` branch. No branch creation, commit,
merge, push or repository-settings change is performed by this task.

## GitHub Pages

`docs/index.html` is the landing page; `docs/assignments/a1/index.html` is the A1
report skeleton; A2/A3 pages clearly say not started. All use shared
`docs/assets/css/style.css` and relative navigation suitable for repository subpaths.

After the foundation is reviewed and committed on `main`, configure GitHub:
**Settings → Pages → Deploy from a branch → Branch: main → Folder: /docs → Save**.
No remote settings have been changed. Fill team information, reports, presentation
links and real results before submission. Source links currently refer to the
known repository; the AI-log link uses `dev`, the branch used during initialization.

## AI disclosure

[AI_USAGE.md](AI_USAGE.md) contains a template and this initialization's factual
record. Complete student identity, responsible reviewer and verification references.
The website also includes shared and assignment-specific AI disclosure sections.
