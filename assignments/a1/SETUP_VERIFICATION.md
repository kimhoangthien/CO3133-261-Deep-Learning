# M1 setup verification — 18 September 2026 (GMT+7)

- Fetched `origin`; local `main`, `dev`, `origin/main` and `origin/dev` all
  matched `ffc4e6f`. No pull/merge was needed. Current branch remains `dev`.
- Created from `dev` and pushed with upstream tracking:
  `feature/a1-data-linear`, `feature/a1-training-mlp`, `feature/a1-evaluation`.
  All start at `ffc4e6f`. No M2/snapshot branches or tags were created.
- Preserved the existing user edit to root `VERIFICATION.md`. Setup documentation
  remains uncommitted in the working tree; branch pushes contain the foundation,
  not these documentation changes. No source/config/test code was modified.
- Initial pytest, smoke and import checks failed on missing Pillow contents,
  including outside the sandbox. Repairing Pillow exposed missing SymPy contents.
  Reinstalled Pillow, SymPy and mpmath inside the ignored project `.venv` using
  pip `--ignore-installed --no-deps`; no dependency manifest was changed.

Final checks used `.venv/Scripts/python.exe` outside the sandbox:

| Check | Result |
| --- | --- |
| `-m pytest -q --tb=short` | 6 passed in 9.68 seconds |
| `scripts/smoke_test.py` | PASS: real Fashion-MNIST, 128 samples per split, one CPU epoch, train/validation/checkpoint/test, temporary output cleanup |
| Import every module via `pkgutil.walk_packages` / `importlib.import_module` | PASS: 23 `src` modules |
| A1 HTML parsing, member IDs/task IDs/deadlines and local stylesheet | PASS |
| `git diff --check` | PASS (Git reports expected LF/CRLF conversion notices) |

These checks exercise the existing Linear foundation shared by main/dev; they
are not full-data assignment results. No visual browser QA, GPU verification,
MLP run or Final-model training was performed. MLP implementation/registration
and EDA are pending. Existing interfaces are compatible with the planned MLP;
no architecture change or conflicting source implementation was found.

The external timeline workbook was not available; documentation mirrors the
confirmed allocation supplied for this setup. Team review, documentation commit,
feature implementation and milestone integration remain future actions.
