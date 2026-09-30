# AI usage disclosure

Maintain one record for each meaningful use. Do not submit unreviewed generated claims.
Student identity and approval fields must be completed by the team.

Last updated: **2026-09-30**. Historical attribution below is limited to the
existing disclosure, commit trailers, and this session. A commit author is not
a confirmed AI operator or reviewer. Missing prompts and approvals remain pending.

## Record template

- Tool/model:
- Team member:
- Development stage/time:
- Purpose:
- Representative prompt or prompt-log reference:
- Affected files/sections:
- How AI output was verified:
- Responsible reviewer:
- Verification source (test log, documentation, commit or experiment):

## Source-base initialization

- Tool/model: Codex (GPT-6).
- Team member: to be filled by the user.
- Development stage/time: repository initialization, 2026-09-15.
- Purpose: create the shared course foundation and A1 Linear verification pipeline.
- Prompt reference: user-provided 36-section source-base initialization request in this session.
- Affected files: source, configs, tests, smoke script, documentation and static site.
- Verification: see `VERIFICATION.md`; review code and rerun the documented commands.
- Responsible reviewer: pending team review.
- Verification source: local tests and integration checks; no full experiment results claimed.

## Test coverage and repository hygiene (2026-09-22)

- Tool/model: Claude Opus 5, as named by the commits' `Co-Authored-By` trailers.
- Team member: commits authored by AntoineWu; AI operator confirmation pending.
- Development stage/time: M1 integration, 2026-09-22.
- Purpose: extend smoke coverage to Linear and MLP with a shared-split assertion;
  test the classifier contract and dataset preprocessing/validation; ignore
  generated data and outputs at any directory depth.
- Representative prompt or prompt-log reference: original prompts unavailable;
  commit descriptions provide the recorded scope, not verbatim prompts.
- Affected files: `scripts/smoke_test.py`, `tests/test_models.py`,
  `tests/test_data.py`, `.gitignore`.
- How AI output was verified: model and dataset commit descriptions report
  mutation checks; the later local integration review records 34 passing tests
  and successful real-data smoke checks for both models. These are historical
  verification records, not tests rerun during this documentation update.
- Responsible reviewer: pending team confirmation and approval.
- Verification source: commits `28383b1`, `d9a988e`, `edbb3fa`, `7244fba`;
  `results/a1/integration-review/integration-report.md` (local, ignored).

## Contributions requiring AI-provenance confirmation (2026-09-16 to 2026-09-23)

The repository also contains landing-page/team updates, task planning, EDA and
Linear work, MLP implementation/integration, evaluation changes, and the final
integration review with nonfinite-value guards. The available records do not
establish a tool/model or prompt for each of these contributions. They are not
being attributed to AI solely because they appear in Git history or local files.

- Evidence to review: commits `ffc4e6f`, `99c4e3b`, `bd57e02`, `876f134`,
  `5848426`, `cebacaf`, `64e3912`, `5445324` and the local integration report.
- Team follow-up: for each meaningful AI use, add the actual tool/model, operator,
  prompt reference, affected files, verification and responsible reviewer using
  the template above. Confirm which contributions did not use AI as appropriate.
- Verification context: the local review reports executed EDA and full Linear/MLP
  runs. It also preserves an earlier failed MLP run with NaN losses and nonfinite
  weights. Later successful runs and new guards do not explain the initial
  failure; its cause and submission readiness remain unresolved.

## Landing page and disclosure refresh (2026-09-23)

- Tool/model: Codex (GPT-6).
- Team member: requesting user; identity not confirmed in this session.
- Development stage/time: M1 documentation update, 2026-09-23.
- Purpose: bring the landing page, linked A1 status, README files and AI disclosure
  up to date with implemented features and available verification evidence.
- Representative prompt: "update the landing page and ai usage up to now".
- Affected files: `docs/index.html`, `docs/assignments/a1/index.html`, `README.md`,
  `assignments/a1/README.md`, `AI_USAGE.md`, `VERIFICATION.md`.
- How AI output was verified: inspected source/configs, commit history, existing
  integration review and both local evaluation JSON files; checked reported
  metrics against those artifacts and checked local HTML links and whitespace.
  No model training or Python test suite was rerun for this documentation change.
- Responsible reviewer: pending team review.
- Verification source: current working-tree documentation diff; 2026-09-23
  addendum in `VERIFICATION.md`; local `results/a1/{linear,mlp}/evaluation.json`.
- Limits: generated experiment artifacts are ignored by Git. Results summarized
  on the site do not publish checkpoints or the executed notebook. No submission,
  deployment, team approval or resolution of the earlier MLP failure is claimed.

## M2 development preparation (2026-09-30)

- Tool/model: Codex (GPT-6).
- Team member: requesting user; identity not confirmed in this session.
- Development stage/time: after completed/merged M1, before M2 model implementation.
- Purpose: audit the M1 base, assign M2 ownership, prepare interface/config/test
  TODOs, document the final protocol and prepare synchronization from `dev` into
  the three existing feature branches.
- Representative prompt: user-provided 23-section M2 preparation request specifying
  "prepare, assign, document, synchronize" and prohibiting model implementations.
- Affected files: `assignments/a1/TASKS.md`, `START_HERE.md`, assignment/root README,
  `src/models/a1/{cnn,rnn,transformer}.py`, their configs, model registry,
  `tests/test_models.py`, `tests/test_training.py`, and this log.
- How AI output was verified: `python -m compileall src` passed;
  `.venv\Scripts\python.exe -m pytest --tb=short` passed **34 tests in 4.93s**
  after allowing access to pytest's temporary directory. Initial bare `pytest`
  used a system environment lacking torch; the first project-environment run
  encountered sandbox temp-directory permissions (14 passed, 20 setup errors).
  No tests or M1 logic were changed to bypass those environment issues.
- Additional checks: compared all common config fields across five models;
  imported the three skeleton modules and confirmed they remain unregistered;
  checked all 17 unique task records contain owner, branch, status, dependencies,
  expected output and Definition of Done. Actionable TODO counts in source/config/tests:
  Kim 9, Khoa 9, Antoine 10, shared integration 4. Documentation examples are excluded.
- Responsible reviewer: pending team review.
- Verification source: preparation commit and the command results from this session.
- Limits: no CNN/RNN/Transformer logic, full experiments, new metrics, final
  conclusions, final report/video or error-analysis notebook were produced.
  M1 completion is supplied by the user; historical MLP-failure provenance remains
  tracked for reproducibility review. The dated M1 website/verification records
  are preserved; final webpage updates are assigned in A1-M2-014.
