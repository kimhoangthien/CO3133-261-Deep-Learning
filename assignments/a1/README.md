# Assignment 1: image classification

Main dataset: Fashion-MNIST. MNIST is debugging-only; CIFAR-10 is optional.
Linear, MLP and the shared training/evaluation pipeline are implemented. CNN,
LSTM/GRU and Transformer remain TODO. Keep seed, split, normalization and evaluation
protocol identical for the main comparison.

**30 September 2026: M1 Draft completed and merged; M2 Final in progress.**
Start at [START_HERE.md](START_HERE.md). [TASKS.md](TASKS.md) defines the 17 M2
tasks: Kim owns RNN/reproducibility, Khoa CNN/training analysis, and Antoine
Transformer/error analysis. The existing feature branches use `dev` as their
integration source. Model skeletons are intentionally unimplemented/unregistered.

As of 23 September 2026, the local integration review records executed EDA,
34 passing tests, both-model smoke checks, and full-data CPU test accuracy of
84.06% (Linear) and 87.59% (MLP). An earlier MLP NaN failure remains unexplained;
finite-value guards and successful reruns do not resolve its cause. Submission
review was open in that historical record; remaining provenance questions are
tracked in M2 reproducibility/QA. See [verification](../../VERIFICATION.md) and the
[A1 page](../../docs/assignments/a1/index.html).

## Remaining work

- Incorporate findings from the executed EDA copy into `notebooks/eda.ipynb`: split, class-balance, and
  representative-sample findings in its final section.
- Investigate the initial MLP nonfinite run and complete team review.
- Implement the three remaining model families and tune using validation data only.
- Extend full experiments beyond Linear/MLP and repeat timing measurements.
- Plot saved training/validation history and confusion matrices outside training.
- Collect correct/incorrect examples and explain representations/inductive biases.
- Write limitations, report/slides and presentation; update the website and AI log.

Smoke-test measurements are infrastructure checks, not assignment results.
