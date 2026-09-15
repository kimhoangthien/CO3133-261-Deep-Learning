# Assignment 1: image classification

Main dataset: Fashion-MNIST. MNIST is debugging-only; CIFAR-10 is optional.
The Linear classifier and shared pipeline are implemented. MLP, CNN, LSTM/GRU,
and Transformer remain TODO. Keep seed, split, normalization and evaluation
protocol identical for the main comparison.

## Remaining work

- EDA in `notebooks/eda.ipynb` (create when analysis begins).
- Implement the four remaining model families and tune using validation data only.
- Full experiments: accuracy, macro-F1, parameters, training/inference timing.
- Plot saved training/validation history and confusion matrices outside training.
- Collect correct/incorrect examples and explain representations/inductive biases.
- Write limitations, report/slides and presentation; update the website and AI log.

Smoke-test measurements are infrastructure checks, not assignment results.
