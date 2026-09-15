# Project tree

Source inventory; runtime outputs are ignored and created on demand.

```text
.
|-- assignments/
|   |-- a1/
|   |   `-- README.md
|   |-- a2/
|   |   `-- README.md
|   `-- a3/
|       `-- README.md
|-- configs/
|   |-- a1/
|   |   |-- cnn.yaml
|   |   |-- linear.yaml
|   |   |-- mlp.yaml
|   |   |-- rnn.yaml
|   |   `-- transformer.yaml
|   |-- a2/
|   |   `-- README.md
|   `-- a3/
|       `-- README.md
|-- docs/
|   |-- assets/
|   |   `-- css/
|   |       `-- style.css
|   |-- assignments/
|   |   |-- a1/
|   |   |   `-- index.html
|   |   |-- a2/
|   |   |   `-- index.html
|   |   `-- a3/
|   |       `-- index.html
|   `-- index.html
|-- scripts/
|   `-- smoke_test.py
|-- src/
|   |-- core/
|   |   |-- __init__.py
|   |   |-- checkpoint.py
|   |   |-- config.py
|   |   |-- device.py
|   |   |-- logger.py
|   |   `-- seed.py
|   |-- datasets/
|   |   |-- __init__.py
|   |   |-- fashion_mnist.py
|   |   `-- registry.py
|   |-- models/
|   |   |-- a1/
|   |   |   |-- __init__.py
|   |   |   `-- linear.py
|   |   |-- a2/
|   |   |   `-- __init__.py
|   |   |-- a3/
|   |   |   `-- __init__.py
|   |   |-- __init__.py
|   |   `-- registry.py
|   |-- tasks/
|   |   |-- classification/
|   |   |   |-- __init__.py
|   |   |   |-- evaluator.py
|   |   |   |-- metrics.py
|   |   |   `-- trainer.py
|   |   |-- __init__.py
|   |   `-- registry.py
|   |-- __init__.py
|   |-- evaluate.py
|   `-- train.py
|-- tests/
|   |-- conftest.py
|   |-- test_data.py
|   |-- test_models.py
|   `-- test_training.py
|-- .gitignore
|-- AI_USAGE.md
|-- PROJECT_TREE.md
|-- README.md
|-- VERIFICATION.md
|-- pytest.ini
`-- requirements.txt
```

Runtime: `data/`, `checkpoints/<assignment>/<experiment>/`,
`logs/<assignment>/<experiment>/`, `results/<assignment>/<experiment>/`.
