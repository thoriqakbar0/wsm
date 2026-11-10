# WSM (Weight Space Modifiers)

WSM is a Python SDK for weight-space modification. Write Python that accesses named tensors, changes weights, evaluates the result, and saves a standalone checkpoint. Coding agents use the same API as developers. The Python distribution, import package, and CLI are named `wsm`.

## Install WSM

Use Python 3.12 or later. From the repository root, create a virtual environment and install the package:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e packages/python
wsm --help
```

The core package has no third-party runtime dependencies.

## Modify weights with Python

Install the optional model dependencies before loading a local model snapshot:

```sh
python3 -m pip install -e 'packages/python[editing]'
```

```python
from wsm import WeightModel

model = WeightModel.load("/path/to/local/snapshot")
model.weights["model.norm.weight"][:16].mul_(0.95)
result = model.evaluate([{"prompt": "Say hello.", "target": "Hello!"}])
model.save("/path/to/new-checkpoint")
```

Create the parent directory with `mkdir -p runs`, then use `wsm execute edit.py --model /path/to/snapshot --cases examples/coding/cases.json --output runs/edit` for a Python file defining `edit(model)`. Open `wsm` and press **p** for the same workflow in the terminal. Read [Modify weights with Python](docs/coding.md) for execution trust, the Modal proof, and open policy decisions. The coding-first API is independent of agent frameworks.


The supported Python API consists of `WeightModel` and `execute_program`. Legacy search, sessions, prompt harnesses, and Tinker integration have been removed. Existing experiment bundles are not supported.

See [CLI commands](docs/cli.md), [architecture](docs/architecture.md), and [verification](docs/verification.md).
