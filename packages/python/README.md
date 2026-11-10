# WSM (Weight Space Mods)

WSM is a Python SDK for model weight editing. Access named tensors, change weights, evaluate the result, and save a standalone checkpoint.

Requires Python 3.12 or later. The core package has no third-party runtime dependencies.

```sh
pip install wsm
wsm --help
```

Install the model dependencies to load a local snapshot:

```sh
pip install 'wsm[editing]'
```

```python
from wsm import WeightModel

model = WeightModel.load("/path/to/local/snapshot")
model.weights["model.norm.weight"][:16].mul_(0.95)
result = model.evaluate([{"prompt": "Say hello.", "target": "Hello!"}])
model.save("/path/to/new-checkpoint")
```

The supported Python API consists of `WeightModel` and `execute_program`. Read the [project documentation](https://github.com/thoriqakbar0/wsm#readme) for CLI usage, execution trust, and optional Modal and Spindle trials. Spindle requires a separate pinned source installation documented in the [trial guide](https://github.com/thoriqakbar0/wsm/blob/main/docs/trials.md).
