# WSM (Weight Space Mods)

WSM is a Python SDK for editing model weights. Use Python to access named tensors, change weights, evaluate the result, and save a standalone checkpoint. Developers and coding agents use the same API. The Python distribution, import package, and CLI are named `wsm`.

## Install WSM

Use Python 3.12 or later. From the repository root, create a virtual environment and install the package:

```sh
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install -e packages/python
wsm --help
```

The core package has no third-party runtime dependencies.

## Modify weights with Python

Install the optional model dependencies:

```sh
uv pip install -e 'packages/python[editing]'
```

Use a local Hugging Face causal-language-model snapshot with its tokenizer files. WSM loads it without downloading files or executing model repository Python. Replace the paths below with your snapshot and an unused checkpoint destination whose parent directory exists.

```python
from wsm import WeightModel

model = WeightModel.load("/path/to/local/snapshot")
cases = [{"prompt": "Say hello.", "target": "Hello!"}]
before = model.evaluate(cases)
model.weights["model.norm.weight"][:16].mul_(0.95)
after = model.evaluate(cases)
print({"baseline": before, "candidate": after})
model.save("/path/to/new-checkpoint")
```

The example requires a tensor named `model.norm.weight`. Inspect `model.weights` for your model's names and shapes before choosing an edit. Scaling this slice demonstrates the API, not an improvement in model quality. Evaluation reports target-token negative log likelihood, not generated responses.

To run the same edit with the CLI, save this as `edit.py`:

```python
def edit(model):
    model.weights["model.norm.weight"][:16].mul_(0.95)
```

From the repository root, create the output parent directory and run the file. Replace the snapshot path and choose an output directory that does not exist:

```sh
mkdir -p runs
wsm execute edit.py --model /path/to/snapshot --cases examples/coding/cases.json --output runs/edit
```

The runner evaluates the unchanged model and the edited model on the same cases. It saves the checkpoint in `runs/edit/checkpoint/` and records both evaluations in `runs/edit/run.json`. You decide whether to keep the edit. Python editing files run with your process permissions and are not sandboxed.

To run an edit from the terminal UI, open `wsm` and press **p**. Read [Modify weights with Python](docs/coding.md) for API details, execution permissions, and the Modal demonstration. WSM does not require an agent framework.

The supported Python API consists of `WeightModel` and `execute_program`. WSM no longer includes search, sessions, prompt harnesses, or the local Tinker bridge. It does not support existing experiment bundles.

See [CLI commands](docs/cli.md), [architecture](docs/architecture.md), and [verification](docs/verification.md).

The experimental [Modal trial workflow](docs/trials.md) runs persistent agent edits with independent grading, a Codex teacher, and a Spindle student. The [Jupyter and marimo examples](examples/notebooks/README.md) include uv setup instructions.
