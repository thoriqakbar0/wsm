# WSM (Weight Space Mods)

WSM is an AI harness for modding model weights. An AI agent writes Python edits, runs them against a model, and uses evaluation results to choose its next experiment. WSM provides access to the weights and records the edits, evaluations, and saved checkpoints.

For example, an agent can change a group of weights, test the changed model, and inspect earlier attempts before deciding what to try next. The Python SDK provides the editing operations used in that loop. Finding changes that reliably improve a model remains a research goal.

## Current direction

WSM's goal is to make weight-space editing easier for others to use. The first priority is a reproducible edit-and-evaluate workflow, so others can rerun an experiment and inspect how an edit affected the model.

The SDK already supports evaluation before and after an edit and checkpoint export. The experimental [benchmark workflow](docs/benchmark.md) adds frozen inputs, retained grades, and offline reports. These are the foundation for the current work, not evidence of reliable model improvement.

Easier checkpoint comparison and recovery are planned next. Existing baseline comparisons and saved checkpoints provide building blocks. A workflow for comparing saved checkpoints and recovering from an unwanted edit remains planned work.

## Install WSM

Use Python 3.12 or later. From the repository root, create a virtual environment and install the package:

```sh
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install -e packages/python
python -c 'from wsm import WeightModel, execute_program'
```

The core package has no third-party runtime dependencies. Model editing and experimental agent trials require optional dependencies.

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

Python editing files run with your process permissions and are not sandboxed. Read [Modify weights with Python](docs/coding.md) for API details, execution permissions, and the Modal demonstration.

## Command-line workflows

Use the [Python editing guide](docs/coding.md) for trusted editing files and the [benchmark guide](docs/benchmark.md) for experimental evaluation campaigns.

## Repeatable evaluation benchmark

The experimental [benchmark workflow](docs/benchmark.md) compares unchanged weights, a frozen scripted edit, and an editing policy without student training. It retains independent task and regression grades, accounts for incomplete trials, and rebuilds reports offline. Start with the synthetic local fixture to check mechanics without cloud calls. Synthetic results do not establish editing efficacy.

## Experimental agent trials and Tinker

The experimental [Modal trial workflow](docs/trials.md) connects an editing agent to a model sandbox. The agent can submit Python edits, request public evaluations, and inspect earlier attempts. Each successful edit becomes the current checkpoint, even when its public score is worse. A separate process grades the final model on held-out cases.

WSM's experimental training integration uses the Tinker API to train the agent that writes weight edits. The target model is separate. WSM changes its weights by running the agent's Python code.

The recorded demonstration used a Codex teacher to write edits, then trained a separate open-weight student on the teacher's recorded actions through Modal Spindle using the Tinker SDK. It confirmed one training update, but did not demonstrate better student editing ability. Hosted Tinker support exists in code, but a hosted training update has not been verified. Tinker is optional for the Python SDK.

The [Jupyter and marimo examples](examples/notebooks/README.md) include uv setup instructions and inspect the recorded experiment offline by default.

## Python API

The supported Python API consists of `WeightModel` and `execute_program`. WSM no longer includes search, sessions, prompt harnesses, or the local Tinker bridge. It does not support existing experiment bundles.

See the [architecture](docs/architecture.md) and [research limitations](docs/research.md).
