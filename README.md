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

To save a checkpoint, use Linux with a C library that exposes `renameat2` and an
output filesystem that supports `RENAME_NOREPLACE`. Check the
[checkpoint requirements](docs/coding.md#checkpoint-requirements) before running an edit that saves.

## Modify weights with Python

Install the optional model dependencies:

```sh
uv pip install -e 'packages/python[editing]'
```

From the repository root, generate a tiny random GPT-2 model and tokenizer.
Once the dependencies are installed, this walkthrough runs without model downloads
or cloud services. Use Linux and a checkpoint filesystem that supports
`renameat2(RENAME_NOREPLACE)`.

```sh
mkdir -p runs
demo_dir=$(mktemp -d runs/sdk-demo.XXXXXX)
python scripts/benchmark_fixture.py --output "$demo_dir/fixture"
```

In the same shell, evaluate the model, scale a named tensor, then save and
reload the edited checkpoint.

```sh
python - "$demo_dir" <<'PYTHON'
import json
from pathlib import Path
import sys

from wsm import WeightModel

demo = Path(sys.argv[1])
fixture = demo / "fixture"
cases = json.loads((fixture / "public.json").read_text())
model = WeightModel.load(fixture / "model")
before = model.evaluate(cases)
tensor = "transformer.ln_f.weight"
original = model.weights[tensor].clone()
model.weights[tensor].mul_(0.95)
after = model.evaluate(cases)
checkpoint = demo / "checkpoint"
model.save(checkpoint)
reloaded = WeightModel.load(checkpoint)
assert not reloaded.weights[tensor].equal(original)
assert reloaded.weights[tensor].equal(model.weights[tensor])
print({"baseline": before, "candidate": after, "checkpoint": str(checkpoint)})
PYTHON
```

The output includes both evaluations and the path to the reloaded checkpoint.
Evaluation measures target-token negative log likelihood. The synthetic model
demonstrates the editing and save workflow.

Python editing files run with your process permissions and are not sandboxed. Read [Modify weights with Python](docs/coding.md) for API details, execution permissions, and the Modal demonstration.

## Command-line workflows

Use the [Python editing guide](docs/coding.md) for trusted editing files and the [benchmark guide](docs/benchmark.md) for experimental evaluation campaigns.

## Use WSM with a coding agent

Copy this prompt into a coding agent with terminal access to this checkout. Replace the placeholders first. Use a local model snapshot and public evaluation cases you are allowed to share with the agent.

```text
Use WSM in this repository to run a bounded weight-editing experiment.

Model snapshot: <absolute path to local Hugging Face snapshot with tokenizer files>
Public cases: <path to JSON prompt/target cases>
Goal: <the behavior to investigate>
Device: <cpu or cuda>
Output directory: <new, unused path whose parent exists>
Maximum candidate edits: <number>

Read README.md and docs/coding.md before acting. Check the Python environment
and install the documented editing dependencies if needed. Confirm the model
and cases exist. Inspect tensor names and shapes before choosing an edit.

Write a small, reviewable Python file defining edit(model). Explain the edit
and review its code before execution. Run it with wsm execute, using the model,
cases, device, and a fresh output directory for each candidate. Start each
candidate from the original snapshot and leave that snapshot unchanged.
Do not download models, call cloud services, or start training.

Inspect each run.json for execution state and baseline/candidate evaluations.
Report failures as failures. Compare public target-token negative log likelihood
and report regressions. Do not claim that lower loss proves better generated
responses or general model quality. Keep any private grading data outside the
editing agent's process and context.

Respect the candidate limit. Do not overwrite existing outputs or automatically
promote a candidate. Report the edit files, commands, receipts, checkpoint paths,
results, and limitations so I can reproduce the experiment. If a prerequisite
is missing, report the blocker instead of inventing a successful result.
```

The agent writes and runs Python. WSM loads the target model, exposes its weights, evaluates the edit, and saves a checkpoint. The local runner executes Python with your process permissions, so use trusted code. See the [Python execution permissions](docs/coding.md#python-execution-permissions).

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
