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

Use `wsm execute edit.py --model /path/to/snapshot --cases examples/coding/cases.json --output runs/edit` for a Python file defining `edit(model)`. Open `wsm` and press **p** for the same workflow in the terminal. Read [Modify weights with Python](docs/coding.md) for execution trust, the Modal proof, and open policy decisions. The SDK is independent of agent frameworks and makes no Tinker compatibility claim.

The prompt harness and durable scalar-session APIs below remain available for their existing experiment contracts.

## Run the offline example

From the repository root, run the scripted signed-addition example:

```sh
python3 examples/demo.py --output /tmp/wsm-run.json
```

The scripted teacher edits a small signed-addition network without model calls, credentials, Docker, or bubblewrap. It prints these decisions:

```json
{"accepted": false, "reason": "not_improved"}
{"accepted": true, "reason": "improved"}
{"accepted": false, "reason": "not_improved"}
```

The saved file contains the selected checkpoint and evidence. To test other edits, pass `--proposals` with a file in the format of [the bundled proposals](examples/teacher-proposals.json).

## Run with an AI teacher

Configure these environment variables in your shell or secret manager:

- `WSM_TEACHER_URL` is the Responses-compatible base URL. The endpoint must support streaming responses.
- `WSM_TEACHER_MODEL` is the model ID.
- `WSM_TEACHER_API_KEY` is the endpoint credential.

The default runner requires a root-authorized Linux host with systemd and bubblewrap. Follow the [sandbox setup](docs/sandbox.md#run-on-this-host). To use Docker instead, add `--runner docker --image <existing-image>` to the run command.

Once the teacher and runner are configured, run the bundled addition task with a new output directory:

```sh
wsm run "Improve signed addition. Inspect the weights and test your changes." \
	--task addition --attempts 20 --output runs/addition
wsm status runs/addition
```

A completed run saves `checkpoint.json`, `mod.json`, and `harness/report.json`. The file `harness/bundle.json` contains public evidence. Keep the run directory private because it also contains teacher requests and recovery state.

The bundled addition task has only public examples, so its final conclusion is `inconclusive`. For custom tasks, private grading, and recovery, follow the [prompt-driven guide](docs/wsm.md).

To open the terminal UI, run `wsm` without arguments.

## Choose another workflow

Use these guides for other tasks:

- [Coding-first Python SDK](docs/coding.md). Modify tensors with Python, compare results, and save a standalone checkpoint.
- [Scalar-session SDK](docs/contract.md#intended-caller). Inspect a checkpoint, submit edits, and replay a frozen run.
- [Configured experiments](docs/driver.md). Run scripted proposals or a teacher from a JSON configuration and export the selected checkpoint.
- [Sessions and model targets](docs/client.md). Resume experiments and work with binary64 or supported local model snapshots.
- [Harbor grading](docs/contract.md#native-harbor-boundary). Use the optional Docker-based evaluator.
- [Verification](docs/verification.md). Run the core checks and check the requirements for native sandbox and Harbor checks.

For the research context and limits of the evidence, read [research scope](docs/research.md). The offline example tests edits and selection on a synthetic task. It does not reproduce the paper's results or establish that an AI discovered a useful model edit.
