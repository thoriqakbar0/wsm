# WSM (Weight Space Modifiers)

WSM is a Python SDK for weight-space modification. Write Python that accesses named tensors, changes weights, evaluates the result, and saves a standalone checkpoint. Coding agents use the same API as developers. The Python distribution, import package, and CLI are named `wsm`.

Start with the offline example to test the experiment loop without downloading a model or configuring credentials. To edit a local model snapshot, use the [Python workflow](#modify-weights-with-python). The tensor SDK is experimental. See [research scope](docs/research.md) for what the current evidence establishes.

## Install WSM

Use Python 3.12 or later. Clone the repository, then create a virtual environment and install the package from source:

```sh
git clone https://github.com/thoriqakbar0/wsm.git
cd wsm
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e packages/python
wsm --help
```

The core package has no third-party runtime dependencies.

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

## Modify weights with Python

Install the optional model dependencies before loading a local model snapshot:

```sh
python3 -m pip install -e 'packages/python[editing]'
```

```python
from wsm import WeightModel

model = WeightModel.load("/path/to/local/snapshot")
cases = [{"prompt": "Say hello.", "target": "Hello!"}]
before = model.evaluate(cases)
model.weights["model.norm.weight"][:16].mul_(0.95)
after = model.evaluate(cases)
model.save("/path/to/new-checkpoint")
```

Use a snapshot whose tensors include `model.norm.weight`, and choose a new output directory. Editing files run with your process privileges. Execute only trusted Python.

Use `wsm execute edit.py --model /path/to/snapshot --cases examples/coding/cases.json --output runs/edit` for a Python file defining `edit(model)`. Open `wsm` and press **p** for the same workflow in the terminal. Read [Modify weights with Python](docs/coding.md) for execution trust, the Modal proof, and open policy decisions. The coding-first API is independent of agent frameworks. The separate [local Tinker backend](docs/tinker.md#local-sdk-backend) provides bounded compatibility with the Tinker SDK.

The prompt harness and durable scalar-session APIs below remain available for their existing experiment contracts.

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

The teacher sends one command at a time. For example:

```json
{"action":"inspect","tensor":"output.weight","indices":[0,1]}
{"action":"edit","edits":[{"tensor":"output.weight","index":0,"value":1}]}
```

Add `--code-executions N` to let the teacher run bounded Python over disclosed coefficients and submit its diff through the session policy. The [local Tinker backend](docs/tinker.md#local-sdk-backend) exposes this editing path alongside SDK sampling and prompt scoring.

A completed run saves `checkpoint.json`, `mod.json`, and `harness/report.json`. The file `harness/bundle.json` contains public evidence. Keep the run directory private because it also contains teacher requests and recovery state.

The bundled addition task has only public examples, so its final conclusion is `inconclusive`. For custom tasks, private grading, and recovery, follow the [prompt-driven guide](docs/wsm.md).

To open the terminal UI, run `wsm` without arguments.

## Choose another workflow

Use these guides for other tasks:

- [Coding-first Python SDK](docs/coding.md). Modify tensors with Python, compare results, and save a standalone checkpoint.
- [Scalar-session SDK](docs/contract.md#intended-caller). Inspect a checkpoint, submit edits, and replay a frozen run.
- [Configured experiments](docs/driver.md). Run scripted proposals or a teacher from a JSON configuration and export the selected checkpoint.
- [Sessions and model targets](docs/client.md). Resume experiments and work with binary64 or supported local model snapshots.
- [Local Tinker backend](docs/tinker.md#local-sdk-backend). Use the Tinker SDK for sampling and scoring, and the WSM extension for isolated Python coefficient edits.
- [Harbor grading](docs/contract.md#native-harbor-boundary). Use the optional Docker-based evaluator.
- [Verification](docs/verification.md). Run the core checks and check the requirements for native sandbox and Harbor checks.

For the research context and limits of the evidence, read [research scope](docs/research.md). The offline example tests edits and selection on a synthetic task. It does not reproduce the paper's results or establish that an AI discovered a useful model edit.

The [Astro documentation site](site/README.md) provides a browsable guide. Run `npm ci` and `npm run dev` from `site/` to start it locally.

The distribution, import package, and command are named `wsm`. If you used the former `weightless-sdk` package, uninstall it and update imports from `weightless` to `wsm`. Serialized `weightless.*` identifiers and hashes, `WeightlessError`, environment variables, and Docker image names keep their existing spelling.
## Contribute and verify

For fixes, include a regression test when practical. For behavior changes, update the relevant guide under `docs/` and add an entry under `Unreleased` in [the changelog](CHANGELOG.md).

After installation, run the Python tests and checkpoint identity check from the repository root:

```sh
python3 -m unittest discover -s packages/python/tests -v
python3 scripts/golden_identity.py
```

Some tests require optional dependencies or a configured Linux sandbox. Follow [the verification guide](docs/verification.md) for native sandbox, installed-package, and Harbor checks. GitHub Actions is currently disabled, and the [saved workflow](docs/workflows/README.md) does not run automatically.

## License and publication status

This repository has no license file yet. A license must be selected before an open-source release. See [the readiness review](docs/open-source-readiness.md) for remaining preparation work.
