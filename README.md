# WSM (Weight Space Modifiers)

WSM lets an AI teacher inspect a model's weights, propose edits, and test them. It saves the selected checkpoint and evaluation evidence.

## Install WSM

Use Python 3.12 or later. From the repository root, create a virtual environment and install the package:

```sh
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

A completed run saves `checkpoint.json`, `mod.json`, and `harness/report.json`. The file `harness/bundle.json` contains public evidence. Keep the run directory private because it also contains teacher requests and recovery state.

The bundled addition task has only public examples, so its final conclusion is `inconclusive`. For custom tasks, private grading, and recovery, follow the [prompt-driven guide](docs/wsm.md).

To open the terminal UI, run `wsm` without arguments.

## Choose another workflow

Use these guides for other tasks:

- [Python SDK](docs/contract.md#intended-caller). Inspect a checkpoint, submit edits, and replay a frozen run.
- [Configured experiments](docs/driver.md). Run scripted proposals or a teacher from a JSON configuration and export the selected checkpoint.
- [Sessions and model targets](docs/client.md). Resume experiments and work with binary64 or supported local model snapshots.
- [Harbor grading](docs/contract.md#native-harbor-boundary). Use the optional Docker-based evaluator.
- [Verification](docs/verification.md). Run the core checks and check the requirements for native sandbox and Harbor checks.

For the research context and limits of the evidence, read [research scope](docs/research.md). The offline example tests edits and selection on a synthetic task. It does not reproduce the paper's results or establish that an AI discovered a useful model edit.

The [Astro documentation site](site/README.md) provides a browsable guide. Run `npm ci` and `npm run dev` from `site/` to start it locally.

The distribution, import package, and command are named `wsm`. If you used the former `weightless-sdk` package, uninstall it and update imports from `weightless` to `wsm`. Serialized `weightless.*` identifiers and hashes, `WeightlessError`, environment variables, and Docker image names keep their existing spelling.
