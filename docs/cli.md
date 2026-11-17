# CLI commands

Install WSM from the repository root in a Python 3.12 or later environment.
The editing extra supplies the model dependencies used by the local examples.

```sh
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install -e 'packages/python[editing]'
wsm --help
```

## Execute a local edit

Use a local Hugging Face causal-language-model snapshot with tokenizer files,
a trusted Python file defining `edit(model)`, and a JSON list of prompt and
target records such as `[{"prompt": "Say hello.", "target": "Hello!"}]`.
Choose an unused output directory with an existing parent. The device defaults
to `cpu`.

```sh
wsm execute /path/to/edit.py --model /path/to/snapshot \
  --cases /path/to/cases.json --output /path/to/new-run
```

WSM runs the file with your process permissions. It evaluates the same cases
before and after the edit. A successful run records both evaluations in
`run.json` and saves the model in `checkpoint/` under the output directory.
Inspect the receipt for a `published` state. See the
[Python editing guide](coding.md) for the edit contract and receipt states.

Pass `--verify-full` to hash every source snapshot file before loading.
The receipt records `source_snapshot_sha256`, and WSM fully verifies the
snapshot before publishing. Keep the output directory outside the source snapshot.

## Open the editing terminal

Run `wsm tui` in an interactive terminal. Running `wsm` without a subcommand
opens the same interface.

## Run an experimental Modal trial

```sh
wsm trial preflight --config /path/to/trial.json
wsm trial run --config /path/to/trial.json --output /path/to/new-trial
wsm trial view /path/to/trial
wsm trial weights /path/to/model.safetensors
```

Preflight validates the configuration offline. The run uses cloud services.
Add `--follow` to `view` to watch the public journal. The `weights` command
inspects a retained safetensors file. Follow the [Modal trial guide](trials.md)
for dependencies, credentials, and cost reservations before running a trial.

## Run an experimental benchmark

```sh
wsm benchmark preflight --config /path/to/config.json
wsm benchmark run --config /path/to/config.json --output /path/to/new-campaign --local-trusted
wsm benchmark report /path/to/campaign
wsm benchmark regrade /path/to/campaign --trial-id task.agent.0000
```

Use `--local-trusted` with the reviewed synthetic fixture. It gives fixture code
your process permissions. For regrading, replace the example trial ID with one
from your campaign. Follow the [benchmark guide](benchmark.md) for fixture
setup, configuration, and offline reports.
