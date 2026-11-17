# Modify weights with Python

Write a Python file that defines `edit(model)`. The model exposes named PyTorch tensors through `model.weights`.

```python
# edit.py
def edit(model):
    model.weights["model.norm.weight"][:16].mul_(0.95)
```

## Checkpoint requirements

Before running `WeightModel.save()` or an edit with `wsm execute`, choose an
unused output path whose parent directory exists. Use Linux with a C library
that exposes `renameat2`, and choose an output filesystem that supports
`renameat2(RENAME_NOREPLACE)` for directories. The flag publishes the verified
checkpoint atomically and refuses to replace an existing destination.

WSM stages the checkpoint beside its destination and syncs staged files and
directories before publication. For `wsm execute`, the checkpoint is inside
the new run directory. Use a compatible local filesystem if your mounted
volume rejects exclusive rename, then copy the published checkpoint to the volume.

WSM attempts the rename after staging and verification. A missing
`renameat2` symbol raises `unsupported_platform`. A destination that appears
before publication raises `export_exists`. Other rename errors, including
unsupported filesystem operations, surface as OS errors. Loading and
evaluating a model do not use this publication step.

## Run an editing file from the repository

These commands use a source checkout. For the published SDK workflow, use the [Python quickstart](/docs/quickstart/).

Run the program against a local Hugging Face snapshot with its tokenizer files. From the repository root, install the optional model dependencies in your Python environment first.

```sh
uv pip install -e 'packages/python[editing]'
mkdir -p runs
wsm execute edit.py --model /path/to/snapshot \
  --cases examples/coding/cases.json --output runs/python-edit --device cpu
```

`--cases` requires a JSON array of public prompt/target pairs. The runner evaluates the unchanged model, executes your function, evaluates the changed model on the same cases, and saves a standalone checkpoint under `runs/python-edit/checkpoint`. `run.json` records execution state and comparison evidence. The runner does not automatically select a winner from the loss comparison. A deterministic norm edit checks mechanics; a changed loss does not establish a capability improvement.

The receipt records `case_count` and `cases_sha256` without storing prompt or target text in the case evidence. To reproduce the digest, hash the UTF-8 bytes of the normalized case list serialized with Python's `json.dumps(cases, sort_keys=True, ensure_ascii=True, separators=(",", ":"))` and prefix the hex digest with `sha256:`. Keep the original case order and use `[]` when cases are omitted.

The first load hashes every file in the source snapshot, including the contents of symlinked files. WSM stores a manifest under `$XDG_CACHE_HOME/wsm/snapshots` or `~/.cache/wsm/snapshots`. Later loads reuse a file's hash when its size, nanosecond modification time, and inode match. Pass `--verify-full` to hash every file before loading.

The loader reads a private copy, which needs temporary disk space for the snapshot while the model is alive. Before publishing a checkpoint, WSM hashes both the copy and the current source and requires them to match the identity captured at load. This also catches a stale cached hash if the source changes and is then restored. Keep the run output and cache outside the source snapshot.

The `source_snapshot_sha256` field in `run.json` identifies the source independently of its location. To reproduce it, compute the lowercase SHA-256 of each file and build a JSON object mapping POSIX relative paths to those hex digests. Sort the keys, use ASCII escaping and compact separators, and hash the UTF-8 JSON with no trailing newline. Prefix the resulting hex digest with `sha256:`.

For CUDA, pass `--device cuda` in a runtime with CUDA-enabled Torch and a GPU. The TUI opens with `wsm`; press **p** to review and execute a Python editing file using the same command. Choose an unused output directory for each run. The TUI supports Ctrl-C interruption, displays a receipt after a successful command, and shows command output on failure.

## Try a synthetic model offline

From a repository checkout, follow the [download-free SDK quickstart](../README.md#modify-weights-with-python).
It creates a tiny GPT-2 model and tokenizer locally, evaluates a tensor edit,
and verifies that the edited tensor survives a save and reload.

## Use the SDK directly

```python
from wsm import WeightModel

model = WeightModel.load("/path/to/snapshot", device="cpu")
cases = [{"prompt": "Say hello.", "target": "Hello!"}]
before = model.evaluate(cases)
model.weights["model.norm.weight"][:16].mul_(0.95)
after = model.evaluate(cases)
model.save("/path/to/new-checkpoint")
reloaded = WeightModel.load("/path/to/new-checkpoint")
```

Tensor names, shapes, slicing, and arithmetic are ordinary PyTorch. Tied tensors share storage. Model loading is offline and does not execute model repository Python. Pass `verify_full=True` to `WeightModel.load` or `execute_program` to force a full scan at load. Saving writes Hugging Face safetensors and tokenizer files, checks finite tensors, reloads and compares the tensors, verifies the original snapshot, and publishes the directory atomically without overwriting an existing destination.

A snapshot must contain every required weight. Transformers can recover a missing tied alias from its saved counterpart. If required weights are still missing, `WeightModel.load` raises `WeightlessError` with code `invalid_model` and names the missing weights.

Indexed CPU devices such as `cpu:0` resolve to `cpu`. The `meta` device cannot store weights, so `WeightModel.load` raises `unsupported_target` before reading the snapshot. Use `cpu` or an available accelerator.

The supported API consists of `WeightModel` and `execute_program`. The runner records comparisons without automatic candidate selection.

## Python execution permissions

The Python runner executes trusted code with the invoking process's privileges. It is not a sandbox or an allowlist of tensor operations. The Modal demonstration also executes a reviewed deterministic program. Do not treat the SDK's tensor mapping as a security boundary.

Arbitrary Python can access files, import installed packages, and change process state. The standalone runner uses the invoking process's permissions. The experimental [Modal trial runner](trials.md) supplies a separate network-blocked sandbox, host-owned history, retained checkpoints, and private final grading. The SDK imports no agent framework. Coding agents can author a Python file and invoke the same command or call the SDK through their own execution environment.

Evaluation cases are public input to the editing process. Independent private grading belongs in a separate process with data the author cannot read. Neither the runner nor its recorded comparison establishes private-data isolation, authenticated evidence, or improved model quality.

## Reproduce the Modal proof

```sh
python scripts/coding_modal.py \
  --model-source /path/to/models--HuggingFaceTB--SmolLM2-135M-Instruct \
  --output /tmp/wsm-coding-modal-proof
```

The edit invocation checks the exact modified tensor and element count, fingerprints every unchanged tensor, compares actual loss against an unchanged model, and verifies that the original snapshot is unchanged. A second image contains only the SDK, example cases, and saved checkpoint through an ephemeral volume. It has neither the original source nor a Hugging Face cache. It checks every tensor fingerprint, tied embeddings, and evaluation parity. The script downloads the published checkpoint to the local output directory before the ephemeral volume closes.

The lifecycle and its limits are described in [the execution statechart](design/coding-statechart.md).

Evaluation encodes prompt and target separately with special tokens disabled and concatenates their token IDs. It reports mean target-token negative log likelihood for each case, then the unweighted mean across cases. It does not apply a chat template or generate text. Empty cases return a null aggregate loss. Receipt states use lowercase names; the chart's **Failed** label corresponds to `failure` in `run.json`.

The Modal launcher uses serialized functions. Run it with Python 3.12 to match the declared image. A Python 3.13 launcher is incompatible with that image.

The proof publishes the SDK checkpoint on the worker's local filesystem before copying it to the ephemeral volume for transport. Modal's mounted volume rejected `renameat2(RENAME_NOREPLACE)` during the first GPU attempt. A destination filesystem must support exclusive atomic rename for `save`; a volume copy is not that publication boundary.

Open `wsm` and press **p** to execute a Python editing file.
