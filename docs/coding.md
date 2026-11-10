# Modify weights with Python

Write a Python file that defines `edit(model)`. The model exposes named PyTorch tensors through `model.weights`.

```python
# edit.py
def edit(model):
    model.weights["model.norm.weight"][:16].mul_(0.95)
```

Run the program against a local Hugging Face snapshot. Install the optional model dependencies in your Python environment first.

```sh
python -m pip install -e 'packages/python[editing]'
mkdir -p runs
wsm execute edit.py --model /path/to/snapshot \
  --cases examples/coding/cases.json --output runs/python-edit --device cpu
```

`--cases` supplies public prompt/target pairs. The runner evaluates the unchanged model, executes your function, evaluates the changed model on the same cases, and saves a standalone checkpoint under `runs/python-edit/checkpoint`. `run.json` records execution state and comparison evidence. The runner does not automatically select a winner from the loss comparison. A deterministic norm edit checks mechanics; a changed loss does not establish a capability improvement.

For CUDA, pass `--device cuda` in a runtime with CUDA-enabled Torch and a GPU. The TUI opens with `wsm`; press **p** to review and execute a Python editing file using the same command. It displays execution state and supports Ctrl-C interruption.

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

Tensor names, shapes, slicing, and arithmetic are ordinary PyTorch. Tied tensors share storage. Model loading is offline and does not execute model repository Python. Saving writes Hugging Face safetensors and tokenizer files, checks finite tensors, reloads and compares the tensors, and publishes the directory atomically without overwriting an existing destination.

The supported API consists of `WeightModel` and `execute_program`. The runner records comparisons without automatic candidate selection.

## Execution boundary and open decisions

The Python runner executes trusted code with the invoking process's privileges. It is not a sandbox or an allowlist of tensor operations. The Modal demonstration also executes a reviewed deterministic program. Do not treat the SDK's tensor mapping as a security boundary.

The operator has not selected a supported-operation policy, execution-permission model, compatibility guarantee, or spending limit. Arbitrary Python can access files, import installed packages, and change process state. A future coding-agent runner must supply the chosen execution boundary separately. The SDK imports no Claude, OpenAI, or DSPy framework. Those frameworks can author a Python file and invoke the same command or call the SDK through their own execution environment.

Evaluation cases are public input to the editing process. Independent private grading belongs in a separate process with data the author cannot read. Neither the new runner nor its recorded comparison claims private-data isolation, authenticated evidence, or scientific efficacy.

## Reproduce the Modal proof

```sh
python scripts/coding_modal.py \
  --model-source /path/to/models--HuggingFaceTB--SmolLM2-135M-Instruct \
  --output /tmp/wsm-coding-modal-proof
```

Use a Python runtime with Modal installed and configured. The source must contain pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The script performs one T4 edit invocation and one CPU reload invocation, each with a 600-second timeout and no application retries. These are demonstration bounds, not an account spending cap or a product policy.

The edit invocation checks the exact modified tensor and element count, fingerprints every unchanged tensor, compares actual loss against an unchanged model, and verifies that the original snapshot is unchanged. A second image contains only the SDK, example cases, and saved checkpoint through an ephemeral volume. It has neither the original source nor a Hugging Face cache. It checks every tensor fingerprint, tied embeddings, and evaluation parity. The script downloads the published checkpoint to the local output directory before the ephemeral volume closes.

The lifecycle and its limits are described in [the execution statechart](design/coding-statechart.md).

Evaluation encodes prompt and target separately with special tokens disabled and concatenates their token IDs. It reports mean target-token negative log likelihood for each case, then the unweighted mean across cases. It does not apply a chat template or generate text. Empty cases return a null aggregate loss. Receipt states use lowercase names; the chart's **Failed** label corresponds to `failure` in `run.json`.

The Modal launcher uses serialized functions. Run it with Python 3.12 to match the declared image. A Python 3.13 launcher is incompatible with that image.

The proof publishes the SDK checkpoint on the worker's local filesystem before copying it to the ephemeral volume for transport. Modal's mounted volume rejected `renameat2(RENAME_NOREPLACE)` during the first GPU attempt. A destination filesystem must support exclusive atomic rename for `save`; a volume copy is not that publication boundary.

Open `wsm` and press **p** to execute a Python editing file.
