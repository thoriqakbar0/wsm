# Run WSM in Jupyter or marimo

Run the same Codex teacher, Modal tensor-editing trial, hidden grader, and Spindle student update from either notebook. Both examples inspect the checked-in proof offline until you explicitly request a live run. They call [`scripts/spindle_poc.py`](../../scripts/spindle_poc.py), so the existing owner, budgets, records, and cleanup watchdog remain in control.

Use Linux, `uv`, and Python 3.12. Run the commands below from the repository root. The notebook project installs the local WSM package in editable mode. The `live` extra adds the existing `editing` and `spindle` dependencies. No notebook contains package installation magics or embedded credentials.

## Open Jupyter

```sh
uv sync --project examples/notebooks --locked
uv run --project examples/notebooks jupyter lab examples/notebooks/wsm_trial.ipynb
```

Select the Python 3 kernel from this environment. **Run All** reads the public verification receipt and checks the bounded owner's CLI without contacting Modal or the proxy. The source notebook has no saved outputs.

## Open marimo

```sh
uv run --project examples/notebooks marimo edit examples/notebooks/wsm_trial_marimo.py
```

The marimo file also has PEP 723 metadata for a standalone uv environment:

```sh
uv run examples/notebooks/wsm_trial_marimo.py
```

Its cloud cell requires both `WSM_NOTEBOOK_LIVE=1` and a click on **Run one authorized cloud trial**. Script execution and HTML export leave the button unpressed. The project-based launch gives Jupyter and marimo the same locked dependencies.

## Prepare one live trial

The original proof already reserved US$9.927083 of its US$10 allocation. That remaining allocation cannot fund another complete trial. A notebook run requires an authorized allocation and an explicit current cumulative reservation. Do not set the prior reservation to zero to bypass previous spending. Reservations are conservative allowances, not reconciled invoices or provider billing caps.

Install the live dependencies and authenticate Modal:

```sh
uv sync --project examples/notebooks --locked --extra live
uv run --project examples/notebooks --extra live modal token new
```

Configure the task-owned Spindle secrets and the local Responses CLI proxy as described in [Run a coding-agent trial on Modal](../../docs/trials.md). Keep the private Spindle API JSON and editor JSON outside the repository. The notebooks pass only their filenames to the existing owner and never read or display their contents.

Download the fixed target and stage it through WSM's existing pin-preserving helper. This command downloads model files but does not allocate cloud compute:

```sh
uv run --project examples/notebooks --extra live python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from wsm.modal_trial import stage_smollm

revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
snapshot = Path(snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", revision=revision))
source = Path("runs/notebook-source")
stage_smollm(snapshot.parent.parent, source)
print("Pinned target staged at", source.absolute())
PY
```

Use a fresh staging directory. If it already exists, inspect it rather than overwriting it. The staged source must contain `snapshots/<revision>/model.safetensors`. Held-out cases remain outside that image input and are supplied by the existing owner.

Set the notebook inputs in the same shell before starting Jupyter or marimo:

```sh
export WSM_NOTEBOOK_MODEL_SOURCE="$PWD/runs/notebook-source"
export WSM_NOTEBOOK_API_KEY_FILE="/private/spindle-api.json"
export WSM_NOTEBOOK_EDITOR_CONFIG_FILE="/private/editor.json"
export WSM_NOTEBOOK_OUTPUT="$PWD/runs/notebook-trial"
export WSM_NOTEBOOK_PRIOR_RESERVED_USD="9.927083333333334"
export WSM_NOTEBOOK_LIVE=1
```

The prior value above deliberately blocks another trial against the original allocation. Replace it only with the current cumulative amount for an allocation you are authorized to use. The cloud cell checks that the full default resource windows fit under US$10 before launching the owner. Check current resource rates in the underlying [trial guide](../../docs/trials.md#set-conservative-cost-reservations) before live use.

Restart the notebook process after changing these variables. Retain the live extra when starting either notebook to keep the dependency-complete environment:

```sh
uv run --project examples/notebooks --locked --extra live jupyter lab examples/notebooks/wsm_trial.ipynb
uv run --project examples/notebooks --locked --extra live marimo edit examples/notebooks/wsm_trial_marimo.py
```

Run the Jupyter live cell, or click the marimo button. A fresh output directory and owner-log filename are required. Re-executing the same run fails rather than repeating a cloud update.

The live cell uses `uv run --project examples/notebooks --extra live` to launch a separate owner process. The grader uses `examples/notebooks/.venv/bin/python`, which has Torch installed through the live extra. Its virtualenv symlink is preserved. On interruption, the cell signals the owner to finish cleanup. The independent watchdog also stops task-owned cloud apps if the notebook or owner dies. Inspect the private owner log and `proof.json` before any retry.

## Read the results

Read the public action history from another terminal:

```sh
uv run --project examples/notebooks --extra live wsm trial view runs/notebook-trial/run/trial
```

Add `--follow` during a trial. Use `wsm trial weights` on a recorded safetensors artifact to inspect tensor names or numerical values. The notebook displays only the overall state, cumulative reservation, and cleanup count. Full commands, source, evaluation feedback, checkpoint changes, the hidden grade, and student training receipt stay in the run directory. Keep raw logs and teacher traces private and clear notebook outputs before sharing.

## Verify without cloud access

Execute every Jupyter cell with its default offline mode:

```sh
uv run --project examples/notebooks python - <<'PY'
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

os.environ.pop("WSM_NOTEBOOK_LIVE", None)
path = Path("examples/notebooks/wsm_trial.ipynb")
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=60, kernel_name="python3").execute(cwd=str(Path.cwd()))
assert all(output.output_type != "error" for cell in notebook.cells
           if cell.cell_type == "code" for output in cell.outputs)
print("Jupyter offline execution passed")
PY
```

Check and execute the actual marimo app while exporting its offline outputs:

```sh
uv run --project examples/notebooks marimo check examples/notebooks/wsm_trial_marimo.py
uv run --project examples/notebooks marimo export html examples/notebooks/wsm_trial_marimo.py \
  --output /tmp/wsm-trial-marimo.html --no-include-code
```

The static export reads the retained proof and leaves the cloud button unpressed, even if live mode is enabled in the environment. These local notebook checks do not repeat the previously recorded cloud trial or spend its remaining allocation.

The commands follow [uv's Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/), [uv's script metadata guide](https://docs.astral.sh/uv/guides/scripts/), and [marimo's run-button contract](https://docs.marimo.io/api/inputs/run_button/).
