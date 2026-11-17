# Verify a WSM change

Run the relevant section from the repository root. The Python and notebook
checks below use Linux, Python 3.12, and uv. The tests use local fixtures and
retained notebook results.

## Python

Create a virtual environment with pip for the wheel smoke test. Use a filesystem
that supports [checkpoint publication](docs/coding.md).

```sh
uv venv --python 3.12 --seed .venv
. .venv/bin/activate
uv pip install -e 'packages/python[editing,trial]' --torch-backend cpu
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false bash scripts/verify.sh
uv build packages/python --wheel
```

The script runs the Python test suite and builds and installs a wheel in a
temporary directory to check its public API and CLI.
[Python CI](.github/workflows/python.yml) runs unittest discovery and builds
the wheel on Ubuntu for changes under `packages/python/` or `scripts/`, or
to the Python workflow. The installed-wheel smoke test is a local check.

## Documentation site

Use Node.js 24 to match CI.

```sh
(
  set -e
  cd site
  npm ci
  npm run check
  npx playwright install chromium
  npm test
)
```

`npm test` checks command spelling, builds the site, and runs Playwright against
a local preview. [Site CI](.github/workflows/site.yml) also installs Chromium's
system dependencies, packages the static output for Cloudflare, and runs the
deployment dry run. It runs for changes under `site/` or `docs/`, or to the
site workflow.

## Notebooks

Run Jupyter in a real kernel and check and export marimo with live mode unset.

```sh
(
  set -e
  unset WSM_NOTEBOOK_LIVE
  uv sync --project examples/notebooks --locked
  uv run --project examples/notebooks --locked python - <<'PY'
from pathlib import Path

import nbformat
from nbclient import NotebookClient

notebook = nbformat.read(Path("examples/notebooks/wsm_trial.ipynb"), as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=60, kernel_name="python3").execute(cwd=str(Path.cwd()))
PY
  uv run --project examples/notebooks --locked marimo check examples/notebooks/wsm_trial_marimo.py
  notebook_output=$(mktemp -d)
  uv run --project examples/notebooks --locked marimo export html examples/notebooks/wsm_trial_marimo.py \
    --output "$notebook_output/wsm-trial-marimo.html" --no-include-code
)
```

The Python and Site workflows do not run these notebook checks. See the
[notebook guide](examples/notebooks/README.md) for interpretation and launch
instructions.
