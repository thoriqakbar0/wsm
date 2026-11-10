# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marimo==0.25.2", "wsm"]
# [tool.uv.sources]
# wsm = { path = "../../packages/python", editable = true }
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import math
    import os
    from pathlib import Path
    import subprocess
    import sys

    return Path, json, math, mo, os, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # Run a WSM teacher and student trial

    Codex edits a fresh SmolLM2 target on Modal. An independent hidden grade weights
    a Qwen3-4B student update through Spindle. This notebook starts offline.
    Read `examples/notebooks/README.md` for uv setup and private credential files.
    """)
    return


@app.cell
def _(Path, json):
    root = next(path for path in [Path.cwd(), *Path.cwd().parents]
                if (path / "scripts/spindle_poc.py").is_file())
    notebooks = root / "examples/notebooks"
    receipt = json.loads((root / "docs/design/trial-verification.json").read_text())
    return notebooks, receipt, root


@app.cell
def _(mo, receipt):
    _teacher = receipt["checks"]["live_codex_teacher"]
    _student = receipt["checks"]["live_spindle_student"]
    mo.md(f"""## Inspect the previous proof

    Teacher **{_teacher['teacher']}** completed {_teacher['turns']} turns.
    Student **{_student['student']}** saved a checkpoint after a confirmed update.
    Cumulative reserved cost was **US${_student['cumulative_reserved_usd']:.6f}**.
    Cleanup confirmed **{_student['cleanup']['remaining_tasks']} remaining tasks**.
    That original US$10 allocation cannot fund another full trial.
    """)
    return


@app.cell
def _(root, subprocess, sys):
    _cli = subprocess.run([sys.executable, str(root / "scripts/spindle_poc.py"), "--help"],
                          cwd=root, capture_output=True, text=True, check=True)
    assert "--prior-reserved-usd" in _cli.stdout
    return


@app.cell
def _(os, root):
    config = {
        "model_source": os.environ.get("WSM_NOTEBOOK_MODEL_SOURCE", ""),
        "api_key_file": os.environ.get("WSM_NOTEBOOK_API_KEY_FILE", ""),
        "editor_config_file": os.environ.get("WSM_NOTEBOOK_EDITOR_CONFIG_FILE", ""),
        "output": os.environ.get("WSM_NOTEBOOK_OUTPUT", str(root / "runs/notebook-trial")),
        "prior_reserved_usd": os.environ.get("WSM_NOTEBOOK_PRIOR_RESERVED_USD"),
        "live": os.environ.get("WSM_NOTEBOOK_LIVE") == "1",
    }
    return (config,)


@app.cell
def _(config, mo):
    live_button = mo.ui.run_button(label="Run one authorized cloud trial", disabled=not config["live"])
    mo.vstack([
        mo.md("""## Run a new trial

        Set the documented `WSM_NOTEBOOK_*` environment variables and restart the app.
        `WSM_NOTEBOOK_LIVE=1` enables this button. You must also click the button.
        Editing a cell, running as a script, or exporting HTML never clicks it.
        The cumulative prior reservation is required. Existing outputs are never reused.
        Cloud logs remain on disk. Interrupting the cell signals the bounded owner,
        whose independent watchdog completes cleanup.
        """),
        live_button,
    ])
    return (live_button,)


@app.cell
def _(Path, config, json, live_button, math, mo, notebooks, root, subprocess):
    mo.stop(not live_button.value, mo.md("Offline inspection complete. No cloud run requested."))
    if not config["live"]:
        raise ValueError("Live mode must be explicitly enabled")
    prior = config["prior_reserved_usd"]
    if prior is None:
        raise ValueError("Set WSM_NOTEBOOK_PRIOR_RESERVED_USD to the current cumulative reservation")
    prior = float(prior)
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("Prior reservation must be finite and nonnegative")
    # Full default resource windows, two image builds, and artifact egress allowance.
    required = 1.0 + 5 * 600 / 3600 + 8 * 1200 / 3600 + 0.1
    if prior + required > 10:
        raise ValueError("Insufficient remaining US$10 reservation for a new complete trial")
    for field in ("model_source", "api_key_file", "editor_config_file"):
        if not config[field] or not Path(config[field]).expanduser().exists():
            raise ValueError("Configure an existing " + field)
    output = Path(config["output"]).expanduser().absolute()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; existing results are never rerun")
    command = [
        "uv", "run", "--project", str(notebooks), "--locked", "--extra", "live", "python",
        str(root / "scripts/spindle_poc.py"),
        "--model-source", str(Path(config["model_source"]).expanduser().absolute()),
        "--grader-python", str(notebooks / ".venv/bin/python"),
        "--api-key-file", str(Path(config["api_key_file"]).expanduser().absolute()),
        "--editor-config-file", str(Path(config["editor_config_file"]).expanduser().absolute()),
        "--prior-reserved-usd", str(prior), "--output", str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / (output.name + ".owner.log")
    with log_path.open("x") as log:
        owner = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        returncode = owner.wait()
    except KeyboardInterrupt:
        import signal
        owner.send_signal(signal.SIGTERM)
        raise RuntimeError("Stop requested. The owner and watchdog finish cloud cleanup. Inspect the local log.") from None
    if returncode:
        raise RuntimeError("The owner failed. Inspect the private owner log and proof.json before any retry.")
    proof = json.loads((output / "proof.json").read_text())
    print({"state": proof["state"], "reserved_usd": proof["reserved_usd"],
           "cleanup_confirmed": proof["cleanup"]["confirmed"],
           "remaining_tasks": proof["cleanup"]["remaining_tasks"]})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inspect retained actions and results

    Use `uv run --project examples/notebooks --extra live wsm trial view
    runs/notebook-trial/run/trial` in another terminal. Add `--follow` during the trial.
    The output's `proof.json` records cleanup. `run/training/receipt.json` records the
    student update and saved checkpoint. Keep raw logs and teacher traces private.
    Resource reservations are not reconciled invoices or provider billing caps.
    """)
    return


if __name__ == "__main__":
    app.run()
