# Run an experiment from a configuration file

Use `wsm run` for the installed prompt-driven workflow. Use `scripts/drive_session.py` from a source checkout when the experiment needs an explicit declaration, a named arm, scripted proposals, or runner resource settings in one JSON file. Both paths use `Session` for admission, budgets, selection, freeze, and private verification. Teacher mode uses the same `wsm.harness.run` loop as `wsm run`.

## Run the offline example

Install the package from this checkout, then run:

```sh
python3 scripts/drive_session.py examples/drive-session.json
python3 scripts/export_checkpoint.py \
  --bundle runs/drive-session-example/evidence.json \
  --target runs/drive-session-example/target.json \
  --arm solver --destination runs/drive-session-export --max-bytes 1000000
```

The example applies supplied replacements to a real binary64 worker and selects the affine function `2x+3`. It writes `evidence.json`, `target.json`, and `driver.json`. The export command prints a receipt and writes `checkpoint.json` into a new destination. It calls the same exporter as `wsm export-model`.

The example is `scripted_plumbing`, not an AI discovery result. Its in-process runner provides no isolation. Model experiments require a configured Docker or bubblewrap runner. The supplied output directory must not already exist. Paths in the configuration are relative to the current working directory.

## Configure a teacher

Replace `proposals` with a `teacher` object. Do not specify both. For example:

```json
{
  "instructions": "Improve the declared behavior using permitted numerical edits.",
  "endpoint": "http://localhost:8317/v1",
  "model": "YOUR_CONFIGURED_MODEL",
  "api_key_env": "WSM_TEACHER_API_KEY",
  "directory": "/private/teacher-artifacts",
  "timeout": 60,
  "max_output_tokens": 2048
}
```

Set a positive `declaration.budgets.teacher_requests`. The harness supplies the edit and inspection protocol and binds proposals to current checkpoints. It preserves uncertain sends instead of silently repeating them. The driver records teacher mode as `fresh_agent`; that label describes the requested execution mode and does not establish provider authenticity or useful improvement. Tests with a local scripted HTTP server remain transport tests.

This source driver starts a new run only. Keep its session database and harness artifacts when execution fails. Do not start another run to conceal an unknown outcome or reset a spent budget. For a directly resumable operator workflow, use the installed `wsm run` and `wsm resume` commands.

## Declare limits and private grading

The configuration requires `declaration.budgets.forward_calls` and `max_new_tokens: 16`, matching the current session generation reservation. A configured verifier also requires `declaration.final_forward_calls`. Session enforces those limits; the driver does not implement a separate selection or budget policy.

A `verifier` object contains `database`, `dataset`, and a `cases` JSON path. Keep verifier files and teacher artifacts outside the exported run and model mount. Credentials can come from the named environment variable or the source driver's existing explicit credentials-file format. They are never copied into the run configuration or teacher prompt.

Supported targets are binary64, the pinned SmolLM profile, and locally staged Hugging Face snapshots accepted by the existing backend. Staging requirements, tensor-layout restrictions, and offline loading still apply. The driver adds no new hosted backend or GPU provisioning.

## Verification scope

`test_drive_session.py` exercises real binary64 edits, replay, numerical reconstruction, the export subprocess, forward-budget exhaustion, rejected configurations, and private-data canaries through a scripted HTTP teacher. It does not make a live provider call or establish model efficacy.
