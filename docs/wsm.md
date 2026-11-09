# Run a prompt-driven weight modification

Weight Space Modifiers (WSM) is a harness for editing a model's weights. An AI teacher chooses scalar replacements. The harness inspects and evaluates through the existing sandboxed client, retains strict development-loss improvements, and saves the result.

The Python distribution, import package, and CLI are named `wsm`. No compatibility shim is provided for the old package name. Existing `weightless.*` checkpoint and evidence format names remain unchanged.

## Configure a teacher and runner

Install Python 3.12 or later and the SDK.

```sh
python3 -m pip install -e packages/python
wsm --help
```

Set `WSM_TEACHER_URL` to your Responses-compatible base URL, `WSM_TEACHER_MODEL` to its exact model ID, and `WSM_TEACHER_API_KEY` to its credential. Configure these in your trusted shell or secret manager. The URL and model can also be passed with `--teacher-url` and `--teacher-model`. There is no route or model fallback.

The broker sends empty provider tool declarations and expects literal JSON commands in text responses. The configured endpoint must support streaming Responses calls and echo the requested model identity. Credentials never go in CLI arguments, the target sandbox, or exported evidence.

The default `BubblewrapRunner` requires a root-authorized Linux host with a running systemd manager and bubblewrap. See the [sandbox setup](sandbox.md). To use an existing Docker worker image, add `--runner docker --image <image>`. WSM does not install dependencies, download target models, or fall back to host execution.

## Start with signed addition

```sh
wsm run "Improve signed addition. Inspect the weights and test your changes." \
  --task addition --attempts 20 --turns 64 --seconds 900 \
  --output runs/addition
```

The bundled `addition` profile contains a fixed affine/ReLU network, editable parameter names, and public input/output examples. The teacher receives the network graph and can inspect coefficient values. Its edits are live teacher choices rather than the scripted proposals in `examples/demo.py`.

The prompt describes the goal. Flags enforce the attempt, teacher-request, and time limits. Writing a limit in natural language does not change those limits. The original deadline and counters survive resume.

Each teacher response contains one command:

```json
{"action":"inspect","tensor":"output.weight","indices":[0,1]}
```

```json
{"action":"edit","edits":[{"tensor":"output.weight","index":0,"value":1}]}
```

The teacher can also request `{"action":"diagnose"}` or `{"action":"finish"}`. The harness supplies operation keys and binds commands to the observation the teacher actually saw. An edit also runs candidate evaluation. A later turn receives public feedback, including acceptance and literal predictions. A stale observation is rejected rather than rebound to newer weights.

The command prints a structured report. Inspect these files in the private run directory:

- `checkpoint.json` contains the selected binary64 checkpoint, ready for `wsm.binary64.predict`.
- `mod.json` contains the original target, selected checkpoint identity, and effective scalar patches.
- For a model target, `selected/` is the standalone snapshot. A later process can load it without the original model directory. `mod.json` remains the patch record.
- `harness/report.json` contains the stop reason, selected development loss, counters, unresolved operations, and receipt-derived final conclusion.
- `harness/bundle.json` contains replayable public session evidence.

`run.json`, `session.db`, `teacher-private/`, and `harness/cursor.json` are private operational state. The cursor retains the pending prompt and response for recovery. Do not publish the run directory as an evidence bundle. The default export contains no separate agent transcript.

The bundled profile has no private final dataset. Its final conclusion is therefore `inconclusive`, even if all public cases pass. Lower loss alone does not establish the requested behavior or retention of unrelated capabilities.

## Resume or inspect a run

```sh
wsm resume runs/addition
wsm status runs/addition
```

`resume` uses the saved prompt, task, teacher route, runner, limits, and pending operation. Supply the same endpoint credential through `WSM_TEACHER_API_KEY`. Changes to the target, runner, or private final dataset fail identity checks. A completed run restores its saved artifacts without another teacher request or evaluation. It does not reopen development or extend the deadline.

`status` reads the last saved report without credentials or sandbox startup. It is available after the harness has stopped. Exit code 0 means the run finished normally, not that the desired behavior was established. Other stops return 1 with the report. Configuration and admission failures return 2 with an error on stderr.

An uncertain teacher send or target operation stops the loop and preserves its reservation. WSM never silently resends it under a new key. During the teacher loop, budget exhaustion, deadline exhaustion, and Ctrl-C freeze and export partial evidence. A freeze without an evaluated baseline or with unresolved operations cannot receive final grading.

## Use your own task

Copy the [bundled task profile](../packages/python/src/wsm/tasks/addition.json) and pass its path to `--task`. A profile has `format: "wsm.task.v1"`, `target`, `editable`, and nonempty `development` cases. Optional public fields are `context`, `max_edits`, `max_inspect`, `development_retention`, and `development_retention_loss_max`.

A binary64 target contains `kind: "binary64"` and `checkpoint`. The checkpoint can be an existing v1 checkpoint or an object with `graph` and `tensors`. A SmolLM target is the exact object returned by `stage_smollm`. Relative `source` paths resolve beside the profile. The existing pinned model and float32 execution restrictions still apply.

For private final grading, add a `final` object to the profile:

```json
{
  "cases": "private/confirmation.json",
  "registry": "private/verifier.db",
  "dataset": "confirmation"
}
```

Those paths resolve beside the task profile. Create the private directory first. The cases file contains a list of task cases, or an object with `task` and optional `retention` lists. Keep the cases and registry outside every mounted target directory. Reuse the same registry across runs for that dataset so final exposure claims remain durable.

For a confirmatory finite-case experiment, also declare `criteria` with `final_exact_min` and `retention_loss_max`. Without both thresholds, use the exploratory profile. The public retention gate requires public retention cases and an explicit absolute loss ceiling. A prompt asking to preserve other abilities does not supply retention tests or thresholds.

On `finish`, the host freezes selection and invokes the configured private verifier only if the freeze is complete. The teacher cannot select private cases, change thresholds, or use final feedback for another edit. Reports take conclusions from the verifier receipt, never from a teacher's success claim. See the [client contract](client.md) for evaluation and exposure limits.

## Attach the harness to an existing session

Trusted Python callers can use their existing `Session`, runner, verifier, and teacher objects:

```python
from wsm.harness import run

report = run(
    session,
    teacher,
    prompt="Improve the declared task within the existing limits.",
    output="runs/custom-harness",
    arm="teacher",
    context={"description": "Public information about the target and task."},
)
```

The output directory must be new or belong to this same session, arm, prompt, and public context. Only pass public information as `context`. Session policy, operation admission, and numerical execution remain in their existing owners. See [create and resume sessions](client.md#session-contract) for their setup.

## Bound model computation

`--forward-calls` limits total model forward calls, including final grading. It defaults to 10000. `--final-forward-calls` declares a private grading envelope charged against that total, defaulting to 1000. Leave enough total budget after development for that envelope. Exact private usage remains in the verifier registry.

For reproducible configuration-driven runs or scripted proposals, use the [experiment driver](driver.md). Both teacher entry points call the same harness and Session.
