# Run a prompt-driven weight modification

Point a model at the weights. WSM keeps a change when the loss goes down.

```sh
python3 -m pip install -e packages/python
export WSM_TEACHER_URL="https://example.invalid"
export WSM_TEACHER_MODEL="your-model-id"
export WSM_TEACHER_API_KEY="your-key"
wsm run "Improve signed addition. Inspect the weights and test your changes." \
  --task addition --output runs/addition
```

The model sends one command at a time.

```json
{"action":"inspect","tensor":"output.weight","indices":[0,1]}
{"action":"edit","edits":[{"tensor":"output.weight","index":0,"value":1}]}
```

WSM keeps the edit when the loss goes down.

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

For reproducible configuration-driven runs or scripted proposals, use the [experiment driver](driver.md). Both teacher entry points call the same loop and `Session`.

The Python distribution, the import package, and the command are named `wsm`. There is no compatibility import for the old package name. Existing `weightless.*` checkpoint and evidence format names stay unchanged.
