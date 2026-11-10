# WSM (Weight Space Modifiers)

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

## Names

The Python distribution, the import package, and the command are `wsm`. With no arguments, `wsm` opens the terminal UI when stdin and stdout are a terminal. Session verbs such as `init`, `observe`, `submit`, `status`, and `export` match the session methods. See [the command reference](docs/cli.md).

This replaces the former `weightless-sdk` distribution and `weightless` namespace. There is no import alias and no second console script. For an existing environment, uninstall `weightless-sdk` before installing from this checkout, then update your imports to `wsm`. Serialized `weightless.*` format identifiers and hashes, `WeightlessError`, environment variables, and Docker image names keep their existing spelling.

The core SDK has no third-party runtime dependencies.

## Run without a model

`examples/demo.py` sends saved proposals through the same selection rules. It does not call a model. Docker and credentials are not required. The network is a fixed affine/ReLU model for signed addition.

```sh
python3 examples/demo.py --output /tmp/wsm-run.json
```

The script prints one line per proposal.

```json
{"accepted": false, "reason": "not_improved"}
{"accepted": true, "reason": "improved"}
{"accepted": false, "reason": "not_improved"}
```

The frozen run is saved to `/tmp/wsm-run.json`. The retained checkpoint computes signed addition on the synthetic example. Pass `--proposals /path/to/proposals.json` to supply your own edit batches in the format used by `examples/teacher-proposals.json`.

## Submit edits through the SDK

Run this example from the repository root using the installed environment:

```python
import json
from wsm import MetricPolicy, NumericalEvaluation, SearchRun, replay
from wsm.binary64 import create_checkpoint, predict

with open("fixtures/addition.json") as file:
    fixture = json.load(file)
initial = create_checkpoint(fixture["graph"], fixture["tensors"])
policy = MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize"))
run = SearchRun(initial, evaluator=NumericalEvaluation(fixture["task"]),
                policy=policy, budget=8)
observation = run.inspect()  # Send this copied observation to your teacher.
proposal = {                # Receive explicit replacements from the teacher.
    "base": observation["checkpoint"]["id"],
    "author": "external-teacher",
    "edits": [
        {"tensor": "output.weight", "index": 0, "value": 1},
        {"tensor": "output.weight", "index": 1, "value": -1},
    ],
}
decision = run.submit(proposal)
frozen = run.finish()
assert decision["accepted"]
assert predict(replay(frozen), [-8, 3]) == [-5]
```

For explicit execution control on an open run, replace `submit(proposal)` with `candidate = run.propose(proposal)`, `result = run.evaluate(candidate)`, and `decision = run.select(candidate, result)`. These are the same stages used by `submit`.

Construction performs no evaluation. The first evaluation retains a separately bound baseline before its candidate. Each valid proposal reserves one attempt and permits one evaluation. Invalid or stale proposals consume nothing. There is at most one pending candidate, and there are no implicit retries. `finish` closes the run without evaluating pending work and returns copies of the same frozen contents.

Selection compares the configured metric paths in order. This example minimizes development loss and then training loss. Exact ties retain the incumbent. A genuine graded-zero reward remains valid feedback. Task failures, invalid responses, and infrastructure failures remain distinct evidence and cannot win. A failed baseline stays inspectable and prevents selection.

## Choose optional execution and grading

For a configuration-driven experiment, run the supplied binary64 example and export its selected checkpoint:

```sh
python3 scripts/drive_session.py examples/drive-session.json
python3 scripts/export_checkpoint.py \
  --bundle runs/drive-session-example/evidence.json \
  --target runs/drive-session-example/target.json \
  --arm solver --destination runs/drive-session-example/checkpoint --max-bytes 1000000
```

This example uses scripted proposals and an explicitly unisolated binary64 runner. The driver also accepts an explicit editing-AI endpoint and model with a Docker or bubblewrap target runner. It supplies the JSON command protocol and returns permitted inspection, diagnostic, and evaluation results to the editing AI. New configurations require forward-call budgets. Incomplete freezing or failed required final grading produces a nonzero exit while retaining evidence. See the [configured experiment guide](docs/client.md#run-a-configured-experiment).

The [durable local client](docs/client.md) adds resumable `Session` experiments, a separate private `Verifier`, and explicit teacher transport through `ResponsesTeacher`. Its execution targets are binary64 checkpoints and a pinned SmolLM2 checkpoint. The [local sandbox guide](docs/sandbox.md) covers Docker-free Linux execution through `BubblewrapRunner`, bubblewrap, and systemd. `DockerRunner` is also available. These paths require host setup beyond the scripted demo.

For native Harbor development feedback, use the optional Verifiers adapter on a Docker host.

Install the optional integration packages, prepare trusted purpose-specific tasks, and build the agent image on a Docker host.

```sh
python3 -m pip install -e 'packages/python[integration]'
python3 scripts/prepare_harbor.py --development /tmp/weightless-development
docker build -t weightless-signed-addition:local examples/harbor/signed-addition/environment
```

Replace only the evaluator in the same teacher loop:

```python
from wsm.harbor import HarborEvaluation, resolve_image

image = resolve_image("weightless-signed-addition:local")
evaluator = HarborEvaluation(task_dir="/tmp/weightless-development", image=image)
run = SearchRun(initial, evaluator=evaluator, policy=policy, budget=8)
# Call inspect and submit with teacher proposals, then freeze.
frozen = run.finish()
final = HarborEvaluation(task_dir="examples/harbor/signed-addition", image=image)
final_result = final.grade(frozen)
```

`resolve_image` explicitly reads Docker metadata. The evaluator records the immutable image ID, trusted package digest, versions, resources, timeout, and purpose. It executes the installed Verifiers `HarborTask` through isolated `grade_task`, with zero retries. Its callback captures native rewards, numerical metrics, `diagnostics.json`, verifier exit code, stdout, and stderr before teardown.

Development grading receives one copied candidate checkpoint and its bound request. Its trusted package contains public training and development examples. Independent final grading receives a frozen run, verifies its expected initial problem and policy, replays patches, and scores heldout examples. The trusted package chooses the split. Submission fields cannot choose one. The separate verifier receives only declared `/app/submission.json` and trusted tests. The agent image excludes tests, heldout data, and oracle code.

The adapter is synchronous and serial. Call it outside an active asynchronous event loop. It runs grading without a teacher model, rollout, hosted service, or GPU.

## Verify the boundaries

Run the core checks and installed-wheel smoke without optional integrations:

```sh
python3 scripts/golden_identity.py
python3 -m unittest discover -s packages/python/tests -v
python3 scripts/package_smoke.py
```

These commands check independent golden bytes, unit behavior, and installation and replay from a built wheel outside the source tree. Native bubblewrap tests skip unless explicitly enabled on a configured Linux host. A skipped test does not prove isolation.

After installing the integration extras above, run `INTEGRATION_PYTHON="$PWD/.venv/bin/python" bash scripts/verify.sh` for additional schema, upstream task-loading, and trusted verifier checks. The [verification guide](docs/verification.md) gives separate commands and prerequisites for native Harbor and sandbox execution. GitHub Actions remains disabled.

The [JSON Schema](schemas/v1.json) validates structural wire records. The SDK additionally checks identities, graph relationships, patch addresses, result bindings, budget accounting, and policy decisions. Numerical `create_checkpoint`, `parse_checkpoint`, `predict`, `evaluate`, and patch operations belong to `wsm.binary64`.

`replay(frozen)` reconstructs weights and checks recorded evidence and decisions without invoking any evaluator. It can replay Harbor-produced runs with no optional dependencies installed. Unsigned hashes prove internal consistency, not external score authenticity or live teacher behavior. Independent heldout grading is the correctness check. Checkpoint v1 bytes remain unchanged. Frozen runs use the new v2 format, with no old API shims.

## Scope and further reading

The Astro documentation site lives in [`site/`](site/README.md). Run `npm ci` and `npm run dev` from that folder to browse the guides locally. The site reads the Markdown files in `docs/` directly.

The target workflow follows [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). An editing AI inspects permitted target weights, proposes explicit replacements without optimizer-driven training, and receives bounded evaluation feedback. The operator chooses the target, editing-AI provider, resource budget, and acceptable regressions. The [Tinker capability boundary](docs/tinker.md) distinguishes hosted Tinker integration from a compatible service interface; neither is established by the local backend.

The scripted demo demonstrates explicit weight edits, strict selection, and replay on a synthetic addition task. It does not reproduce a paper result or establish optimizer superiority, general extrapolation, or useful language-model weight discovery. The binary64 backend uses scalar arithmetic, which need not match every accelerated numerical runtime. The [research scope](docs/research.md) records the current research direction, historical example provenance, and untested claims.

Use the [offline interactive tutorial](docs/tutorial.html) for a walkthrough, the [architecture](docs/architecture.md) for ownership boundaries, and the [shared contract](docs/contract.md) for wire records and validation. The [client design and validation specification](docs/design/agent-client.md) describes the broader acceptance program. Its requirements are separate from demonstrated results in the verification guide.
