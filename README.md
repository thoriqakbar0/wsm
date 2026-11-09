# Weight Space Modifiers

WSM is a harness for editing a model's weights. Give an AI teacher a prompt and a task profile. The harness lets it inspect coefficients, propose edits, and test candidates, then saves the selected checkpoint and evaluation evidence. The Python distribution, import package, and CLI are named `wsm`.

```sh
python3 -m pip install -e packages/python
# Configure WSM_TEACHER_URL, WSM_TEACHER_MODEL, and WSM_TEACHER_API_KEY first.
wsm run "Improve signed addition. Inspect the weights and test your changes." \
  --task addition --attempts 20 --output runs/addition
wsm status runs/addition
```

The default runner requires a root-authorized Linux systemd host with bubblewrap. Docker is available with `--runner docker --image <existing-image>`. There is no unsandboxed fallback. The bundled addition profile has public examples only. Read [Run a prompt-driven weight modification](docs/wsm.md) for setup, custom targets, private final evaluation, recovery, and the Python entry point.

## Numerical SDK

Weightless lets an external teacher propose explicit numerical edits to a fixed neural network. A Python run owns copied candidates and bound evaluation evidence, then selects strict improvements by a declared metric policy. Numerical access lives in a lower-level backend. The evaluator can use local arithmetic or the actual Verifiers Harbor adapter in Docker.

The signed-addition example uses a fixed affine/ReLU network and an offline scripted teacher. It makes no model call and claims no paper result. Read the [offline interactive tutorial](docs/tutorial.html), [architecture](docs/architecture.md), [shared contract](docs/contract.md), and [research scope](docs/research.md).

The [durable local client](docs/client.md) adds resumable sessions, a separate private verifier, and enforced execution for binary64 and pinned SmolLM2 targets. The [local sandbox](docs/sandbox.md) runs on Linux with bubblewrap and systemd, without Docker or KVM. The [client design and validation specification](docs/design/agent-client.md) describes the broader acceptance program.

## Run the example

Python 3.12 or later is required. The core SDK has no runtime dependencies.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e packages/python
.venv/bin/python examples/demo.py --output /tmp/weightless-run.json
```

The decisions are `not_improved`, `improved`, and `not_improved`. The retained checkpoint computes signed addition. Replace `examples/teacher-proposals.json` and pass `--proposals /path/to/proposals.json` to supply other explicit edit batches.

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

## Use native Harbor development feedback

Install the optional integration packages, prepare trusted purpose-specific tasks, and build the agent image on a Docker host.

```sh
.venv/bin/python -m pip install -e 'packages/python[integration]'
.venv/bin/python scripts/prepare_harbor.py --development /tmp/weightless-development
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

```sh
INTEGRATION_PYTHON="$PWD/.venv/bin/python" bash scripts/verify.sh
python3 scripts/package_smoke.py
.venv/bin/python scripts/harbor_native.py --failures --output /tmp/weightless-native-proof
.venv/bin/harbor run --path examples/harbor/signed-addition --agent oracle --env docker \
  --force-build --jobs-dir /tmp/weightless-jobs --job-name oracle
python3 scripts/check_harbor_reward.py /tmp/weightless-jobs/oracle
```

The first command checks independent golden bytes, unit behavior, schema, upstream task loading, and local trusted verifier scripts. The package smoke builds and installs the wheel outside the source tree. The native adapter command requires Docker and proves development decisions plus independent final grading through Verifiers. The Harbor CLI command separately proves the native task. [Verification details](docs/verification.md) distinguish each result. GitHub Actions remains disabled.

The [JSON Schema](schemas/v1.json) validates structural wire records. The SDK additionally checks identities, graph relationships, patch addresses, result bindings, budget accounting, and policy decisions. Numerical `create_checkpoint`, `parse_checkpoint`, `predict`, `evaluate`, and patch operations belong to `wsm.binary64`.

`replay(frozen)` reconstructs weights and checks recorded evidence and decisions without invoking any evaluator. It can replay Harbor-produced runs with no optional dependencies installed. Unsigned hashes prove internal consistency, not external score authenticity or live teacher behavior. Independent heldout grading is the correctness check. Checkpoint v1 bytes remain unchanged. Frozen runs use the new v2 format, with no old API shims.
