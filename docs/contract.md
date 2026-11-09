# Weightless experiment contract

The data model is an owned run with a copied baseline, at most one pending candidate, bound evaluation evidence, and an ordered metric policy. The core never interprets numerical example arrays. Numerical access belongs to `weightless.binary64`. Optional Docker execution belongs to `weightless.harbor`.

## Intended caller

```python
import json
from weightless import MetricPolicy, NumericalEvaluation, SearchRun, replay
from weightless.binary64 import create_checkpoint, predict

fixture = json.load(open("fixtures/addition.json"))
initial = create_checkpoint(fixture["graph"], fixture["tensors"])
policy = MetricPolicy(("development.loss", "minimize"), ("train.loss", "minimize"))
run = SearchRun(initial, evaluator=NumericalEvaluation(fixture["task"]),
                policy=policy, budget=8)

# Send run.inspect() to an external teacher. Receive explicit numerical edits.
observation = run.inspect()
proposal = {
    "base": observation["checkpoint"]["id"],
    "author": "scripted-teacher",
    "edits": [
        {"tensor": "output.weight", "index": 0, "value": 1},
        {"tensor": "output.weight", "index": 1, "value": -1},
    ],
}
decision = run.submit(proposal)
assert decision["accepted"]
frozen = run.finish()
assert predict(replay(frozen), [-8, 3]) == [-5]
```

The ordinary workflow is `inspect`, `submit`, `finish`. `submit` calls the same public stages an advanced caller can use individually:

```python
candidate = run.propose(proposal)
result = run.evaluate(candidate)
decision = run.select(candidate, result)
```

Use these stages on an open run. They are an alternative to submitting that proposal, not calls to make after closing the example above. `run.evaluate(run.inspect()["incumbent"])` explicitly evaluates the baseline before any proposal. Otherwise the first candidate evaluation does so automatically. Construction never invokes the evaluator.

Only evaluator configuration changes for native development feedback:

```python
from weightless.harbor import HarborEvaluation, resolve_image

image = resolve_image("weightless-signed-addition:local")
evaluator = HarborEvaluation(task_dir="/tmp/weightless-development",
                             image=image)
run = SearchRun(initial, evaluator=evaluator, policy=policy, budget=8)
# The same teacher proposal loop runs here.
frozen = run.finish()
final = HarborEvaluation(task_dir="examples/harbor/signed-addition",
                         image=image)
final_result = final.grade(frozen)
```

Prepare the development task with `scripts/prepare_harbor.py --development /tmp/weightless-development`. Its trusted manifest chooses development grading. The authored signed-addition task chooses final grading. Submission fields cannot choose or change a split. Final grading is independent and receives a frozen run only.

## Checkpoint v1 and numerical backend

`create_checkpoint(graph, tensors)`, `parse_checkpoint(value)`, `predict(checkpoint, inputs)`, `evaluate(checkpoint, examples)`, `validate_proposal(proposal, checkpoint)`, and `apply_patch(checkpoint, proposal)` live in `weightless.binary64`. The top-level package does not re-export numerical parameter access.

A checkpoint has exactly `format`, `graph`, `tensors`, and `id`. Its format remains `weightless.checkpoint.v1`. A tensor has `shape` and row-major `values`. A graph has `inputSize`, ordered `nodes`, and `output`. Affine nodes have `id`, `op`, `input`, `weight`, and `bias`. ReLU nodes have only `id`, `op`, and `input`. Nodes refer to the reserved `input` or an earlier node. Every declared tensor is used. Shared tensors require compatible shapes.

Identifiers match `[A-Za-z][A-Za-z0-9_.-]{0,63}`. Positive dimensions are at most 1024. There are at most 128 nodes, 100000 stored values, 10000 examples per split, and 100000 edits per proposal. Booleans are never numbers or indices. Integral JSON dimensions normalize to integers. Numeric values normalize to finite binary64, with negative zero replaced by positive zero.

Checkpoint identity bytes remain unchanged:

1. UTF-8 `weightless.checkpoint.v1` followed by a zero byte.
2. Canonical graph JSON followed by a zero byte.
3. For each tensor name in ASCII ascending order, its UTF-8 name and a zero byte, canonical shape JSON and a zero byte, then each value as eight big-endian IEEE-754 binary64 bytes.

The identifier is `sha256:` followed by the lowercase hex SHA-256 digest. Canonical JSON sorts object keys, retains array order, uses ASCII escapes, and has no insignificant whitespace. `fixtures/golden.json` remains the independent byte fixture.

Inference adds products in ascending row-major order, starting at zero, then adds bias. ReLU maps negative values to zero. Nonfinite predictions or metrics are task failures. Scaled residual is `(prediction - target) / max(1, abs(target))`. Loss is half the mean squared scaled residual. RMSE uses raw residuals. Exact counts examples with every output equal to target.

A proposal has exactly `base`, `author`, and nonempty `edits`. Each edit has exactly `tensor`, flattened `index`, and replacement `value`. Unknown tensors, duplicate addresses, out-of-range indices, stale bases, unknown fields, and malformed numbers fail atomically. Proposals cannot change topology or execute code.

`NumericalEvaluation(task)` validates and copies the public task. The task retains `weightless.task.v1`, `name`, `train`, `development`, and its fixture budget hint. `SearchRun` takes its sole authoritative budget explicitly. Example widths are checked by the numerical evaluator against the checkpoint before scoring, never by the experiment core. The numerical evaluator has no final grading method and never receives heldout examples through this public task.

## Owned run and stages

`SearchRun(initial, *, evaluator, policy, budget)` copies its checkpoint, evaluator config, and policy. Its random SHA-256 run identifier distinguishes identical experiments. Budget is 1 through 1000. All public records are JSON dictionaries returned as deep copies. Mutating inputs, observations, candidates, results, decisions, or frozen copies cannot mutate retained state.

`MetricPolicy(*terms)` creates a pure dictionary with `format: weightless.policy.v1`, ordered `metrics` containing `path` and `direction`, and `ties: retain`. A term is `(dotted_path, minimize|maximize)`. One through eight unique paths are allowed. The example compares development loss first and training loss second. Exact ties retain the incumbent. The core knows only those configured paths, not task metric names.

An exported candidate has exactly `run`, `attempt`, `base`, and `checkpoint`. Attempt zero is the baseline with null base. Other attempts are consecutive starting at one. `base` is the incumbent checkpoint identity when proposed. Candidate checkpoint content is a copy. The run accepts only a candidate exactly equal to its retained record, with its own run identity and expected attempt. Equal weights can have equal checkpoint IDs and distinct attempts.

`propose` validates the entire proposal before reserving an attempt and constructing a candidate. Invalid or stale proposals consume nothing and dispatch nothing. A pending candidate blocks another proposal. Each valid candidate has one permitted evaluation and one selection. The run serializes operations and rejects reentrant or concurrent mutations. Each evaluation also receives independent request and checkpoint copies.

`evaluate` retains baseline evidence before dispatching a candidate. A baseline is called at most once, including when it fails. Evaluator exceptions become retained infrastructure failures. Malformed or mismatched results become retained invalid-response failures with bounded error details. No retry, repair, or replacement score occurs. One budget bounds candidate attempts and therefore evaluator dispatch to at most budget plus one baseline.

`select` requires the exact retained candidate and result. A candidate can win only when both baseline and incumbent evidence are completed, its own result is completed, and every policy metric is finite. Failures never win. Decisions have `attempt`, `accepted`, `reason`, `before`, `after`, `evaluation`, `incumbentEvaluation`, and `policy`. Before and after are incumbent attempt numbers. Reasons are `improved`, `not_improved`, `evaluation_failed`, or `baseline_failed`. Policy and evaluation fields bind their content identities.

`inspect` exports `run`, `checkpoint`, `incumbent`, `config`, `policy`, `budget`, `used`, `baseline`, `events`, and `closed`. `baseline` is its retained result or null. Each event has `attempt`, `baseAttempt`, `proposal`, candidate checkpoint identity in `checkpoint`, `evaluation`, and `decision`. The last event may remain unevaluated or unselected. Earlier events must be resolved.

`finish` permanently closes the run and returns the same frozen contents on repeated calls. It does not evaluate pending work. Pending evidence remains recorded and cannot replace the incumbent. Runs with no baseline or failed baseline can freeze. Every mutating operation on a closed run raises `closed`.

## Bound evaluator protocol

An evaluator supplies a copied `config` property and is callable as `evaluator(request, checkpoint)`. Config has exactly `format: weightless.evaluator.v1`, `kind`, `purpose`, `settings`, and `id`. Purpose is `development` or `final`. Config ID hashes all other config fields using the format as a domain prefix. Settings are evaluator-owned, bounded JSON. The core validates JSON and bindings, without interpreting example arrays.

A request has exactly `format: weightless.request.v1`, `run`, `attempt`, `checkpoint`, `config`, and `purpose`. Every result copies that exact request. `evaluation_result(request, *, status, metrics, reward=None, native_status=None, diagnostics=None, error=None)` builds the shared result dictionary. The result has exactly `format: weightless.evaluation.v1`, `request`, `status`, `native`, `error`, and `id`. Native evidence has exactly `reward`, `metrics`, `status`, and `diagnostics`.

Result status is one of:

| Status | Meaning |
| --- | --- |
| `completed` | A valid graded response, including genuine native reward zero. |
| `task_failure` | Task execution or numerical scoring failed. |
| `infrastructure_error` | Provisioning, transport, adapter execution, or teardown failed. |
| `invalid_response` | Missing, malformed, or mismatched evaluator output. |

Completed results have null error. Failure errors have exactly `type` and `message`. Native reward is the upstream reward map unchanged or null for numerical grading. Metrics remain native names and numbers. Diagnostics preserve native verifier output, status, exit code, stdout, stderr, error detail, and truncation indicators. A genuine graded zero has completed status and usable metrics. It is never converted into an infrastructure failure or invented score.

Results bind run, attempt, exact candidate checkpoint, purpose, and configuration. Selection requires the previously retained result, not a newly supplied lookalike. Returned IDs use domain-separated canonical JSON. JSON boundary checks reject duplicate keys, unknown envelope fields, nonfinite numbers, booleans as metrics, excessive depth, excess collections, and payloads over eight million bytes. Admission validates configuration and result depth in their enclosing frozen-run context. Diagnostics and captured streams have explicit smaller limits. Malformed native fields stay in raw diagnostic evidence rather than becoming usable scores; bounded raw fallback records explicitly indicate truncation.

Each evaluation record is at most 262144 bytes. Construction and proposal validation reserve enough history space for baseline and candidate evidence plus a decision before any dispatch. A proposal that would exceed the eight-million-byte frozen-history limit is invalid and spends no attempt. This storage bound is independent of the one caller-selected attempt budget.

Numerical settings retain the copied public task and arithmetic version. Direct numerical evaluator calls reject checkpoint, configuration, or purpose mismatches before scoring. Harbor settings bind trusted task content digest, resolved Docker image content ID, installed Harbor and Verifiers versions, scoring timeout, execution limits, and trusted manifest. Mutable image tags and paths alone do not identify configuration. `resolve_image(tag)` explicitly reads Docker metadata before construction. `HarborEvaluation` requires that immutable image ID and does no external execution in its constructor.

## Native Harbor boundary

`HarborEvaluation` is optional. Core import requires only Python 3.12 and the standard library. Its module uses installed Verifiers `parse_task`, `HarborTask`, `DockerConfig`, `Deployment`, and isolated `grade_task(on_score=...)`. It builds the actual upstream trace containing only declared JSON. There is no agent or model rollout. Retries are zero. The synchronous API runs serially and rejects calls from an active asynchronous event loop.

The `on_score` hook captures `diagnostics.json` before verifier teardown. Native reward files stay authoritative even when the verifier exits nonzero. Upstream scoring metadata and native diagnostics are both retained. The adapter validates echoed request, checkpoint, configuration, and trusted manifest before accepting metrics. It retains infrastructure and malformed-response failures separately from verifier rejection and graded-zero performance.

Development submissions have exactly `format: weightless.development.v1`, `request`, and `checkpoint`. Trusted development code checks graph and tensor structure against the evaluator-owned baseline, permits different valid numerical coefficients, scores public train and development examples, and echoes the actual checkpoint and request binding. It never reads a run or invokes replay. Its package excludes heldout data.

Final submissions are `weightless.run.v2`. Trusted final code checks the evaluator-owned initial checkpoint, policy, budget, and complete expected development manifest including its split identity, replays recorded patches and policy evidence, then scores heldout examples. `grade(frozen)` wraps independent final evidence in the shared result envelope. Final purpose and its request come from the adapter and trusted final package. Final results cannot enter a development run.

The agent image contains the SDK and public problem. Tests, oracle solution, and heldout data remain outside its Docker build context. The native task uses a separate verifier, no-network containers, and only `/app/submission.json` as the declared transferred artifact. Local temporary-directory smoke checks do not prove container isolation.

## Frozen run v2 and replay

A frozen run has exactly `format: weightless.run.v2`, `run`, `initial`, `config`, `policy`, `budget`, `used`, `baseline`, `events`, `checkpoint`, `provenance`, and `id`. Provenance is exactly `sdk: python` and `version: 0.1.0`. The artifact ID hashes all other fields. It is a closed artifact by definition. No old run-v1 import or compatibility path remains.

`replay(frozen)` validates the artifact, configuration, policy, provenance, budget usage, consecutive attempt order, proposal bases, checkpoints, result bindings, decisions, and selected checkpoint. It reconstructs candidate weights with backend patches and reruns only pure policy comparison against recorded evidence. It never calls a numerical evaluator, Harbor, or an external teacher. Replay has no evaluator argument.

Changing a patch, metric, binding, policy, config, event order, decision, counter, or checkpoint without updating all dependent identities fails validation. An attacker can recompute an entirely self-consistent unsigned history. Replay therefore proves consistency and checkpoint reconstruction. It does not authenticate external scores, prove live teacher actions, or promise reproducible behavioral evaluation. Independent final grading establishes heldout correctness.

`WeightlessError.code` distinguishes invalid checkpoint, task, proposal, policy, config, result, artifact, stale base, foreign candidate, pending work, invalid state, closed, budget exhausted, and numerical failure. Messages explain the failure but are not stable API text.
