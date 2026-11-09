# Ownership of an experiment

Weightless separates experiment state from numerical model access and external grading. The ordinary caller still uses `inspect`, `submit`, and `finish`. The [contract](contract.md) specifies every record and boundary.

`weightless.core` owns one run, candidate attempts, copied evidence, the proposal budget, pure metric comparison, and replay. `submit` composes public `propose`, `evaluate`, and `select`. The constructor copies the initial checkpoint, declarative policy, and evaluator configuration without evaluating anything. The core never parses train or development examples. It reads only configured metric paths from a validated result.

`weightless.binary64` owns checkpoint validation and unchanged v1 identity bytes, tensor addresses, patch materialization, affine/ReLU arithmetic, and the numerical evaluator. `NumericalEvaluation` validates and copies its own public task. Changing evaluator does not change the example's lexicographic development-loss and train-loss policy.

`weightless.harbor` is optional. It reuses installed Verifiers `HarborTask`, actual Docker configuration, `Deployment`, and isolated `grade_task(on_score=...)`. Explicit `resolve_image` binds a local image content ID before construction. The evaluator binds package digest, versions, timeout, resources, and purpose. The pre-teardown callback captures native reward files and trusted diagnostics. It preserves native verifier status, exit code, stdout, stderr, and errors.

Trusted `tests/grade.py` owns correctness and split selection. Preparation produces a development package containing only public feedback data and a separate final package containing heldout tests. The agent build context contains the SDK and public problem. Harbor owns provisioning, artifact restoration, verifier staging, scoring, and cleanup. There is no second sandbox implementation or generic backend registry.

## Candidate and evidence lifecycle

A run owns the baseline at attempt zero. Valid proposals reserve consecutive attempts after complete validation and before construction. One pending candidate blocks another proposal. Each candidate has one evaluation and one decision. The first candidate evaluation first retains baseline evidence. Evaluator failures stay in the run, including baseline failures, and never trigger silent retry or repair.

Every request binds run, attempt, checkpoint, purpose, and evaluator configuration. The run checks the returned request, result content identity, finite policy metrics, and exact retained evidence before selection. Deep copies protect nested records. Canonical comparison distinguishes booleans from integers. One run lock rejects overlapping or reentrant mutations. Evaluators receive request and checkpoint copies.

A completed native zero remains a task result. It differs from task execution failure, malformed output, or failed infrastructure. Only completed results with usable policy metrics can improve a completed incumbent. A failed baseline prevents all selection but remains freezeable and inspectable.

## Development and final grading do not form a cycle

Development accepts a checkpoint submission. It neither receives a run nor replays earlier evaluations. The trusted grader chooses public training and development examples and echoes actual checkpoint and request bindings.

`finish` closes the run without doing external work. It preserves pending evidence and freezes the selected checkpoint. Independent final grading accepts this frozen run. The trusted final verifier checks its expected public problem, initial checkpoint, budget, and metric policy, then replays patches and evaluates heldout examples once. Final scores never enter the closed development run.

Replay validates history and reruns the pure policy against recorded metrics. It never calls the numerical evaluator, Harbor, or a teacher. A wheel without Harbor or Verifiers can therefore replay native runs. Hashes catch inconsistent changes, but a malicious author can recompute a complete unsigned history. Replay authenticates neither external scores nor live teacher actions. The separate final verifier establishes task correctness.

The example uses a scripted external proposer and fixed affine/ReLU target. It adds no model client, optimizer, new target backend, hosted service, or GPU path.
