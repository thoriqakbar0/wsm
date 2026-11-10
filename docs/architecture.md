# Ownership of an experiment

The coding-first API is `wsm.editing.WeightModel`. It exposes live PyTorch weights and owns local model loading, public evaluation, and verified checkpoint saving. `wsm.program.execute_program` owns execution of a trusted Python editing file and its lifecycle receipt. Neither imports an agent framework or translates the edit into scalar commands. [The coding guide](coding.md) describes this experimental API and its separate execution boundary. The durable scalar-session architecture below remains available for existing recorded experiments.

WSM separates experiment state from numerical model access and external grading. The ordinary caller still uses `inspect`, `submit`, and `finish`. The [contract](contract.md) specifies every record and boundary.

`wsm.core` owns one run, candidate attempts, copied evidence, the proposal budget, pure metric comparison, and replay. `submit` composes public `propose`, `evaluate`, and `select`. The constructor copies the initial checkpoint, declarative policy, and evaluator configuration without evaluating anything. The core never parses train or development examples. It reads only configured metric paths from a validated result.

`wsm.binary64` owns checkpoint validation and unchanged v1 identity bytes, tensor addresses, patch materialization, affine/ReLU arithmetic, and the numerical evaluator. `NumericalEvaluation` validates and copies its own public task. Changing evaluator does not change the example's lexicographic development-loss and train-loss policy.

`wsm.harbor` is optional. It reuses installed Verifiers `HarborTask`, actual Docker configuration, `Deployment`, and isolated `grade_task(on_score=...)`. Explicit `resolve_image` binds a local image content ID before construction. The evaluator binds package digest, versions, timeout, resources, and purpose. The pre-teardown callback captures native reward files and trusted diagnostics. It preserves native verifier status, exit code, stdout, stderr, and errors.

Trusted `tests/grade.py` owns correctness and split selection. Preparation produces a development package containing only public feedback data and a separate final package containing heldout tests. The agent build context contains the SDK and public problem. Harbor owns provisioning, artifact restoration, verifier staging, scoring, and cleanup. There is no second sandbox implementation or generic backend registry.

## Candidate and evidence lifecycle

A run owns the baseline at attempt zero. Valid proposals reserve consecutive attempts after complete validation and before construction. One pending candidate blocks another proposal. Each candidate has one evaluation and one decision. The first candidate evaluation first retains baseline evidence. Evaluator failures stay in the run, including baseline failures, and never trigger silent retry or repair.

Every request binds run, attempt, checkpoint, purpose, and evaluator configuration. The run checks the returned request, result content identity, finite policy metrics, and exact retained evidence before selection. Deep copies protect nested records. Canonical comparison distinguishes booleans from integers. One run lock rejects overlapping or reentrant mutations. Evaluators receive request and checkpoint copies.

A completed native zero remains a task result. It differs from task execution failure, malformed output, or failed infrastructure. Only completed results with usable policy metrics can improve a completed incumbent. A failed baseline prevents all selection but remains freezeable and inspectable.

## Development and final grading do not form a cycle

Development accepts a checkpoint submission. It neither receives a run nor replays earlier evaluations. The trusted grader chooses public training and development examples and echoes actual checkpoint and request bindings.

`finish` closes the run without doing external work. It preserves pending evidence and freezes the selected checkpoint. Independent final grading accepts this frozen run. The trusted final verifier checks its expected public problem, initial checkpoint, budget, and metric policy, then replays patches and evaluates heldout examples once. Final scores never enter the closed development run.

Replay validates history and reruns the pure policy against recorded metrics. It never calls the numerical evaluator, Harbor, or a teacher. A wheel without Harbor or Verifiers can therefore replay native runs. Hashes catch inconsistent changes, but a malicious author can recompute a complete unsigned history. Replay authenticates neither external scores nor live teacher actions. The separate final verifier establishes task correctness.

The example uses a scripted external proposer and fixed affine/ReLU target. Optional local model snapshots, an explicit CUDA device, and full-checkpoint export sit beside that example. Default runners stay on CPU. Research-aligned edits remain explicit replacements. Hosted Tinker is documented as an open capability boundary, not a live integration.

## Durable local experiment ownership

The [local client](client.md) adds a SQLite-owned session beside `SearchRun`. The session owns admission, budget reservations, arm selection, and the freeze barrier. The private verifier registry owns final dataset claims and disclosure. The Docker runner owns enforced process isolation and numerical execution. The teacher broker owns explicit transport and durable send uncertainty. Agent commands cross an isolated JSON bridge before privileged dispatch.

`Session.run_code` discloses a bounded set of coefficients to an independent probe without model mounts or network access. The probe returns a proposed diff and bounded stdout. Session owns validation and routes changes through `submit`; the code process never owns a candidate checkpoint. Code executions, inspections, attempts, and evaluations retain separate reservations. A retry reuses retained output, and uncertain execution cannot be silently repeated.

The persisted code result names its child submission. Session projects that child through the original code key, so direct results and status recovery share identity, stdout, evaluation, and selection. A retained proposal without an admitted child remains pending and prevents complete freezing. Replay checks that relationship at the freeze cutoff.

`wsm.tinker_backend` translates the supported Tinker sampling protocol onto `Session.infer`, and exposes the WSM editing extension through `wsm.tinker_client.WeightEditor`. The HTTP server owns ephemeral SDK handles and a separate session connection on its server thread. Each sampling handle binds one selected checkpoint. The numerical worker owns token-level greedy inference and prompt log probabilities, while Session reserves and records their cost. The service has no independent model registry or selection policy.
