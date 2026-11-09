# Verify the experiment and grading boundaries

The migration checks passed locally on 2026-11-09. Core checks used Python 3.12.3. Integration checks used Python 3.13.11 with Harbor 0.25.0, Verifiers 0.4.1, and JSON Schema 4.26.0. These are synthetic signed-addition results, not research or model-discovery results.

## Docker-free Linux execution

On 2026-11-09, all 75 tests passed with `WEIGHTLESS_LOCAL_SANDBOX=1`, including 14 real bubblewrap and systemd tests. The host used Python 3.12.3, bubblewrap 0.9.0 from Ubuntu package `0.9.0-1ubuntu0.3`, and systemd 255. No Docker daemon or KVM device was used.

```sh
WEIGHTLESS_LOCAL_SANDBOX=1 PYTHONPATH=packages/python/src \
  python3 -m unittest discover -s packages/python/tests -v
PYTHONPATH=packages/python/src python3 scripts/client_native.py \
  --binary64-only --output /tmp/weightless-local-proof
python3 scripts/package_smoke.py
```

The native checks denied reads of host canaries, host service connections, inherited environment credentials, model writes, and writes to the root and runtime. They exhausted memory, process counts, temporary storage, and both output streams. They checked the actual kernel `memory.max`, `memory.swap.max`, `pids.max`, and `cpu.max` values. Each launched unit and its workload cgroup disappeared after execution, including timeout, overflow, and callback cancellation. A detached child did not survive its workload.

Two additional regressions failed before their fixes and passed afterward. Service-manager expansion changed literal `${PATH}` inside submitted Python source into a manager environment value. Disabling expansion preserves the exact source. A virtual environment under `/tmp` was hidden by the private temporary mount. Mounting the configured environment after that temporary filesystem preserves its imports without exposing sibling host files.

The standalone binary64 driver passed isolation, strict selection, independent private final grading, replay, and reconstruction. It reported `smollm: null` and no fresh teacher run. The wheel smoke passed outside the source tree without optional imports. The [sandbox boundary](sandbox.md) records the threat model, upstream source revisions, and remaining host trust.

The repository-wide `scripts/verify.sh` passed golden identity and its unit phase, then stopped at the optional Harbor check because `harbor` is not installed on this host. Generated Harbor packages were prepared and imported successfully with the new module inventory. Native Harbor execution was not rerun. Torch and Transformers are also absent, so this change does not claim native SmolLM validation.

## Dependency-free behavior and installed package

```sh
INTEGRATION_PYTHON="$PWD/.venv/bin/python" bash scripts/verify.sh
python3 scripts/package_smoke.py
```

The seventeen unit groups passed. They cover unchanged checkpoint golden identity, independent addition predictions across signed inputs, row-major arithmetic, strict proposal validation, stale bases, copied state, foreign or modified candidates, modified results, lazy baseline evaluation, retained baseline failure, genuine graded-zero reward, full request bindings including direct numerical calls, configuration drift, malformed metrics, one-attempt accounting, pending and closed runs, reentrant mutation, training-loss tie-break, development priority, numerical failure, history tampering, bounded history, nested record admission, and oversized evaluator output.

The three example decisions match the pre-migration output exactly. They remain `not_improved`, `improved`, and `not_improved`. Golden checkpoint bytes remain unchanged. The separate identity script encodes the published fixture without importing the SDK.

The package smoke built a wheel, installed it outside the source tree, and ran the new search caller and pure replay. It also replayed a Harbor-produced frozen run without importing Harbor or Verifiers. Pass such a run as an optional argument to `scripts/package_smoke.py` to repeat that check.

## Schema, task loading, and local verifier scripts

`scripts/harbor_smoke.py` validated the JSON Schema, loaded both purpose-specific task packages through actual Harbor and Verifiers models, and checked separate no-network verification and the single declared submission artifact. It checked that the agent build context excludes grader, oracle, and heldout data. The development task excludes heldout data entirely.

The actual trusted final verifier script graded the oracle with reward one and the unchanged network with reward zero. It rejected malformed or duplicate JSON, checkpoint tampering, changed public configuration, a different declared development split, wrong initial checkpoint, wrong policy, and an oversized file. The trusted development script accepted a bound checkpoint submission and rejected a frozen run. These local checks use temporary directory overrides. They do not prove container execution.

## Actual optional adapter in Docker

After installing `packages/python[integration]`, prepare and build the image:

```sh
.venv/bin/python scripts/prepare_harbor.py
docker build -t weightless-signed-addition:local examples/harbor/signed-addition/environment
.venv/bin/python scripts/harbor_native.py --failures --output /tmp/weightless-native-proof
```

The adapter ran the installed Verifiers `HarborTask`, `DockerConfig`, `Deployment`, and isolated `grade_task(on_score=...)` in real Docker. No agent or model rollout occurred. The baseline returned completed native reward zero. The three candidate evaluations returned completed native rewards zero, one, zero, with the original reject, accept, reject decisions and native train/development metrics. Independent final grading of the frozen Harbor-produced run returned reward one with all six heldout examples exact.

Native failure checks passed:

- Nonfinite numerical inference returned `task_failure`, native reward zero, structured failure detail, and verifier exit one.
- A structurally different valid checkpoint returned `task_failure` with native rejected status.
- Malformed diagnostics returned `invalid_response` while preserving authoritative native reward one, raw malformed diagnostics, stdout, and exit zero.
- Valid JSON lists, null values, and nonnumeric metric fields also returned `invalid_response` with native reward one retained. Invalid fields stayed in diagnostic evidence and were excluded from usable metrics.
- Deeply nested verifier diagnostics returned `invalid_response` with native reward one retained as well. When structured evidence cannot fit the envelope, the adapter retains bounded canonical raw JSON and an explicit truncation flag. Run admission checks also reserve the nesting context of the frozen artifact.
- Missing reward returned `invalid_response` with native stderr and verifier exit seven.
- An unavailable Docker socket returned `infrastructure_error` with no invented native reward.

The malformed-diagnostics test initially failed because Verifiers wrapped the callback error in `TaskError`. The adapter now retains capture-error evidence inside the callback and classifies from that evidence. The original failing log remains beside the successful rerun. No retry or score repair was used. Docker had no containers left after the native checks.

Local migration evidence is under `/tmp/weightless-api-update/implementation-proof` and independent verification is under `/tmp/weightless-api-update/parent-proof`. The latter retains failing and passing request-binding, malformed native JSON, and declared split controls. The implementation's `native-final.log` retains the original wrapped callback failure. Unit, schema, wheel, image-build, decision-equivalence, and cleanup evidence remain alongside native JSON artifacts. These paths are local working artifacts.

The parent independently checked 1681 signed predictions and all four training/development loss states against Python integer addition and exact fractions. Current native evidence is in `parent-proof/native-complete`, with the current wheel replay in `parent-proof/wheel-complete.log`. The replay Python had neither Harbor nor Verifiers installed.

## Independent native Harbor CLI

```sh
.venv/bin/harbor run --path examples/harbor/signed-addition --agent oracle --env docker \
  --force-build --jobs-dir /tmp/weightless-jobs --job-name oracle
python3 scripts/check_harbor_reward.py /tmp/weightless-jobs/oracle
```

The native CLI oracle trial passed with reward one and zero trial exceptions. A second native CLI task copy whose solution wrote the unchanged frozen run passed through grading with reward zero and zero trial exceptions. Its diagnostics reported graded status, so this is a genuine task-solving zero. Evidence is in `implementation-proof/jobs/api-oracle` and `implementation-proof/jobs/api-unchanged`.

GitHub Actions remains disabled. The offline tutorial rendered at desktop and phone widths without console errors. Its controls remain an explanatory browser calculation, while its Python callers use the actual SDK. Replay verifies unsigned consistency and checkpoint reconstruction. It neither authenticates external scores nor reruns behavioral evaluation.

## Durable client checks

The standard unit suite includes session, verifier, sandbox, and no-tools transport regressions. `scripts/package_smoke.py` imports the new client from a built wheel without Torch, Transformers, Harbor, or Verifiers. The [native client guide](client.md#reproduce-the-native-checks) runs actual Docker isolation and pinned SmolLM editing. The [acceptance specification](design/agent-client-validation.md) retains broader scientific-validity and unfamiliar-agent comparisons as distinct requirements.
