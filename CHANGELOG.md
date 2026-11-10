# Changelog

## Unreleased

- Keep program output paths anchored when editing code changes directories, return a failing CLI status for `SystemExit`, and create output parents in quickstart instructions.

- Remove the legacy experiment API, serialized run support, scalar sessions, prompt harness, Tinker bridge, and broker client. `WeightModel` and `execute_program` are the only supported Python API; the CLI and terminal execute Python editing files.

- Preserve Tinker stopping semantics, give code commands stable recovery results, retain encoded inference responses once, and make Session own connection reopening.

- Add a local Tinker SDK sampling backend and explicit weight-editing extension, with isolated teacher Python, durable candidate selection, and budgeted token inference.
- Add an Astro documentation site in `site/` with repository-backed articles, an offline quickstart, local search, code copying, and responsive navigation.
- Add an experimental coding-first tensor SDK, Python-file execution with unchanged-model comparison and verified atomic checkpoint publication, a matching TUI action, and a rerunnable SmolLM Modal proof with independent source-free reload.

- Preserve atomic no-overwrite checkpoint publication, export provenance, failed-run CLI status, and replay validation while integrating the shared harness and generic model backend.

- Preserve checkpoint-specific inspection feedback across teacher turns so the agent can combine observations without repeatedly reading the same tensors.

- Integrate the configuration-driven experiment and export scripts with the shared teacher harness; enforce export size and source boundaries and preserve explicit workspace routing.

- Consolidate the installed `wsm` package, command, and terminal UI. Session remains the policy owner. The prompt harness is `wsm run`. Generic local snapshots, an explicit CUDA device, forward-call reservations, and full checkpoint export sit on the existing CPU identity path.
- Add the operator broker and the unprivileged `wl` byte pipe. `wl` does not own session policy. There is no `weightless` import or console-script shim.
- Keep one introduction at `docs/index.html`. Record the terminal-bench distinction between a handwritten program that passed its verifier and an inconclusive weight-editing run.

- Add the prompt-first CLI and reusable teacher harness with durable recovery and private final grading.

- Rename the project to WSM (Weight Space Modifiers), the Python distribution from `weightless-sdk` to `wsm`, and imports from `weightless` to `wsm`; update callers, sandbox workers, and documentation while preserving serialized format identifiers and hashes.

- Add Docker-free Linux execution with bubblewrap namespaces, systemd resource limits, bounded output, verified workload teardown, and native adversarial checks while preserving numerical and private-verifier contracts.

- Add durable local sessions, private final-verifier registries, explicit no-tools teacher transport, and enforced Docker workers for binary64 and pinned SmolLM2 scalar edits.

- Separate owned search, binary64 access, and optional native Harbor evaluation with bound evidence, retained failures, pure v2 replay, and independent final grading.
- Validate direct evaluator bindings and declared development splits, and retain native rewards when verifier diagnostics are malformed.
- Initialize Weightless with direct numerical weight edits, a Python SDK, independent identity fixtures, and a Harbor-compatible example task.
