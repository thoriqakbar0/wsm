# Changelog

## Unreleased

- Consolidate the installed `wsm` package, command, and terminal UI. Session remains the policy owner. The prompt harness is `wsm run`. Generic local snapshots, an explicit CUDA device, forward-call reservations, and full checkpoint export sit on the existing CPU identity path.
- Add the operator broker and the unprivileged `wl` byte pipe. `wl` does not own session policy. There is no `weightless` import or console-script shim.
- Keep one introduction at `docs/index.html`. Record the terminal-bench distinction between a handwritten program that passed its verifier and an inconclusive weight-editing run.

- Rename the project to WSM (Weight Space Mods), the Python distribution from `weightless-sdk` to `wsm`, and imports from `weightless` to `wsm`; update callers, sandbox workers, and documentation while preserving serialized format identifiers and hashes.

- Add Docker-free Linux execution with bubblewrap namespaces, systemd resource limits, bounded output, verified workload teardown, and native adversarial checks while preserving numerical and private-verifier contracts.

- Add durable local sessions, private final-verifier registries, explicit no-tools teacher transport, and enforced Docker workers for binary64 and pinned SmolLM2 scalar edits.

- Separate owned search, binary64 access, and optional native Harbor evaluation with bound evidence, retained failures, pure v2 replay, and independent final grading.
- Validate direct evaluator bindings and declared development splits, and retain native rewards when verifier diagnostics are malformed.
- Initialize Weightless with direct numerical weight edits, a Python SDK, independent identity fixtures, and a Harbor-compatible example task.
