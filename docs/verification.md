# Verification

Run `bash scripts/verify.sh` for editing, execution, CLI, and terminal tests. Model tests require the editing dependencies and tokenizers. Run `python3 scripts/package_smoke.py` to build and install a wheel outside the source tree and check its public API and CLI.

The model tests create a tiny local model, edit its tensors, save and reload checkpoints, and check evaluation and publication failures. No model downloads or credentials are needed.

The optional [Modal proof](coding.md#reproduce-the-modal-proof) checks a GPU edit and independent CPU reload. Historical receipts in `docs/design/` describe their original code version and do not verify the current checkout.

## API removal verification

On 2026-11-10, all 25 retained Python tests passed with the model dependencies installed. A real `wsm execute` invocation edited a locally generated tiny model, published its checkpoint, and reloaded it successfully. The terminal opened the Python editing menu and exited cleanly. The wheel smoke and stale-build packaging regression passed. The documentation passed Astro checking and building and all 17 browser tests, including internal links and responsive layouts.

Regression checks also confirm that changing directories inside `edit(model)` preserves the requested output location, `sys.exit(0)` returns a failing CLI status without publication, and the quickstart creates its output parent before execution.

## Modal teacher and Spindle student proof

On 2026-11-10, a real Codex teacher through the CLI proxy completed three turns against a fresh pinned SmolLM2-135M-Instruct target on Modal. Two Python files published checkpoints. Independent final grading found 272 changed canonical tensors, excluding the tied output embedding alias counted by the worker. Held-out mean target-token loss changed from 2.714207 to 1.907411. The two held-out cases were unavailable to the teacher sandbox. This toy task is an integration check, not evidence of general editing ability.

The retained teacher trajectory trained Qwen3-4B LoRA through Modal Spindle using Tinker SDK 0.39.0. One response-only, reward-weighted cross-entropy forward/backward processed 12,469 tokens, including 481 response tokens, followed by one confirmed optimizer update and a saved training checkpoint. Local verification matched the trajectory and datum hashes and checked prompt masks. No student sampling client was created. The [receipt](design/trial-verification.json) links the local proof artifacts.

Recovery preserved the original teacher artifact. An incorrectly resolved virtualenv executable first blocked grading; a missing Jinja2 renderer then stopped the first student attempt before training dispatch. Both were fixed and tested. Explicit recovery verified the prior dispatch journal and zero remaining tasks, then reproduced the same trusted grade before training a fresh student. It did not retry an uncertain optimizer update.

Persistent worse edits, explicit undo, history access, fresh target reset, retained tensor inspection, and sandbox termination passed in a separate scripted Modal mechanics run. They were not all exercised by the three-turn Codex rollout. The final model-runtime suite passed 247 tests with 29 skips; the dependency-free suite passed 248 with 69 skips. The final eight-test distillation suite and wheel smoke passed. All task-owned cloud apps were stopped with zero remaining tasks. Cumulative conservative reservations were US$9.927083 of the authorized US$10, including failed attempts; provider billing has not been reconciled. No hosted Thinking Machines update or student efficacy improvement is claimed.


## Notebook examples

The [Jupyter and marimo examples](../examples/notebooks/README.md) use one uv project and lockfile. Both inspect the retained public proof offline and invoke the existing bounded owner only for an explicitly requested live run. The live extra supplies the editing and Spindle runtime, including the grader's Torch dependencies.

The Jupyter notebook passed nbformat validation and execution of all eight cells in a real kernel. A second kernel execution with live mode enabled and the prior US$9.927083 reservation failed at the insufficient-budget check before process launch. The marimo notebook passed standalone execution and HTML export with live mode enabled while its run button remained unpressed. These notebook checks made no cloud requests or new training updates. The existing Modal and Spindle integration evidence above remains the live proof.

## PR 16 integration with the editing-only base

The merge retains the editing-only public API from main and the experimental trial CLI from PR 16. Trial transport no longer depends on removed sessions, and pinned model staging belongs to the Modal trial module. Legacy experiment commands and the local Tinker bridge remain removed. The prior cloud receipt describes its original code version; cloud training was not repeated for this merge.

The merged wheel built and passed installed API checks. Astro checking and building passed, as did Jupyter offline execution, marimo checking and HTML export, and 35 focused SDK tests with three optional model skips.
The full merged model-runtime suite passed 64 tests with one optional skip.
