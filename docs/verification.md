# Verification

Run `bash scripts/verify.sh` for editing, execution, CLI, and terminal tests. Model tests require the editing dependencies and tokenizers. Run `python3 scripts/package_smoke.py` to build and install a wheel outside the source tree and check its public API and CLI.

The model tests create a tiny local model, edit its tensors, save and reload checkpoints, and check evaluation and publication failures. No model downloads or credentials are needed.

The optional [Modal proof](coding.md#reproduce-the-modal-proof) checks a GPU edit and independent CPU reload. Historical receipts in `docs/design/` describe their original code version and do not verify the current checkout.

## API removal verification

On 2026-11-10, all 25 retained Python tests passed with the model dependencies installed. A real `wsm execute` invocation edited a locally generated tiny model, published its checkpoint, and reloaded it successfully. The terminal opened the Python editing menu and exited cleanly. The wheel smoke and stale-build packaging regression passed. The documentation passed Astro checking and building and all 17 browser tests, including internal links and responsive layouts.

Regression checks also confirm that changing directories inside `edit(model)` preserves the requested output location, `sys.exit(0)` returns a failing CLI status without publication, and the quickstart creates its output parent before execution.
