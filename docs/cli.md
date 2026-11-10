# Commands

`wsm execute edit.py --model /local/snapshot --cases public-cases.json --output runs/edit --device cpu` executes a trusted Python file defining `edit(model)`. It compares unchanged and edited weights on the same public cases and publishes a reload-verified checkpoint. The output directory must be new. `--device cuda` explicitly selects a GPU runtime. See [the coding guide](coding.md) for the API and execution boundary. In the TUI startup screen, **p** opens this workflow.

The installed command is `wsm`. It is also `python -m wsm` and `python -m wsm.cli`. There is no second console script.

Session remains the owner of budgets, stale operations, uncertain sends, selection, and replay. The command names for those steps match the session methods.

## Session workspace

`wsm init` creates a workspace. The default directory is `wsm-experiment`. `observe`, `inspect`, `propose`, `submit`, `diagnose`, `freeze`, `finalize`, `cancel`, `export`, `verify`, `teacher`, `models`, and `stage-model` use that workspace. `export` writes the evidence bundle. `export-model` writes the selected checkpoint as files a later process can load without the original source.

`status KEY --workspace DIR` reads one saved operation. `resume REQUEST --workspace DIR` retries one saved command envelope. Repeating a key does not spend the budget again.

With no arguments, `wsm` opens the terminal UI when stdin and stdout are a terminal. Otherwise it returns a JSON `terminal_required` error.

Errors from these commands are JSON on stderr. Stdout is empty when the command fails.

## Prompt harness

`wsm run PROMPT --task TASK --output DIR` starts a prompt-driven run. `addition` selects the bundled task. The teacher URL, model, and `WSM_TEACHER_API_KEY` are explicit. There is no route fallback.

Add `--code-executions N` to permit bounded teacher Python over explicitly inspected coefficients. Each code command runs in the configured isolated runner and submits its resulting diff through session selection. Omitting the flag disables code execution.

`wsm status DIR` and `wsm resume DIR` use the prompt harness when `DIR` contains `run.json` or `harness/report.json` and the command does not pass `--workspace`. Passing `--workspace` keeps the session meaning above.

Harness errors use the text form `wsm: code. message`.

## What is not a second policy

`wl` copies one `weightless.command.v1` document to the operator broker and writes one `weightless.response.v1` document. It refuses database, socket, Docker, and verifier arguments. The broker calls `Session.dispatch`. See [the agent note](agent.md).
