# Integration decisions

This checkout starts from `7b44644`. The public name is Weight Space Modifiers. The distribution, import, and command are `wsm`. Serialized `weightless.*` formats and the golden checkpoint identity stay as they are on that commit.

Session remains the only policy owner. The broker, the command line, the terminal UI, and the prompt harness call it. They do not reserve budgets or select arms themselves.

## Two command compositions

Candidate 1 starts from the prompt harness verbs `run`, `resume`, and `status`, then adds the session verbs beside them. `status` and `resume` already mean two different things, so the reader would learn a second vocabulary for the same session methods.

Candidate 2 starts from the session verbs. Those names are the methods in `Session` and the names already used in the client guide. The prompt harness is the extra verb `wsm run`. A directory that contains `run.json` or `harness/report.json` keeps the harness meaning of `status` and `resume` when `--workspace` is absent. An explicit `--workspace` keeps the session meaning.

Candidate 2 is the installed command. It gives one entry point and reuses the names a reader of the session already knows.

## What was kept, and what was not

The prompt harness, the workspace client, the terminal UI, generic local snapshots, the explicit CUDA device, forward-call reservations, and full checkpoint export are in this tree. CPU identity bytes stay on the previous algorithm. CUDA fields appear only when the caller asks for CUDA.

`wl` stays a separate binary because it is an unprivileged byte pipe. The envelopes remain `weightless.command.v1` and `weightless.response.v1`. It does not grow session flags. The earlier idea of a second user-facing experiment command under another name is dropped. There is no `weightless` console script and no import shim.

The short introduction in `docs/index.html` is the one landing page. The generated stage page is not copied. Its useful agent-pipe material is in `docs/agent.md`.

The later format and hash rename is not imported. Packaging copies a clean tree before it builds a wheel so a stale `build/` directory cannot enter the archive or be deleted.

Tinker stays a documented capability boundary. This checkout does not add a live hosted call. The hosted product choice remains open.

Full model export writes the selected numerical state. A later process loads that directory after the original source is gone. A patch journal is not that export.
