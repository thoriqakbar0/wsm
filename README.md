# WSM (Weight Space Modifiers)

WSM lets an AI teacher inspect a model's weights, propose edits, and test them. It saves the selected checkpoint and evaluation evidence.

```sh
wsm run "Improve signed addition. Inspect the weights and test your changes." \
  --task addition --attempts 20 --output runs/addition
wsm status runs/addition
```

This example needs a teacher endpoint and a sandboxed runner. Setup is below.

## Run with a teacher

Set `WSM_TEACHER_URL` to a streaming Responses-compatible base URL, `WSM_TEACHER_MODEL` to the model ID, and `WSM_TEACHER_API_KEY` to its credential.

The default runner needs a root-authorized Linux host with systemd and bubblewrap. Follow the [sandbox setup](docs/sandbox.md#run-on-this-host), or use Docker with `--runner docker --image <existing-image>`. There is no unsandboxed fallback for `wsm run`.

Run the command above with a new output directory. A completed run saves `checkpoint.json`, `mod.json`, and `harness/report.json`. Its public evidence is in `harness/bundle.json`. Keep the run directory private because it also contains teacher requests and recovery state.

The bundled addition task has public examples only, so its final conclusion is `inconclusive`. See the [prompt-driven guide](docs/wsm.md) for custom tasks, private grading, and recovery. Run `wsm --help` for commands, or open the terminal UI by running `wsm` without arguments.

## Try the offline example

From the repository root, with the `wsm` package available:

```sh
python3 examples/demo.py --output /tmp/wsm-run.json
```

The scripted teacher edits a small signed-addition network. It needs no model call, credentials, Docker, or bubblewrap. The output is:

```json
{"accepted": false, "reason": "not_improved"}
{"accepted": true, "reason": "improved"}
{"accepted": false, "reason": "not_improved"}
```

The saved file contains the selected checkpoint and evidence. To try other edits, pass `--proposals` with a file in the format of [the bundled proposals](examples/teacher-proposals.json).

## Other workflows

- [Python SDK](docs/contract.md#intended-caller): inspect a checkpoint, submit edits, and replay a frozen run.
- [Configured experiments](docs/driver.md): run scripted proposals or a teacher from a JSON configuration and export the selected checkpoint.
- [Sessions and model targets](docs/client.md): resume experiments and work with binary64 or supported local model snapshots.
- [Harbor grading](docs/contract.md#native-harbor-boundary): use the optional Docker-based evaluator.
- [Verification](docs/verification.md): run the core checks and see the requirements for native sandbox and Harbor checks.

## Research scope

WSM follows the direct-editing workflow described in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). The offline example demonstrates edits and selection on a synthetic task. It does not reproduce the paper's results or establish that an AI discovered a useful model edit. See [research scope](docs/research.md) for the current evidence and limits.
