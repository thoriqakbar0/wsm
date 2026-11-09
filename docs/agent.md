# Finish an experiment

The operator command is `wsm`. `wl` only copies one command document and one response document. It does not choose a policy. `wl --help` describes setup. It has no database, socket, Docker, or verifier flag.

Start with this command.

```json
{"format":"weightless.command.v1","action":"observe","key":"wl:observe","payload":{}}
```

Copy one object from `next` and send its `command`. Change only the paths listed in `fill`. Leave every other field as copied.

For a numerical action, choose a unique nonempty key that does not start with `wl:`. Keep that key when retrying the same request. Each edit is `{"tensor":"name","index":0,"value":1.0}`. Inspection indices are integers. Tensor names come from the observation's `parameters`.

When a `submit` command is present and the task asks for a trial, fill that command before you send a `stop` command. Then send each `finish` command in the order listed. A `read` command refreshes the view or repeats a receipt.

A missing response means the process died. Send the same bytes again.

`outcome` `unknown` means the operation was reserved. Do not repeat it under a new key. Send the `stop` command in that response.

`error.code` `stale_base` means nothing was reserved. Send the `observe` command in `next`, then copy the new `submit` command. You may keep the same submit key.

Do not invent a command from `result.actions`. Do not send a conclusion to `wl`. After `export`, read `result.final` for the verifier receipt. If it is null, final grading did not complete. Report that limitation rather than inferring a conclusion.
