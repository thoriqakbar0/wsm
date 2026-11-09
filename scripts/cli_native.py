"""Exercise installed commands against the enforced runner and retained SDK state."""
import argparse
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

from wsm import Experiment, replay_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Installed wsm executable")
    parser.add_argument("--output", type=Path, required=True, help="Unused proof directory")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(mode=0o700)
    workspace, private = root / "experiment", root / "private"
    trace = []

    def call(*command, workspace_path=workspace, executable=args.command, success=True):
        argv = [executable, "--json", "--workspace", str(workspace_path), *map(str, command)]
        process = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        trace.append({"arguments": argv[1:], "exit": process.returncode})
        if success:
            assert process.returncode == 0, (argv, process.stderr)
            return json.loads(process.stdout)
        assert process.returncode != 0, argv
        assert not process.stdout, process.stdout
        return json.loads(process.stderr)

    call("init", "--example", "addition", "--private-dir", private)
    initial = call("observe")
    assert initial["remaining"]["attempts"] == 8
    inspected = call("inspect", "--tensor", "output.weight", "--indices", 0, 1, "--key", "read-zero")
    assert [item["value"] for item in inspected["result"]["values"]] == [0, 0]
    edits = root / "edits.json"
    edits.write_text(json.dumps([{"tensor": "output.weight", "index": 0, "value": 1},
                                {"tensor": "output.weight", "index": 1, "value": -1}]))
    proposal = root / "proposal.json"
    call("propose", "--edits", edits, "--output", proposal)
    stale = root / "stale.json"
    stale.write_bytes(proposal.read_bytes())
    selected = call("submit", proposal, "--key", "improve")
    assert selected["selection"]["accepted"] is True
    remaining = call("observe")["remaining"]
    assert remaining["attempts"] == 7 and remaining["evaluations"] == 10
    assert call("submit", proposal, "--key", "improve") == selected
    assert call("observe")["remaining"] == remaining
    call("submit", stale, "--key", "stale", success=False)
    assert call("observe")["remaining"] == remaining
    frozen = call("freeze", "--key", "freeze")
    final = call("--verifier", private / "verifier.json", "finalize", "--key", "grade")
    assert final["arms"]["researcher"]["task"]["exact"] == 4
    bundle_path = root / "bundle.json"
    call("export", "--output", bundle_path)
    bundle = json.loads(bundle_path.read_text())
    assert "demo-final-" not in json.dumps(bundle)
    assert replay_session(bundle) == frozen
    assert call("replay", bundle_path) == frozen
    call("verify", bundle_path)
    with Experiment.open(workspace) as sdk:
        assert sdk.session.export()["frozen"] == frozen
        assert sdk.session.export()["used"] == bundle["used"]

    interrupted = root / "interrupted"
    call("init", "--example", "addition", workspace_path=interrupted)
    interrupted_proposal = root / "interrupted-proposal.json"
    call("propose", "--edits", edits, "--output", interrupted_proposal, workspace_path=interrupted)
    argv = [args.command, "--json", "--workspace", str(interrupted), "submit",
            str(interrupted_proposal), "--key", "interrupted-edit"]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 10
    reserved = False
    try:
        while process.poll() is None and time.monotonic() < deadline:
            with sqlite3.connect(interrupted / "session.db") as db:
                row = db.execute("SELECT body FROM operations WHERE key=?", ("interrupted-edit",)).fetchone()
            if row and json.loads(row[0])["status"] == "dispatched":
                reserved = True
                process.send_signal(signal.SIGINT)
                break
            time.sleep(.01)
        stdout, stderr = process.communicate(timeout=30)
        assert reserved, "Did not observe dispatch before the operation finished"
        interrupt_exit = process.returncode
        if interrupt_exit != 130:
            assert interrupt_exit == 1 and json.loads(stdout)["status"] == "outcome_unknown", (interrupt_exit, stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    with Experiment.open(interrupted) as sdk:
        before = sdk.session.export()["used"]
        command = {"action": "submit", "key": "interrupted-edit",
                   "payload": json.loads(interrupted_proposal.read_text())}
        status = sdk.execute(command)
        assert status["status"] in ("dispatched", "outcome_unknown"), status
        assert sdk.session.export()["used"] == before
        partial = sdk.session.freeze(key="partial")
        assert partial["complete"] is False
        assert "interrupted-edit" in partial["obligations"]
    result = {"passed": True, "installed_commands": [args.command],
              "final_exact": 4, "cli_sdk_same_frozen_checkpoint": True,
              "same_key_retry_spends_no_extra_budget": True,
              "stale_proposal_spends_no_budget": True,
              "interrupt_after_dispatch_exit": interrupt_exit,
              "interrupted_operation_not_resent": True,
              "partial_freeze_keeps_obligations": True, "trace": trace}
    (root / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
