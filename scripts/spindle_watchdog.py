"""Independent lease for task-owned Spindle apps, including owner-process loss."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        return fields[21]
    except (OSError, IndexError):
        return None


def child_groups(owner):
    children = {owner}
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().split()
            rows.append((int(path.name), int(fields[3])))
        except (OSError, IndexError, ValueError):
            continue
    for _ in range(len(rows)):
        found = {pid for pid, parent in rows if parent in children}
        if found <= children:
            break
        children |= found
    groups = {}
    for pid in children - {owner, os.getpid()}:
        try:
            group = os.getpgid(pid)
            if group == pid and group != os.getpgrp():
                groups[group] = process_identity(group)
        except ProcessLookupError:
            continue
    return groups


def stop_children(owner, known):
    groups = {**known, **child_groups(owner)}
    groups = {group: identity for group, identity in groups.items()
              if identity is not None and process_identity(group) == identity}
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def owned_apps(frontend):
    result = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--env", "main", "--json"],
                                capture_output=True, text=True, check=True, timeout=30)
    return [row for row in json.loads(result.stdout)
                if row["description"] == frontend or (row["description"] or "").startswith(frontend + "-")]


def cleanup(frontend, *, seconds=330):
    deadline = time.monotonic() + seconds
    while True:
        rows = owned_apps(frontend)
        active = [row for row in rows if row["state"] != "stopped" or int(row["tasks"]) != 0]
        if not active:
            return {"confirmed": True, "remaining_apps": [], "remaining_tasks": 0}
        if time.monotonic() >= deadline:
            return {"confirmed": False, "remaining_apps": [row["description"] for row in active],
                    "remaining_tasks": sum(int(row["tasks"]) for row in active)}
        for row in sorted(active, key=lambda row: row["description"] != frontend):
            # Interrupted deploys have an app ID before they have a resolvable name.
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", row["app_id"],
                            "--env", "main"], capture_output=True, text=True,
                           timeout=max(1, min(30, deadline - time.monotonic())))
        time.sleep(3)


def watch(owner, identity, deadline, output, frontend):
    groups = {}
    while process_identity(owner) == identity and time.time() < deadline:
        if (output / "cleanup-confirmed.json").is_file():
            return
        groups.update(child_groups(owner))
        phase = output / "training-deadline.json"
        if phase.is_file():
            try:
                deadline = json.loads(phase.read_text())["deadline"]
            except (OSError, ValueError, KeyError):
                pass
        time.sleep(1)
    if not (output / "cleanup-confirmed.json").is_file():
        stop_children(owner, groups)
        result = cleanup(frontend)
        (output / "watchdog-cleanup.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=int)
    parser.add_argument("--identity")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend", default="wsm-spindle-poc")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        result = cleanup(args.frontend)
        name = "cleanup-confirmed.json" if result["confirmed"] else "cleanup-incomplete.json"
        (args.output / name).write_text(json.dumps(result) + "\n")
        if not result["confirmed"]:
            raise SystemExit(1)
    else:
        watch(args.owner, args.identity, args.deadline, args.output, args.frontend)
