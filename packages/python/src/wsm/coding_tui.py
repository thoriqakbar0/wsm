"""Terminal controls for the same Python execution command used by coding agents."""
import curses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def stop(process, first_signal):
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def execute(screen):
    from .tui import prompt, put, show_result

    program = Path(prompt(screen, "Python editing file defining edit(model)")).expanduser().resolve()
    show_result(screen, "Review Python source before execution", program.read_text().splitlines())
    model = Path(prompt(screen, "Local model snapshot directory")).expanduser().resolve()
    cases = Path(prompt(screen, "Public evaluation cases JSON")).expanduser().resolve()
    output = Path(prompt(screen, "New output directory")).expanduser().resolve()
    device = prompt(screen, "Torch device", default="cpu")
    command = [sys.executable, "-m", "wsm", "execute", str(program), "--model", str(model),
               "--cases", str(cases), "--output", str(output), "--device", device]
    process = None
    with tempfile.TemporaryFile(mode="w+b") as log:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            screen.timeout(100)
            while process.poll() is None:
                screen.erase()
                put(screen, 1, "WSM / execute Python", attr=curses.A_BOLD)
                put(screen, 3, str(program))
                state = "starting"
                try:
                    state = json.loads((output / "run.json").read_text()).get("state", state)
                except (OSError, ValueError):
                    pass
                put(screen, 5, "State  " + state)
                put(screen, 7, "Output  " + str(output))
                put(screen, 9, "Ctrl-C interrupts execution. Publication is recorded in run.json.")
                screen.refresh()
                try:
                    screen.get_wch()
                except curses.error:
                    pass
        except KeyboardInterrupt:
            if process is not None:
                stop(process, signal.SIGINT)
        finally:
            screen.timeout(-1)
            if process is not None:
                stop(process, signal.SIGTERM)
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 16000))
        text = log.read(16000).decode('utf-8', errors='replace')
    try:
        result = json.loads((output / "run.json").read_text())
    except (OSError, ValueError):
        result = {"state": "unknown", "output": str(output),
                  "detail": "Inspect output before retrying. No completion receipt was readable."}
    show_result(screen, "Python execution result", {"run": result, "process_output": text})
    return result
