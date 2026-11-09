"""Drive an installed curses client through a real terminal using only the standard library."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import termios
import time


class Terminal:
    def __init__(self, command, workspace, log):
        self.master, slave = pty.openpty()
        self.log, self.buffer = log, b""
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 110, 0, 0))
        self.process = subprocess.Popen([command, "--workspace", str(workspace)], stdin=slave, stdout=slave,
                                        stderr=slave, env={**os.environ, "TERM": "xterm-256color"},
                                        start_new_session=True)
        os.close(slave)

    def send(self, value):
        os.write(self.master, value.encode())

    def expect(self, text):
        expected = text.encode()
        deadline = time.monotonic() + 30
        while expected not in self.buffer:
            if time.monotonic() >= deadline:
                raise AssertionError(f"Terminal did not show {text!r}; inspect the saved transcript")
            if not select.select([self.master], [], [], .1)[0]:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    raise AssertionError(f"Terminal exited before {text!r}") from None
                raise
            if not chunk:
                raise AssertionError(f"Terminal closed before {text!r}")
            self.log.write(chunk)
            self.log.flush()
            self.buffer += chunk
        self.buffer = self.buffer.split(expected, 1)[1]

    def resize(self, rows, columns):
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        self.process.send_signal(signal.SIGWINCH)

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
        finally:
            os.close(self.master)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(mode=0o700)
    workspace, private = root / "experiment", root / "private"
    edits, bundle = root / "edits.json", root / "bundle.json"
    edits.write_text(json.dumps([{"tensor": "output.weight", "index": 0, "value": 1},
                                {"tensor": "output.weight", "index": 1, "value": -1}]))
    with (root / "terminal.ansi").open("wb") as log:
        terminal = Terminal(args.command, workspace, log)
        try:
            terminal.expect("New signed-addition example")
            terminal.send("n")
            terminal.expect("Workspace directory")
            terminal.resize(2, 1)
            terminal.send("x")
            time.sleep(.2)
            assert terminal.process.poll() is None, "Shrinking the prompt terminated the TUI"
            terminal.resize(32, 110)
            terminal.expect("Workspace directory")
            terminal.send("\x15" + str(workspace) + "\n")
            terminal.expect("Private demo directory")
            terminal.send(str(private) + "\n")
            terminal.expect("Session opened")
            for key, prompt, answer, result in (
                ("e", "Edits JSON file", str(edits), "Submit completed"),
                ("x", "Export bundle file", str(bundle), "Evidence exported"),
            ):
                if key == "x":
                    terminal.send("f")
                    terminal.expect("Freeze completed")
                    terminal.send("g")
                    terminal.expect("Finalize completed")
                terminal.send(key)
                terminal.expect(prompt)
                terminal.send(answer + "\n")
                terminal.expect(result)
            terminal.send("q")
            assert terminal.process.wait(timeout=10) == 0
        finally:
            terminal.close()
        terminal = Terminal(args.command, workspace, log)
        try:
            terminal.expect("closed")
            terminal.send("q")
            assert terminal.process.wait(timeout=10) == 0
        finally:
            terminal.close()
    evidence = json.loads(bundle.read_text())
    assert evidence["final"]["arms"]["researcher"]["task"]["exact"] == 4
    assert evidence["frozen"]["complete"]
    result = {"passed": True, "real_terminal": True, "resize_during_prompt": True,
              "create_submit_freeze_finalize_export": True, "same_command_reopens_closed_session": True,
              "final_exact": 4, "transcript": str(root / "terminal.ansi")}
    (root / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
