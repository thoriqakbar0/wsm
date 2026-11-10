"""Keyboard-driven experiment notebook using the same saved commands as the CLI."""
import curses
import json
from pathlib import Path
import sys
import unicodedata
import uuid

from .binary64 import WeightlessError
from .client import Experiment, addition_config, create_example_verifier, read_json, write_json
from .cli import ask_teacher, result_exit_code, save_command


class CancelPrompt(Exception):
    pass


def clean(value):
    return "".join(c if c.isprintable() and not unicodedata.category(c).startswith("C") else " " for c in str(value))


def put(screen, row, text, *, col=2, attr=0):
    height, width = screen.getmaxyx()
    if not 0 <= row < height or col >= width - 1:
        return
    try:
        screen.addnstr(row, col, clean(text), max(0, width - col - 2), attr)
    except curses.error:
        pass


def prompt(screen, label, *, default=""):
    value = default
    curses.curs_set(1)
    try:
        while True:
            height, width = screen.getmaxyx()
            if height < 8 or width < 30:
                screen.erase()
                put(screen, 0, "Resize to continue. Esc cancels.", col=0)
                screen.refresh()
                if screen.get_wch() == "\x1b":
                    raise CancelPrompt()
                continue
            screen.move(max(0, height - 3), 0)
            screen.clrtoeol()
            put(screen, height - 3, label + "  (Esc cancels)", attr=curses.A_BOLD)
            screen.move(max(0, height - 2), 0)
            screen.clrtoeol()
            visible = clean(value)[-max(1, width - 6):]
            put(screen, height - 2, "> " + visible)
            screen.move(max(0, height - 2), min(width - 2, 4 + len(visible)))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\n", "\r", curses.KEY_ENTER):
                return value
            if key == "\x1b":
                raise CancelPrompt()
            if key in ("\b", "\x7f", curses.KEY_BACKSPACE):
                value = value[:-1]
            elif key == "\x15":
                value = ""
            elif isinstance(key, str) and key.isprintable() and len(value) < 4096:
                value += key
    finally:
        curses.curs_set(0)


def show_result(screen, title, result):
    lines = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).splitlines()
    offset = 0
    while True:
        screen.erase()
        height, _ = screen.getmaxyx()
        put(screen, 1, title, attr=curses.A_BOLD)
        for row, line in enumerate(lines[offset:offset + max(1, height - 6)], 3):
            put(screen, row, line)
        put(screen, height - 2, "Up/down scroll   PgUp/PgDn page   Enter or q returns", attr=curses.A_REVERSE)
        screen.refresh()
        key = screen.get_wch()
        if key in ("q", "\n", "\r", "\x1b"):
            return
        page = max(1, height - 6)
        if key in ("j", curses.KEY_DOWN):
            offset = min(max(0, len(lines) - page), offset + 1)
        elif key in ("k", curses.KEY_UP):
            offset = max(0, offset - 1)
        elif key == curses.KEY_NPAGE:
            offset = min(max(0, len(lines) - page), offset + page)
        elif key == curses.KEY_PPAGE:
            offset = max(0, offset - page)


class Terminal:
    def __init__(self, screen, workspace, verifier_config, arm):
        self.screen = screen
        self.workspace = str(Path(workspace).resolve())
        self.verifier_config, self.arm = verifier_config, arm
        self.initial_arm = arm
        self.experiment = None
        self.message = "Choose an experiment to begin."
        self.last_result = None
        self.last_request = None
        self.accent = curses.A_BOLD
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            self.accent |= curses.color_pair(1)

    def draw(self, view=None):
        screen = self.screen
        screen.erase()
        height, width = screen.getmaxyx()
        put(screen, 1, "WSM / weight-editing experiments", attr=self.accent)
        if height < 20 or width < 64:
            put(screen, 3, "Resize to at least 64 columns by 20 rows.")
            put(screen, height - 2, "q quits without freezing the experiment")
            screen.refresh()
            return
        put(screen, 3, "Workspace  " + self.workspace)
        if view is None:
            put(screen, 5, "Write Python that changes tensors, then compare and save a checkpoint.")
            put(screen, 7, "p  Review and execute a Python editing file")
            put(screen, 8, "n  New signed-addition example")
            put(screen, 9, "c  Create from a workspace configuration JSON")
            put(screen, 10, "o  Open an existing experiment")
            put(screen, 11, "Teacher models propose edits. Target models provide the editable weights.")
            put(screen, 12, "Custom local target snapshots are selected in workspace configuration.")
            put(screen, 14, "q  Quit")
        else:
            target = self.experiment.config["target"]
            put(screen, 5, f"Phase  {view['phase']}    Arm  {view['arm']}    Target  {target.get('model', target['kind'])}", attr=self.accent)
            put(screen, 6, "Checkpoint  " + view["base"])
            remaining = view["remaining"]
            put(screen, 8, "Remaining budgets", attr=curses.A_BOLD)
            put(screen, 9, "   ".join(f"{name} {value}" for name, value in list(remaining.items())[:3]))
            put(screen, 10, "   ".join(f"{name} {value}" for name, value in list(remaining.items())[3:]))
            names = ", ".join(view["parameters"])
            put(screen, 12, "Editable  " + names)
            put(screen, 13, "Available  " + ", ".join(view["actions"]))
            if view["pending"]:
                put(screen, 14, "Pending operation  " + str(view["pending"]), attr=curses.A_BOLD)
            else:
                history = view["history"]
                put(screen, 14, "Latest decision  " + (json.dumps(history[-1], separators=(",", ":")) if history else "No proposals evaluated yet"))
            put(screen, 16, "P Python program   i Inspect   e Scalar edit   p Proposal   t Teacher   h History")
            put(screen, 17, "f Freeze    g Final grade    x Export    d Diagnose    v Last result")
            put(screen, 18, "s Status    r Resume saved request    a Arm    b Back    q Quit")
        put(screen, height - 4, self.message, attr=curses.A_BOLD)
        if self.last_request:
            put(screen, height - 3, "Saved request  " + str(self.last_request))
        put(screen, height - 2, "Quit preserves the session. Freeze closes selection. No automatic retries.", attr=curses.A_REVERSE)
        screen.refresh()

    def open(self, directory, *, create_config=None):
        if self.experiment is not None:
            self.experiment.close()
            self.experiment = None
        self.workspace = str(Path(directory).resolve())
        factory = Experiment.open if create_config is None else Experiment.create
        opened = factory(self.workspace, **({} if create_config is None else {"config": create_config}),
                         verifier_config=self.verifier_config)
        try:
            self.arm = self.initial_arm or opened.config["declaration"]["arms"][0]
            opened.session.observe(self.arm)
        except BaseException:
            opened.close()
            raise
        self.experiment = opened
        self.last_result = None
        self.last_request = None
        self.message = "Session opened. Select an action."

    def execute(self, action, payload, *, key=None):
        key = key or f"{action}-{uuid.uuid4().hex}"
        command = {"action": action, "key": key, "payload": payload}
        self.last_request = save_command(self.experiment, command, self.arm)
        self.message = f"Running {action}. Operation {key}"
        self.draw(self.experiment.session.observe(self.arm))
        self.last_result = self.experiment.execute(command, arm=self.arm)
        state = "needs attention" if result_exit_code(self.last_result) else "completed"
        self.message = f"{action.capitalize()} {state}. Operation {key}. Press v for the result."

    def startup(self, key):
        if key == "p":
            from .coding_tui import execute
            self.last_result = execute(self.screen)
            self.message = "Python execution finished. Its receipt is in the output directory."
        elif key == "n":
            directory = prompt(self.screen, "Workspace directory", default=self.workspace)
            private = prompt(self.screen, "Private demo directory (blank skips final verifier)")
            self.verifier_config = None
            if private:
                if Path(private).resolve().is_relative_to(Path(directory).resolve()):
                    raise WeightlessError("invalid_config", "Keep private demo data outside the workspace")
                self.verifier_config = str(create_example_verifier(private))
            self.open(directory, create_config=addition_config())
        elif key == "c":
            config = read_json(prompt(self.screen, "Workspace configuration JSON"))
            directory = prompt(self.screen, "Workspace directory", default=self.workspace)
            verifier = prompt(self.screen, "Verifier JSON (blank skips final verifier)", default=self.verifier_config or "")
            self.verifier_config = verifier or None
            self.open(directory, create_config=config)
        elif key == "o":
            directory = prompt(self.screen, "Workspace directory", default=self.workspace)
            verifier = prompt(self.screen, "Verifier JSON (blank skips final verifier)", default=self.verifier_config or "")
            self.verifier_config = verifier or None
            self.open(directory)

    def action(self, key, view):
        if key == "P":
            from .coding_tui import execute
            self.last_result = execute(self.screen)
            self.message = "Python execution finished. Its receipt is in the output directory."
        elif key == "i":
            tensor = prompt(self.screen, "Tensor name", default=next(iter(view["parameters"])))
            indices = [int(item) for item in prompt(self.screen, "Indices separated by spaces", default="0").split()]
            self.execute("inspect", {"checkpoint": view["base"], "tensor": tensor, "indices": indices})
        elif key == "e":
            path = prompt(self.screen, "Edits JSON file (blank enters one scalar)")
            if path:
                edits = read_json(path)
            else:
                tensor = prompt(self.screen, "Tensor name", default=next(iter(view["parameters"])))
                index = int(prompt(self.screen, "Scalar index", default="0"))
                value = float(prompt(self.screen, "Replacement value"))
                edits = [{"tensor": tensor, "index": index, "value": value}]
            proposal = self.experiment.proposal(edits, arm=self.arm)
            self.execute("submit", proposal)
        elif key == "p":
            proposal = read_json(prompt(self.screen, "Bound proposal JSON file"))
            self.execute("submit", proposal)
        elif key == "t":
            config = prompt(self.screen, "Private teacher configuration JSON")
            prompt_path = prompt(self.screen, "Literal teacher prompt file")
            operation = prompt(self.screen, "Teacher operation key (reuse exactly on resume)", default="teacher-" + uuid.uuid4().hex)
            self.message = "Teacher request in progress. Its reply will not execute automatically."
            self.draw(view)
            self.last_result = ask_teacher(self.experiment, config, prompt_path, key=operation, arm=self.arm)
            self.message = "Teacher reply retained. Press v to read it; submit a reviewed proposal with p."
        elif key in ("f", "g", "d"):
            self.execute({"f": "freeze", "g": "finalize", "d": "diagnose"}[key], {})
        elif key == "x":
            path = prompt(self.screen, "Export bundle file (must not exist)")
            bundle = self.experiment.execute({"action": "export", "key": "export", "payload": {}}, arm=self.arm)
            write_json(path, bundle)
            self.message = "Evidence exported to " + path
        elif key == "s":
            operation = prompt(self.screen, "Operation key")
            self.last_result = self.experiment.session.status(operation, arm=self.arm)
            show_result(self.screen, "Operation status", self.last_result)
        elif key == "r":
            path = prompt(self.screen, "Saved request file", default=str(self.last_request or ""))
            record = read_json(path)
            if type(record) is not dict or set(record) != {"arm", "command"}:
                raise WeightlessError("invalid_request", "Expected a saved WSM command envelope")
            self.arm = self.experiment.arm(record["arm"])
            self.execute(record["command"]["action"], record["command"]["payload"], key=record["command"]["key"])
        elif key == "h":
            show_result(self.screen, "Decision history", view["history"])
        elif key == "v" and self.last_result is not None:
            show_result(self.screen, "Last action result", self.last_result)
        elif key == "a":
            arm = prompt(self.screen, "Arm name")
            self.experiment.session.observe(arm)
            self.arm = arm
        elif key == "b":
            self.experiment.close()
            self.experiment = None
            self.message = "Session closed locally. Choose another experiment."

    def run(self):
        curses.curs_set(0)
        if (Path(self.workspace) / "session.db").is_file():
            try:
                self.open(self.workspace)
            except (WeightlessError, OSError) as error:
                self.message = "Cannot open workspace. " + str(error)
        try:
            while True:
                view = self.experiment.session.observe(self.arm) if self.experiment else None
                self.draw(view)
                key = self.screen.get_wch()
                if key == "q":
                    return 0
                height, width = self.screen.getmaxyx()
                if height < 20 or width < 64:
                    continue
                try:
                    if self.experiment is None:
                        self.startup(key)
                    else:
                        self.action(key, view)
                except CancelPrompt:
                    self.message = "Action cancelled before dispatch."
                except WeightlessError as error:
                    self.message = f"{error.code}: {error}"
                except (OSError, ValueError, TypeError, KeyError, StopIteration):
                    self.message = "Action failed. Check file paths and the expected input format."
        finally:
            if self.experiment is not None:
                self.experiment.close()


def launch(workspace, *, verifier_config=None, arm=None):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise WeightlessError("terminal_required", "The TUI requires an interactive terminal")
    try:
        return curses.wrapper(lambda screen: Terminal(screen, workspace, verifier_config, arm).run())
    except curses.error:
        raise WeightlessError("terminal_unavailable", "Terminal setup failed; select a supported TERM or use scriptable commands") from None
