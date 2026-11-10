"""Keyboard controls for trusted Python weight editing."""
import curses
import json
import unicodedata

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



def _main(screen):
    from .coding_tui import execute

    curses.curs_set(0)
    while True:
        screen.erase()
        put(screen, 1, "WSM / Python weight editing", attr=curses.A_BOLD)
        put(screen, 3, "p  Review and execute a trusted Python editing file")
        put(screen, 5, "q  Quit")
        screen.refresh()
        key = screen.get_wch()
        if key in ("q", "\x1b"):
            return 0
        if key == "p":
            try:
                execute(screen)
            except CancelPrompt:
                pass
            except (OSError, ValueError) as error:
                show_result(screen, "Execution error", {"error": str(error)})


def launch():
    return curses.wrapper(_main)
