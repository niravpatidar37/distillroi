"""Record the demo session as an asciicast v2 file, without asciinema (which has no Windows build).

    .venv/Scripts/python.exe assets/demo/record.py assets/demo/demo.cast

Each step either prints an amber comment, or "types" a command at the prompt and then RUNS it for real
(subprocess, no shell, no pty) while stdout/stderr are captured in chunks with their real wall-clock
timestamps. Nothing is replayed or edited: every byte after a prompt is the command's own output. The only
synthetic parts are the comments, the prompt and the typing pace.

Precondition: the stack is up (see assets/demo/README.md): vLLM serving the merged q15_r16 student on
127.0.0.1:8105 and the router on 127.0.0.1:8115 with TEACHER_ENABLED=false.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLS, ROWS = 100, 30
PY = sys.executable

# A step is (comment, None, pause) or (shown command, argv, pause). Python is shown as `python` and run
# with this interpreter (the project venv).
STEPS: list[tuple[str, list[str] | None, float]] = [
    ("distillroi: a 1.5B student on vLLM, a confidence router, and the unit economics", None, 1.0),
    ("1. the stack: vLLM serving the merged q15_r16 student + the router API", None, 0.6),
    (
        'docker compose ps --format "table {{.Service}}\\t{{.Status}}"',
        ["docker", "compose", "ps", "--format", "table {{.Service}}\t{{.Status}}"],
        1.2,
    ),
    ("curl -s 127.0.0.1:8115/readyz", ["curl", "-s", "127.0.0.1:8115/readyz"], 1.5),
    ("2. real banking77 test queries: student first, escalate when conf < 0.82", None, 0.6),
    (
        "python assets/demo/ask.py assets/demo/queries.txt",
        [PY, "assets/demo/ask.py", "assets/demo/queries.txt"],
        6.0,
    ),
    ("3. the three axes on all 770 test queries, and the break-even volume", None, 0.6),
    ("python assets/demo/summary.py", [PY, "assets/demo/summary.py"], 6.0),
]

AMBER, GREEN, RESET = "\x1b[1;33m", "\x1b[1;32m", "\x1b[0m"


class Cast:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.events: list[list[object]] = []

    def out(self, data: str) -> None:
        self.events.append([round(time.monotonic() - self.t0, 4), "o", data])

    def sleep(self, s: float) -> None:
        time.sleep(s)


def run(cast: Cast, argv: list[str]) -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "COLUMNS": str(COLS)}
    p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)  # noqa: S603
    assert p.stdout is not None
    buf = b""
    last = "\n"
    while True:
        chunk = os.read(p.stdout.fileno(), 4096)
        if not chunk:
            break
        buf += chunk
        try:
            text = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue  # wait for the rest of a multi-byte character
        buf = b""
        cast.out(text.replace("\r\n", "\n").replace("\n", "\r\n"))
        last = text[-1:] or last
    if last != "\n":
        cast.out("\r\n")  # e.g. curl's JSON has no trailing newline; keep the next prompt on its own line
    if p.wait() != 0:
        raise SystemExit(f"step failed ({p.returncode}): {argv}")


def main(dst: str) -> None:
    cast = Cast()
    for shown, argv, pause in STEPS:
        if argv is None:
            cast.out(f"{AMBER}# {shown}{RESET}\r\n")
        else:
            cast.out(f"{GREEN}${RESET} ")
            for ch in shown:
                cast.out(ch)
                cast.sleep(0.025)
            cast.sleep(0.3)
            cast.out("\r\n")
            run(cast, argv)
        cast.sleep(pause)
    cast.out(
        f"{AMBER}# model weights stay local (artifacts/ is gitignored); next: docker compose down{RESET}\r\n"
    )
    cast.sleep(1.0)
    cast.out("\r\n")
    header = {
        "version": 2,
        "width": COLS,
        "height": ROWS,
        "timestamp": int(time.time()),
        "title": "distillroi: real session",
        "env": {"TERM": "xterm-256color", "SHELL": "bash"},
    }
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        for ev in cast.events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"wrote {dst}: {len(cast.events)} events, {cast.events[-1][0]:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "assets/demo/demo.cast"))
