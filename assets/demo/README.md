# Demo recording

`demo.gif` (1280×944, 101 frames, ~32 s, ~0.5 MB) is a **real session** recorded on Windows 11 against the local
stack from `docker-compose.yml`: vLLM v0.31.0 serving the merged **q15_r16** student (Qwen2.5-1.5B QLoRA,
bf16) on an RTX 3060 Laptop GPU at 127.0.0.1:8105, and the router API at 127.0.0.1:8115.

## What is real, what is cached

| In the GIF | Source |
|---|---|
| `docker compose ps`, `curl /readyz` | Live output of those commands. |
| Student label, confidence, ACCEPT/ESCALATE reason, latency (`ask.py`) | **Live**: each query is a real `POST /v1/classify` to the router, which called the vLLM student. The two queries that escalated did so in this run (one low confidence, 0.561 < 0.82; one output that is not a valid label, which the router answers with HTTP 503 when the teacher is off). |
| `teacher <label> cached, results/teacher_test_770.jsonl` | **Cached**: the router ran with `TEACHER_ENABLED=false` so the recording used no Groq quota. The teacher label shown is gpt-oss-120b's real answer for that same test query from the Phase-1 teacher run, read from the committed file and labelled "cached" on screen. A live escalation run with real teacher calls is in `results/router_live.json`. |
| Three-axis table and break-even (`summary.py`) | Read from the committed `results/benchmark.json`; V* is **recomputed** with `distill.costs.break_even_requests_per_day` and asserted equal to the committed value. |
| Comments (`# ...`), the `$` prompt, typing pace | Added by `record.py`. `python` on screen is the project venv's interpreter. |

The queries are banking77 **test-split** items from `data/processed/test_770.jsonl`, listed in `queries.txt`.
They were picked from the offline predictions (two confident, two the router escalates); the live run decided
afresh and agreed. The first `ask.py` request in a fresh stack is slower (warm-up); the recording followed a
dry run, so the latencies shown are warm.

## How it was recorded

asciinema has no Windows build, so `record.py` writes the asciicast v2 file itself: it runs each command with
`subprocess` (no shell, no pty), reads stdout/stderr in chunks and stamps every chunk with the real wall-clock
time. [agg](https://github.com/asciinema/agg) 1.5.0 renders the cast, and `frame.py` (adapted from mcpsum's)
adds the window chrome.

```bash
# 1. take the shared GPU lock, free VRAM, start the stack with paid teacher calls off
mkdir F:/.portfolio-brief/gpu.lock
"$LOCALAPPDATA/Programs/Ollama/ollama.exe" stop llama3.1:8b        # if another build left it loaded
STUDENT_MODEL_DIR=./artifacts/merged/q15_r16 ROUTER_THRESHOLD=0.82 TEACHER_ENABLED=false docker compose --profile gpu up -d
curl -s 127.0.0.1:8115/readyz                                      # wait for {"student":true,...}
.venv/Scripts/python.exe assets/demo/ask.py assets/demo/queries.txt  # warm-up dry run

# 2. record, then stop everything and release the lock
.venv/Scripts/python.exe assets/demo/record.py assets/demo/demo.cast
docker compose down && rmdir F:/.portfolio-brief/gpu.lock

# 3. render (brand palette: Ink 14110f background, Snow f5f0e8 text, Copper f0975a for comments/escalations)
agg --theme 14110f,f5f0e8,14110f,f47067,3fb950,f0975a,58a6ff,bc8cff,39c5cf,c9d1d9,6e6a64,ff7b72,56d364,f0975a,79c0ff,d2a8ff,56d4dd,f5f0e8 \
    --font-family "Cascadia Mono,Consolas,Segoe UI Symbol" --font-size 20 \
    --idle-time-limit 6 --last-frame-duration 6 assets/demo/demo.cast raw.gif
uv run --no-project --with pillow python assets/demo/frame.py raw.gif assets/demo/demo.gif "distillroi  —  real session"
```

The model weights (`artifacts/`) are never committed; to re-record you need a locally trained and merged student
(see the README's quick start).
