# Experimeter

A Python library for running continuous, iterative improvement experiments powered by a local LLM.

Give it a single prompt describing your goal. The LLM figures out the rest.

## Key differences from traditional experiment frameworks

| | Traditional | Experimeter |
|---|---|---|
| **Setup** | User picks a "mode" and configures metrics | Single natural-language prompt |
| **Success criteria** | Hard-coded per mode | LLM derives them from your prompt at start |
| **Code execution** | Always on or always off | LLM decides whether execution is needed |
| **Execution sandbox** | `exec()` with full builtins | Isolated child process (`subprocess.run`) with hard timeout |
| **Context management** | Append-only (grows forever) | Compressed: old iterations summarised, JSONL rewritten |

## Installation

```bash
pip install -r experimeter/requirements.txt
```

## Quick start

```python
from experimeter import Experiment, OllamaConfig

exp = Experiment(
    name="fast_fibonacci",
    prompt=(
        "Optimize fibonacci(n) for speed. "
        "Must return correct results for n in 0..40."
    ),
    max_iterations=20,
    ollama_config=OllamaConfig(model="llama3.1"),
)

result = exp.run(initial="""
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)
""")

print(result["best_result"])
```

Run `python example_code.py` or `python example_text.py` for a ready-to-use demonstration.

## Resuming

`run()` detects an existing experiment automatically and resumes it.  You can
also be explicit:

```python
exp.resume()   # raises RuntimeError if no saved state exists
```

Kill the process at any time with Ctrl-C — progress is saved after every
iteration.

## How it works

### Initialisation

Before the loop starts, the LLM reads your prompt and produces a `Criteria`
object containing:

- **task\_description** — what the LLM understood you want
- **success\_indicators** — 3–6 concrete, measurable outcomes
- **evaluation\_rubric** — how to score results 0.0–1.0
- **requires\_execution** — whether code needs to run to measure progress

The criteria are saved to `state.json` and reused on resume.

### Iteration loop

Each iteration follows seven steps:

1. **STRATEGIC REVIEW** — LLM reads compressed summaries + recent iterations
   and proposes the next concrete change.
2. **IMPLEMENT** — LLM writes the new implementation (retried up to 3× on
   syntax error).
3. **SYNTAX CHECK** — `ast.parse()` validates Python before execution.
4. **EXECUTE** — if `requires_execution`, the implementation is appended with
   an LLM-generated harness and run in a child process with a hard timeout.
   The harness prints `{"passed": bool, "details": str}` to stdout.  If the
   harness appears stale (execution fails unexpectedly), it is regenerated
   once and the run is retried.
5. **EVALUATE** — LLM scores the result 0.0–1.0 against the criteria.
6. **STORE** — everything is persisted atomically (temp-file + rename).
7. **COMPRESS** — if the token budget is exceeded, the oldest half of the
   iteration history is summarised by the LLM and the raw JSONL is rewritten
   to contain only the kept iterations.

## Storage layout

```
experiments/{name}/
├── state.json               current progress, criteria, best score
├── memory.jsonl             one JSON object per kept iteration
├── compressed.json          LLM summaries of evicted iterations
├── artifacts/
│   └── best_result.txt      highest-scoring implementation
└── code/
    ├── iteration_N.py
    └── iteration_N_result.json
```

## API reference

### `Experiment`

```python
Experiment(
    name,                        # unique identifier, used as storage folder name
    prompt,                      # natural-language goal (the only required input)
    max_iterations=50,
    max_time_seconds=7200.0,
    storage_dir="./experiments",
    verbose=True,
    ollama_config=OllamaConfig(),
)
```

| Method | Description |
|---|---|
| `run(initial="")` | Start or resume; `initial` is seed content for new experiments |
| `resume()` | Resume only; raises if no saved state |
| `stop()` | Request graceful stop after current iteration |
| `pause()` | Request pause after current iteration |
| `get_status()` | Dict with `status`, `current_iteration`, `best_score`, `elapsed_seconds` |

### `OllamaConfig`

```python
OllamaConfig(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0.4,
    timeout=180,        # per-call timeout in seconds
    num_ctx=32768,      # model context window
)
```

## License

MIT
