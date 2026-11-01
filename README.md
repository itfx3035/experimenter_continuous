# Experimenter

A Python library for running continuous, iterative improvement experiments
powered by a local LLM.

Give it a single prompt describing your goal plus one flag
(`requires_execution`), and the LLM figures out the rest.

## Key differences from traditional experiment frameworks

| | Traditional | Experimenter |
|---|---|---|
| **Setup** | User picks a "mode" and configures metrics | Single natural-language prompt + one `requires_execution` flag |
| **Success criteria** | Hard-coded per mode | LLM derives them from your prompt at start |
| **Code execution** | Always on or always off | Chosen by the user; the LLM then writes its own test harness |
| **Execution sandbox** | `exec()` with full builtins | Isolated child process (`subprocess.run`) with hard timeout |
| **Context management** | Append-only (grows forever) | Compressed: old iterations summarised, JSONL rewritten, evicted history archived |

## Prerequisites

- Python 3.10+
- An Ollama server you can reach — `ollama serve` (https://ollama.com), or any
  OpenAI-compatible Ollama endpoint over HTTP
- A model pulled into that server, e.g. `ollama pull llama3.1`
  (`ollama list` shows the names you can use)

The default model name is `llama3.1`.  **Change it** to a model you actually have
(both in the examples and in `OllamaConfig(model=...)`); a name that is not
pulled locally fails at the first LLM call.

`experimenter/ollama_config.example.json` is shipped as a template only — copy it
to `./ollama_config.json` to customise without editing code.  Lookup order when
no path is passed: explicit `config_path` → `./ollama_config.json` (current
working directory) → `ollama_config.json` inside the `experimenter` package →
built-in defaults.  Only the `.example.json` template is committed; personal
`ollama_config.json` files are gitignored.

## Installation

```bash
pip install -r requirements.txt      # or: pip install -e .
```

## Quick start

```python
from experimenter import Experiment, OllamaConfig

exp = Experiment(
    name="fast_fibonacci",
    prompt=(
        "Optimize fibonacci(n) for speed. "
        "Must return correct results for n in 0..40."
    ),
    requires_execution=True,          # required argument
    max_iterations=20,
    ollama_config=OllamaConfig(model="llama3.1"),   # must be a model you have pulled
)

result = exp.run(initial="""
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)
""")

print(result["best_result"])
```

Run `python example_code.py` or `python example_text.py` for a ready-to-use
demonstration (both use `llama3.1` — point `model` at a model you have pulled).

## Resuming

`run()` detects an existing experiment automatically and resumes it.  You can
also be explicit:

```python
exp.resume()   # raises RuntimeError if no saved state exists,
               # ValueError if state.json exists but is corrupt
```

Kill the process at any time with Ctrl-C — progress is saved after every
iteration.  The first Ctrl-C means "finish this iteration, save, then stop"; a
second Ctrl-C aborts immediately (state is still saved before the exception
propagates).

Signal handlers can only be installed from the main thread; when an `Experiment`
is built inside a thread the library logs a warning instead of crashing and you
must call `exp.stop()` yourself.

## How it works

### Initialisation

Before the loop starts, the LLM reads your prompt and produces a `Criteria`
object containing:

- **task\_description** — what the LLM understood you want
- **success\_indicators** — 3–6 concrete, measurable outcomes
- **evaluation\_rubric** — how to score results 0.0–1.0
- **evaluation\_rubric** — how to score results 0.0–1.0
- **requires\_execution** — copied from the constructor argument (the user
  chooses; the LLM only writes the harness)

The criteria are saved to `state.json` and reused on resume.

### Iteration loop

Each iteration follows seven steps:

1. **STRATEGIC REVIEW** — LLM reads compressed summaries + recent iterations
   and proposes the next concrete change.
2. **IMPLEMENT** — LLM writes the new implementation (retried up to 3× on
   syntax error).
3. **SYNTAX CHECK** — `ast.parse()` validates Python before execution.
4. **EXECUTE** — if `requires_execution`, the implementation is appended with
   an LLM-generated harness and run in a child process (`python -I`, minimal
   environment, optional memory/CPU rlimits) with a hard timeout.  Output
   produced before the timeout is kept, which is often what tells you *why* it
   hung.  The harness prints `{"passed": bool, "correct": bool, "score": float,
   "details": str}` to stdout.  A top-level `if __name__ == "__main__":` block
   is removed from the implementation so the harness stays in control.  If the
   harness appears stale (execution fails unexpectedly), it is regenerated once
   and the run is retried.
5. **EVALUATE** — the harness score is used when it provides one; otherwise the
   LLM scores the result 0.0–1.0 against the criteria.
6. **STORE** — everything is persisted atomically (temp-file + rename),
   including iterations where implementation failed, so failed attempts show up
   in the "already tried" list.
7. **COMPRESS** — the token budget comes from `num_ctx` (not a fixed constant)
   minus room reserved for the model's reply and the static prompt.  When it is
   exceeded, the oldest half of the iteration history is summarised by the LLM
   and the raw JSONL is rewritten to contain only the kept iterations.  The
   global best is never evicted, evicted iterations are archived, the number of
   summaries is capped (older ones are merged), and the "already tried" list is
   capped so the prompt cannot grow without bound.

## Storage layout

```
experiments/{name}/
├── state.json               current progress, criteria, best score,
│                            stagnation count
├── lock                     lock file — one process per experiment name
├── memory.jsonl             one JSON object per kept iteration
├── compressed.json          LLM summaries of evicted iterations
├── archive.jsonl            evicted iterations (never lost)
├── experiment.log           step-by-step log (experimenter.* loggers)
├── llm.log                  full prompts + responses
├── artifacts/
│   └── best_result.txt      highest-scoring implementation
└── code/
    ├── iteration_N.py
    └── iteration_N_result.json   (execution tasks only)
```

## API reference

### `Experiment`

```python
Experiment(
    name,                        # unique identifier, used as storage folder name
    prompt,                      # natural-language goal
    requires_execution,          # required: run code or evaluate text only
    max_iterations=50,
    max_time_seconds=7200.0,
    storage_dir="./experiments",
    verbose=True,
    ollama_config=OllamaConfig(),
    memory_config=MemoryConfig(),    # context/memory knobs (see below)
    use_lock=True,                   # refuse to run if another process holds this name
    experiment_config=None,          # ExperimentConfig; overrides the arguments above
)
```

| Method | Description |
|---|---|
| `run(initial="")` | Start or resume; `initial` is seed content for new experiments |
| `resume()` | Resume only; raises `RuntimeError` if no saved state, `ValueError` if it is corrupt |
| `stop()` | Request graceful stop after current iteration |
| `pause()` | Request pause after current iteration |
| `get_status()` | Dict with `status`, `current_iteration`, `best_score`, `elapsed_seconds` |

A lock file (`experiments/{name}/lock`) prevents two processes writing the same
experiment.  A lock left behind by a dead pid is taken over automatically.

### `MemoryConfig` — bounded prompt growth

```python
MemoryConfig(
    recent_window=10,              # iterations rendered verbatim
    best_window=3,                 # top-scoring iterations outside the recent window
    max_compressed_blocks=10,      # summaries kept before the oldest are merged
    max_findings_per_block=6,      # per summary — this was the runaway growth
    max_tried_items=20,            # size of the "already tried" list
    compression_trigger=0.85,      # fraction of the usable budget that triggers compression
    reserved_response_tokens=1024, # space left for the model's reply (num_predict)
    static_prompt_tokens=600,      # system prompt + fixed prompt templates
)
```

`effective_budget(usable_ctx)` = `usable_ctx - reserved_response_tokens -
static_prompt_tokens`, floored at 512 tokens.  Without this reserve a 4096-token
model "fits" a 2457-token prompt and then errors, because Ollama counts the
intended completion against `num_ctx` too.

### `OllamaConfig`

```python
OllamaConfig(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0.4,
    timeout=600,        # per-call timeout in seconds
    num_ctx=32768,      # model context window
)
```

Values from `ollama_config.json` (see **Prerequisites** for where it is read from)
are coerced to the right types and validated (`num_ctx > 0`, `timeout > 0`,
`0 <= temperature <= 2`), so a `"0.2"` string in the JSON file no longer becomes
a `TypeError` inside a prompt.

## Project layout

```
experimenter/       library (config, parsing, llm, memory, storage, executor, experiment)
tests/              pure-logic tests, no Ollama server needed
example_code.py     runnable code-optimisation demo
example_text.py     runnable text-quality demo
pyproject.toml      packaging + ruff + pytest config
requirements.txt    runtime dependencies
ollama_config.example.json  (in experimenter/) config template — copy to use
```

`experimenter/setup.py` used to live inside the package, which made
`find_packages()` return `[]` when run from inside it; packaging now lives at
the repository root.

## Tests

Pure-logic tests (parsing, storage, memory, executor, logging) run without an
Ollama server:

```bash
python -m unittest discover -t . -s tests   # or: pytest
ruff check experimenter tests
```

CI (`.github/workflows/tests.yml`) runs lint plus the test suite on Python
3.10–3.13.

## License

MIT
