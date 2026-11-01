import json
import os
from dataclasses import dataclass, field

# Safety margin for context window usage.
# We use 60% of the model's num_ctx for context, leaving 40% for the response.
CONTEXT_SAFETY_MARGIN = 0.6
# Start compressing when we hit this fraction of the budget.
COMPRESSION_TRIGGER = 0.75
# Characters per token.  Prose is ~4 chars/token, but Python code tokenises
# denser (~3), so 3 is used as the conservative estimate everywhere.
TOKEN_CHAR_FACTOR = 3

# Fallback token budget (for llama3.1 with num_ctx=32768, 60% ≈ 19660).  Used
# only when no OllamaConfig-derived budget is supplied to MemoryManager.
CONTEXT_TOKEN_BUDGET = 20_000

# Tokens left unused by the context: room for the model's reply plus the fixed
# prompt framing (system prompt, section headers).  Without this the budget is
# fully consumed by the context and Ollama rejects the call.
RESERVED_RESPONSE_TOKENS = 1024
STATIC_PROMPT_TOKENS = 600

# Minimum usable budget — below this, compression is pointless and prompts
# would be truncated to nothing.
MIN_EFFECTIVE_BUDGET = 512


@dataclass
class MemoryConfig:
    """Windowing / budget knobs for MemoryManager (all previously hardcoded)."""

    recent_window: int = 5              # iterations rendered verbatim under "Recent"
    best_window: int = 3                # top-scoring iterations rendered verbatim
    max_compressed_blocks: int = 10     # rolling-summary cap (stops unbounded growth)
    max_findings_per_block: int = 6
    max_tried_items: int = 20           # cap on the "already tried" list
    compression_trigger: float = COMPRESSION_TRIGGER
    reserved_response_tokens: int = RESERVED_RESPONSE_TOKENS
    static_prompt_tokens: int = STATIC_PROMPT_TOKENS

    def effective_budget(self, total_budget: int) -> int:
        """Budget actually available for the rendered context."""
        return max(
            MIN_EFFECTIVE_BUDGET,
            int(total_budget) - self.reserved_response_tokens - self.static_prompt_tokens,
        )

    def __post_init__(self) -> None:
        if self.recent_window <= 0:
            raise ValueError("recent_window must be > 0")
        if self.best_window <= 0:
            raise ValueError("best_window must be > 0")
        if self.max_compressed_blocks <= 0:
            raise ValueError("max_compressed_blocks must be > 0")
        if not 0 < self.compression_trigger <= 1:
            raise ValueError("compression_trigger must be in (0, 1]")
        if self.reserved_response_tokens < 0 or self.static_prompt_tokens < 0:
            raise ValueError("reserved/static token counts must be >= 0")


def load_ollama_config_from_json(config_path: str | None = None) -> dict:
    """
    Load Ollama configuration from a JSON file.

    Lookup order when `config_path` is not given:
      1. `./ollama_config.json` in the current working directory
      2. `ollama_config.json` inside the `experimenter` package
      3. the built-in defaults below

    The repository ships `experimenter/ollama_config.example.json` as a template;
    copy it to one of the two paths above (or pass `config_path`) to customise.
    Missing keys are filled from the defaults, and a missing/invalid file falls
    back to the defaults instead of crashing.
    """
    if config_path is None:
        for candidate in (
            os.path.join(os.getcwd(), "ollama_config.json"),
            os.path.join(os.path.dirname(__file__), "ollama_config.json"),
        ):
            if os.path.exists(candidate):
                config_path = candidate
                break

    default_config = {
        "base_url": "http://localhost:11434",
        "model": "llama3.1",
        "temperature": 0.4,
        "timeout": 600,
        "num_ctx": 32768,
    }

    if not config_path:
        return default_config

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            if not isinstance(config, dict):
                return default_config
            # Ensure all required keys are present
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default config if file doesn't exist or is invalid
        return default_config


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    temperature: float = 0.4
    # Per-call HTTP timeout (seconds).  LLM threads are daemon threads so a
    # hung call will not block process exit.
    timeout: int = 600
    # Context window passed to the model.
    num_ctx: int = 32768

    @classmethod
    def from_json(cls, config_path: str | None = None) -> "OllamaConfig":
        """
        Create an OllamaConfig instance from a JSON configuration file.
        Falls back to default values if the file doesn't exist or is invalid.
        """
        config = load_ollama_config_from_json(config_path)
        return cls(
            base_url=str(config["base_url"]),
            model=str(config["model"]),
            temperature=float(config["temperature"]),
            timeout=int(config["timeout"]),
            num_ctx=int(config["num_ctx"]),
        )

    def get_context_budget(self) -> int:
        """Return the token budget for context, with safety margin applied."""
        return int(self.num_ctx * CONTEXT_SAFETY_MARGIN)

    def __post_init__(self) -> None:
        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be > 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")


@dataclass
class ExecutorConfig:
    """Sandbox settings for running generated code."""
    timeout: int = 600
    memory_mb: int = 512
    cpu_seconds: int = 120
    isolated: bool = True        # python -I: ignore PYTHONPATH / user site-packages
    scrub_env: bool = True       # do not inherit PYTHONSTARTUP, PYTHONPATH, etc.

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("ExecutorConfig.timeout must be > 0")
        if self.memory_mb <= 0:
            raise ValueError("ExecutorConfig.memory_mb must be > 0")
        if self.cpu_seconds <= 0:
            raise ValueError("ExecutorConfig.cpu_seconds must be > 0")


@dataclass
class ExperimentConfig:
    name: str
    prompt: str
    requires_execution: bool
    max_iterations: int = 50
    max_time_seconds: float = 7200.0
    storage_dir: str = "./experiments"
    verbose: bool = True
    ollama_config: OllamaConfig = field(default_factory=OllamaConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    # Iterations without a best-score improvement of at least this much are
    # reported to the LLM as stagnation (it is asked to change strategy).
    stagnation_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be > 0")
        if self.stagnation_threshold < 0:
            raise ValueError("stagnation_threshold must be >= 0")

    def context_budget(self) -> int:
        """Token budget derived from the model's real context window."""
        return self.ollama_config.get_context_budget()

