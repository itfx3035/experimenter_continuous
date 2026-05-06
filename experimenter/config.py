from dataclasses import dataclass, field
import json
import os
from typing import Optional

# Safety margin for context window usage.
# We use 60% of the model's num_ctx for context, leaving 40% for the response.
CONTEXT_SAFETY_MARGIN = 0.6
# Start compressing when we hit this fraction of the budget.
COMPRESSION_TRIGGER = 0.75

# Default token budget (for llama3.1 with num_ctx=32768, 60% ≈ 19660)
CONTEXT_TOKEN_BUDGET = 20_000

def load_ollama_config_from_json(config_path: Optional[str] = None) -> dict:
    """
    Load Ollama configuration from JSON file.
    If config_path is None, looks for ollama_config.json in the experimenter directory.
    Returns default values if the file doesn't exist.
    """
    if config_path is None:
        # Default path in the experimenter directory
        config_path = os.path.join(os.path.dirname(__file__), "ollama_config.json")

    default_config = {
        "base_url": "http://localhost:11434",
        "model": "llama3.1",
        "temperature": 0.4,
        "timeout": 600,
        "num_ctx": 32768
    }

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
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
    def from_json(cls, config_path: Optional[str] = None) -> "OllamaConfig":
        """
        Create an OllamaConfig instance from a JSON configuration file.
        Falls back to default values if the file doesn't exist or is invalid.
        """
        config = load_ollama_config_from_json(config_path)
        return cls(
            base_url=config["base_url"],
            model=config["model"],
            temperature=config["temperature"],
            timeout=config["timeout"],
            num_ctx=config["num_ctx"]
        )

    def get_context_budget(self) -> int:
        """Return the token budget for context, with safety margin applied."""
        return int(self.num_ctx * CONTEXT_SAFETY_MARGIN)


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

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be > 0")
