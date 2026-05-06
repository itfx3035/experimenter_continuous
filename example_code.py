"""
Example: optimize a Fibonacci implementation for speed.

Run:
    python example_code.py

The experiment auto-saves after every iteration.  Kill it at any time with
Ctrl-C; re-running the script will resume from where it left off.
"""

from experimenter import Experiment, OllamaConfig

exp = Experiment(
    name="fast_fibonacci",
    prompt=(
        "Optimize the fibonacci(n) function for execution speed. "
        "The function must return the correct Fibonacci number for any n in 0..40. "
        "Measure speed by calling fibonacci(35) repeatedly."
    ),
    requires_execution=True,
    max_iterations=20,
    max_time_seconds=3600,
    ollama_config=OllamaConfig(
        base_url="http://localhost:11434",
        model="hf.co/unsloth/gpt-oss-20b-GGUF:UD-Q6_K_XL",
        temperature=0.4,
    ),
    verbose=True,
)

result = exp.run()

print("\n=== Final result ===")
print(f"Status:         {result['status']}")
print(f"Best score:     {result['best_score']:.2f}")
print(f"Best iteration: {result['best_iteration']}")
print(f"Total iters:    {result['total_iterations']}")
print(f"\nBest implementation:\n{result['best_result']}")
