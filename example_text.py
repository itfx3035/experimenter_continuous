"""
Example: essay.

Run:
    python example_text.py

The experiment auto-saves after every iteration.  Kill it at any time with
Ctrl-C; re-running the script will resume from where it left off.

Change `model` to a model you actually have pulled — `ollama pull llama3.1`,
then `ollama list` to see the available names.
"""

from experimenter import Experiment, OllamaConfig

exp = Experiment(
    name="essay",
    prompt=(
        "Write a short essay about war consequences on economy."
        "Measure quality of text as how clear economic effect is explained."
    ),
    requires_execution=False,
    max_iterations=20,
    max_time_seconds=3600,
    ollama_config=OllamaConfig(
        base_url="http://localhost:11434",
        model="llama3.1",          # ← change this to a model you have pulled
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
