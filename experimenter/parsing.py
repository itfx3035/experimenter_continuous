"""
Shared parsing helpers.

Kept in their own module (with no LLM dependencies) so they can be unit-tested
without installing langchain, and so both `llm.py` and `executor.py` use exactly
the same tolerant JSON/code extraction.
"""

from __future__ import annotations

import ast
import json
import re

from .config import TOKEN_CHAR_FACTOR

_FENCE_OPEN_RE = re.compile(r"```(?:python|json)?\s*", re.IGNORECASE)
_FENCE_RE = re.compile(r"```")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_MAIN_GUARD_RE = re.compile(r"^\s*if\s+__name__\s*==\s*[\"']__main__[\"']\s*:")



def strip_fences(text: str) -> str:
    """Remove markdown code fences from a response."""
    text = _FENCE_OPEN_RE.sub("", text)
    return _FENCE_RE.sub("", text)


def extract_json(text: str) -> dict:
    """
    Return the first complete JSON object found in text.
    Handles markdown code fences and leading/trailing prose.

    Raises ValueError if no complete JSON object is present.
    """
    text = strip_fences(text)

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response. Response head: {text[:300]!r}")

    depth = 0
    in_string = False
    escape_next = False
    for idx, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError(f"Unclosed JSON object in LLM response. Text from opening brace: {text[start:start+300]!r}")


def parse_json_output(text: str) -> dict | None:
    """
    Tolerant JSON extraction for program output (harness stdout).

    Scans from the end (the harness prints its result last), trying the single
    line first and then the rest of the buffer so pretty-printed / multi-line
    JSON is handled too.  Returns None instead of raising, and never returns a
    non-dict.
    """
    lines = [line.strip() for line in text.splitlines()]
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].startswith("{"):
            continue
        for candidate in (lines[i], "\n".join(lines[i:])):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data

    # Final fallback: first JSON object anywhere in the buffer.
    try:
        data = extract_json("\n".join(lines))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def extract_code(text: str) -> str:
    """Return code inside the first ```python or ``` fence, or the full text."""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ------------------------------------------------------------------ #
#  Context fitting
# ------------------------------------------------------------------ #

def fit_to_budget(
    text: str,
    budget_tokens: int,
    char_factor: int = TOKEN_CHAR_FACTOR,
) -> str:
    """
    Return `text` shortened to fit `budget_tokens`.

    The caller MUST use the returned value — the old implementation truncated a
    local copy and then sent the untruncated message.  Cuts on a line boundary
    where possible (mid-line cuts are only used when a single line is itself
    over budget), and notes the cut so the model knows the context is partial.
    """
    max_chars = max(1, int(budget_tokens) * char_factor)
    if len(text) <= max_chars:
        return text

    cut = max_chars
    newline = text.rfind("\n", 0, cut)
    if newline > 0:
        cut = newline
    truncated = text[:cut].rstrip()
    return truncated + "\n[context truncated to fit the model's context window]"


# ------------------------------------------------------------------ #
#  __main__ guard detection
# ------------------------------------------------------------------ #

def _is_main_guard(node: ast.AST) -> bool:
    """True for a top-level `if __name__ == "__main__":` statement.

    AST-based so a string literal that merely contains the phrase is not
    mistaken for a guard, and so an indented guard is still detected.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    operands = [test.left] + list(test.comparators)
    has_name = any(isinstance(n, ast.Name) and n.id == "__name__" for n in operands)
    has_literal = any(isinstance(n, ast.Constant) and n.value == "__main__" for n in operands)
    return has_name and has_literal


def has_main_guard(source: str) -> bool:
    """True if the source contains a `if __name__ == "__main__":` block anywhere."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return bool(_MAIN_GUARD_RE.search(source))
    return any(_is_main_guard(node) for node in ast.walk(tree))


def top_level_guards(source: str) -> list[ast.AST]:
    """Top-level guards — the only ones that can be removed safely."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [node for node in tree.body if _is_main_guard(node)]


def strip_main_guard(source: str) -> tuple[bool, str]:
    """Return (removed, cleaned_source) for a top-level `__main__` guard block.

    Only top-level guards are removed: deleting a nested guard can leave its
    parent block with an empty body (a SyntaxError), so those are left alone.
    """
    guards = top_level_guards(source)
    if not guards:
        return False, source

    lines = source.splitlines()
    drop = set()
    for node in guards:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        if start is None:
            continue
        drop.update(range(start, end + 1))

    cleaned = "\n".join(line for i, line in enumerate(lines, 1) if i not in drop).strip()
    return True, cleaned
