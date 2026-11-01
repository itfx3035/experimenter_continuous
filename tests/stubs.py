"""Minimal stand-ins for the langchain imports.

`experimenter.llm` imports langchain at module load.  The tests here only
exercise pure logic, so these stubs let the package import without a server.
"""

import sys
import types

try:
    import langchain_core  # noqa: F401
except ImportError:
    core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content=""):
            self.content = content

    messages.HumanMessage = _Message
    messages.SystemMessage = _Message
    core.messages = messages
    sys.modules["langchain_core"] = core
    sys.modules["langchain_core.messages"] = messages

try:
    import langchain_ollama  # noqa: F401
except ImportError:
    ollama = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            raise RuntimeError("stub LLM — no server")

    ollama.ChatOllama = ChatOllama
    sys.modules["langchain_ollama"] = ollama
