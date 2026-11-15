"""Minimal client for a local model served by Ollama (chat with tools + embeddings).

Tool calling is done in the prompt (Hermes/Qwen `<tool_call>` format) and parsed here,
instead of using Ollama's built-in tool parser: that parser is strict JSON and silently
drops a call when the model writes multi-line SQL (raw newlines inside a JSON string),
returning an empty message. `json.loads(strict=False)` accepts those newlines.
"""

from __future__ import annotations

import json
import re

import httpx
import numpy as np

from airquality.config import OLLAMA_EMBED_MODEL, OLLAMA_MODEL, OLLAMA_URL

TIMEOUT = httpx.Timeout(300, connect=5)  # a 7B model on a mid-range GPU can take a while
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOLS_PREAMBLE = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def parse_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Split model output into (remaining text, tool calls). Tolerates newlines in strings."""
    calls = []
    for block in TOOL_CALL.findall(content):
        try:
            obj = json.loads(block, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"function": {"name": obj["name"],
                                       "arguments": obj.get("arguments") or {}}})
    return TOOL_CALL.sub("", content).strip(), calls


def with_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Append the tool signatures to the system message; drop structured tool_calls (the
    assistant's raw <tool_call> text is kept in its content, so the model sees it)."""
    spec = TOOLS_PREAMBLE.format(tools="\n".join(json.dumps(t, ensure_ascii=False)
                                                 for t in tools))
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k != "tool_calls"}
        if m["role"] == "system" and tools:
            m["content"] += spec
        out.append(m)
    return out


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, embed_model: str = OLLAMA_EMBED_MODEL,
                 url: str = OLLAMA_URL):
        self.model, self.embed_model = model, embed_model
        self.http = httpx.Client(base_url=url, timeout=TIMEOUT)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One chat turn. Returns the assistant message; `tool_calls` holds parsed calls and
        `content` keeps the raw text (incl. <tool_call> blocks) for the conversation history.
        """
        r = self.http.post("/api/chat", json={
            "model": self.model, "messages": with_tools(messages, tools or []),
            "stream": False,
            # Deterministic, and a context window large enough for schema + tool results.
            "options": {"temperature": 0, "num_ctx": 8192},
        })
        r.raise_for_status()
        raw = r.json()["message"].get("content", "")
        text, calls = parse_tool_calls(raw)
        return {"role": "assistant", "content": raw, "text": text, "tool_calls": calls}

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.http.post("/api/embed", json={"model": self.embed_model, "input": texts})
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype="float32")
