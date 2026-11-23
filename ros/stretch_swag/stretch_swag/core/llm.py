"""Talking to the LLM: any OpenAI-compatible /v1/chat/completions endpoint, with tools.

Standard library only (runs in ROS's system Python). llama-server needs `--jinja`
for tool calls: it then formats tools with the model's own chat template and parses
the model's tool calls back into the standard `tool_calls` field.
"""

from __future__ import annotations

import itertools
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 20.0,
                 temperature: float = 0.3, max_tokens: int = 150):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], tools: list[dict] | None = None, require_tool: bool = False) -> dict:
        """Returns the assistant message: {"content": str | None, "tool_calls": [...]}."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_prompt": True,                                   # llama.cpp: reuse the unchanged prefix
            "chat_template_kwargs": {"enable_thinking": False},     # llama.cpp: no thinking
            "reasoning_effort": "none",                             # LM Studio: no thinking
        }
        if tools:
            body["tools"] = tools
            if require_tool:
                body["tool_choice"] = "required"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer none"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError,
                KeyError, IndexError) as e:
            raise LLMError(f"LLM request to {self.base_url} failed: {e}") from e

    def health(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=5) as r:
                data = json.loads(r.read())
            return [m.get("id") or m.get("name") for m in data.get("data") or data.get("models") or []]
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            raise LLMError(f"can't reach LLM at {self.base_url}: {e}") from e


_ids = itertools.count(1)


@dataclass
class Reply:
    speech: str = ""                     # what to say out loud ("" = nothing)
    tool: str | None = None              # the tool the model called, if any (first allowed one)
    args: dict = field(default_factory=dict)
    call_id: str | None = None


_JUNK = re.compile(r"<\|?/?(tool_call|think)\|?>.*", re.S)   # stray template markup in the text


def parse_message(msg: dict, allowed: list[str]) -> Reply:
    reply = Reply(speech=_JUNK.sub("", msg.get("content") or "").strip())
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") in allowed:
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            reply.tool, reply.args = fn["name"], args if isinstance(args, dict) else {}
            reply.call_id = call.get("id") or f"call_{next(_ids)}"
            break
    if reply.tool == "say_to_visitor":
        reply.speech, reply.tool = str(reply.args.get("text") or reply.speech).strip(), None   # plain speech
    elif reply.tool in ("fetch_item", "end_conversation"):
        reply.speech = str(reply.args.pop("say", "") or reply.speech).strip()
    elif reply.tool == "wait_for_visitor":
        reply.speech = ""                # models narrate the wait ("Waiting for the visitor..."): never say it
    return reply


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
