"""Talking to the LLM: any OpenAI-compatible /v1/chat/completions endpoint.

Standard library only (runs in ROS's system Python). Replies are streamed, so the
robot can start speaking the first sentence while the rest is generated.

The reply format (see prompt.SYSTEM_PROMPT):
    <listening>
    <speaking> Great, a pen it is! <name="Jeff"> <transition request_item item="pen">

llama.cpp's server accepts a `grammar` field that forces this format token by token.
Other servers ignore it, so parse() also copes with slightly-off output.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from .states import Trigger


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 20.0,
                 temperature: float = 0.3, max_tokens: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(self, messages: list[dict], grammar: str | None = None) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_prompt": True,                                   # llama.cpp: reuse the unchanged prefix
            "chat_template_kwargs": {"enable_thinking": False},     # llama.cpp: no thinking
            "reasoning_effort": "none",                             # LM Studio: no thinking
        }
        if grammar:
            body["grammar"] = grammar
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer none"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    chunk = json.loads(payload)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            yield delta["content"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            raise LLMError(f"LLM request to {self.base_url} failed: {e}") from e

    def health(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=5) as r:
                return [m["id"] for m in json.loads(r.read()).get("data", [])]
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            raise LLMError(f"can't reach LLM at {self.base_url}: {e}") from e


def build_grammar(can_listen: bool, items: list[str], triggers: list[Trigger]) -> str:
    """GBNF grammar for llama.cpp: the reply can only be one of the allowed forms."""
    trans = []
    for trig in triggers:
        if trig == Trigger.REQUEST_ITEM and items:
            trans.append('"request_item item=\\"" item "\\""')
        elif trig != Trigger.REQUEST_ITEM:
            trans.append(f'"{trig.value}"')
    rules = [
        "root ::= " + ('"<listening>" | speak' if can_listen else "speak"),
        'speak ::= "<speaking> " text name?' + (" transition?" if trans else ""),
        "text ::= [^<\\n]+",
        'name ::= " <name=\\"" [A-Za-z .\'-]+ "\\">"',
    ]
    if trans:
        rules.append('transition ::= " <transition " (' + " | ".join(trans) + ') ">"')
        if items:
            rules.append("item ::= " + " | ".join(f'"{i}"' for i in items))
    return "\n".join(rules) + "\n"


@dataclass
class Reply:
    kind: str                      # "listening" | "speaking"
    speech: str = ""
    name: str | None = None
    trigger: Trigger | None = None
    item: str | None = None

    def canonical(self) -> str:
        """The exact text stored as the robot's turn (and shown back to the model)."""
        if self.kind == "listening":
            return "<listening>"
        out = f"<speaking> {self.speech}"
        if self.name:
            out += f' <name="{self.name}">'
        if self.trigger == Trigger.REQUEST_ITEM:
            out += f' <transition request_item item="{self.item}">'
        elif self.trigger:
            out += f" <transition {self.trigger.value}>"
        return out


_NAME = re.compile(r'<name="([^"]{1,40})">')
_TRANS = re.compile(r'<transition\s+(\w+)(?:\s+item="(\w+)")?\s*>')


def parse(text: str) -> Reply:
    text = text.strip()
    if text.startswith("<listening>") or not text:
        return Reply("listening")
    body = text[len("<speaking>"):] if text.startswith("<speaking>") else text
    speech = body.split("<", 1)[0].strip()
    if not speech:
        return Reply("listening")
    name = _NAME.search(body)
    trans = _TRANS.search(body)
    trigger, item = None, None
    if trans:
        try:
            trigger = Trigger(trans.group(1))
        except ValueError:
            trigger = None
        item = trans.group(2)
    return Reply("speaking", speech=speech, name=name.group(1).strip() if name else None,
                 trigger=trigger, item=item)


class SentenceSplitter:
    """Feed streamed text; get back whole sentences of the spoken part as they complete."""

    _END = re.compile(r"(.+?[.!?])(\s+|$)")

    def __init__(self):
        self.text = ""
        self.sent = 0           # characters of speech already emitted
        self.closed = False     # reached the first tag after the speech

    def _speech(self) -> str:
        if not self.text.startswith("<speaking>"):
            return ""          # <listening>, or the tag hasn't fully arrived yet
        body = self.text[len("<speaking>"):]
        if "<" in body:
            self.closed = True
            body = body.split("<", 1)[0]
        return body

    def feed(self, delta: str) -> list[str]:
        self.text += delta
        speech = self._speech()
        out = []
        while True:
            rest = speech[self.sent:]
            m = self._END.match(rest)
            if not m or (not m.group(2) and not self.closed):
                break
            sentence = m.group(1).strip()
            if sentence:
                out.append(sentence)
            self.sent += m.end()
        return out

    def flush(self) -> list[str]:
        speech = self._speech()
        rest = speech[self.sent:].strip()
        self.sent = len(speech)
        return [rest] if rest else []
