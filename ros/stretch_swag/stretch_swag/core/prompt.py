"""Builds the LLM's messages from the event log, framed as an ordinary tool-using chat.

Small chat models are trained on: a system prompt, user messages, assistant replies,
and tool calls with tool results. So the robot's world is mapped onto exactly that:

    visitor speech              -> user message (plain words; "..." where they paused)
    robot speech                -> assistant message
    robot decisions             -> tool calls: wait_for_visitor, fetch_item, end_conversation
    what happened during a fetch -> the fetch_item tool result (walked to the table,
                                   picking up took 52 s, found the visitor again, ...)
    current situation           -> the system prompt, in plain sentences

Nothing here mentions states, transcripts or tags.
"""

from __future__ import annotations

import json
import re

from .events import Event
from .registry import Person, Registry
from .states import EXPECTED_S, RobotState, State

ITEM_WORDS = {"pen": "pen", "shirt": "t-shirt", "plushie": "plushie"}
ITEM_PLURAL = {"pen": "pens", "shirt": "t-shirts", "plushie": "plushies"}
PAUSE_MARK_S = 0.5      # a gap this long between heard words shows as "..."
SLOW_FACTOR = 1.5

GREETINGS = [
    "Hi there! I'm Stretch, and I'm giving away free swag today. What's your name?",
    "Hello! I'm Stretch, the swag robot. What's your name?",
    "Hey! I'm Stretch, and I've got free goodies to give away. What should I call you?",
]

PERSONA = """You are Stretch, a friendly robot at a tech event giving away free swag, one item per person. \
You talk out loud with visitors, so keep every reply to one or two short sentences, with no emojis or lists."""

# Variant D in eval: every robot action is a tool and the model must call exactly one per turn.
# Free-form replies + optional tools scored 64%: the model attached fetch_item to almost every
# reply ("Which item would you like?" + fetch_item). With a say_to_visitor tool it scored 79%.
SPEECH_NOTE = ("The visitor's words come from speech recognition. They can arrive before the visitor has finished "
               "speaking, and can contain errors (\"tea shirt\" means t-shirt).")
RULES = {
    "say_to_visitor": "- say_to_visitor: to answer or ask something. Use this when they tell you their name, say hi, "
                      "or ask a question.",
    "fetch_item": "- fetch_item: only when they have clearly chosen one item. Its \"say\" tells them to wait right "
                  "here while you get it.",
    "wait_for_visitor": "- wait_for_visitor: their message is cut off mid-sentence; say nothing.",
    "end_conversation": "- end_conversation: they don't want anything; its \"say\" is your goodbye.",
}
TOOL_ORDER = ["say_to_visitor", "fetch_item", "wait_for_visitor", "end_conversation"]

# Name extraction stays in code: as a tool argument the model filled it with "Visitor", "a", etc.
_NAME_PATTERNS = [r"\b(?:i'?m|i am|my name is|my name's|name's|it'?s|call me|this is)\s+([A-Za-z][a-z'-]+)"]
NOT_NAMES = {"stretch", "visitor", "pen", "pens", "shirt", "plushie", "please", "thanks", "hi", "hello", "hey", "good",
             "fine", "here", "just", "looking", "not", "sure", "okay", "ok", "yes", "no", "so", "a", "the", "all",
             "going", "great", "interested", "done", "set", "alright", "well", "really", "actually", "kind", "also"}


def extract_name(text: str, asked_name: bool) -> str | None:
    """A first name the visitor said: "I'm Jeff", "my name is Maria", or just "Sam." after being asked."""
    for pattern in _NAME_PATTERNS:
        m = re.search(pattern, text, re.I)
        if m and m.group(1).lower() not in NOT_NAMES:
            return m.group(1).capitalize()
    first = re.match(r"^\s*([A-Z][a-z'-]+)\s*[.!,]", text)
    if asked_name and first and first.group(1).lower() not in NOT_NAMES:
        return first.group(1)
    return None


def spoken_list(words: list[str]) -> str:
    if not words:
        return "nothing"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


# ---- tools ------------------------------------------------------------------------
def tool_specs(names: list[str], items: list[str]) -> list[dict]:
    specs = {
        "say_to_visitor": {
            "description": "Say something to the visitor.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "wait_for_visitor": {
            "description": "The visitor hasn't finished their sentence yet. Say nothing and keep listening.",
            "parameters": {"type": "object", "properties": {}},
        },
        "fetch_item": {
            "description": "Go to your table and bring the visitor the item they chose.",
            "parameters": {"type": "object", "properties": {
                "item": {"type": "string", "enum": items},
                "say": {"type": "string", "description": "What you tell the visitor before you go."},
            }, "required": ["item", "say"]},
        },
        "end_conversation": {
            "description": "The visitor doesn't want anything.",
            "parameters": {"type": "object", "properties": {
                "say": {"type": "string", "description": "Your goodbye."}}, "required": ["say"]},
        },
    }
    if not items:
        names = [n for n in names if n != "fetch_item"]
    return [{"type": "function", "function": {"name": n, **specs[n]}} for n in TOOL_ORDER if n in names]


# What the robot says if the model calls a tool without saying anything.
def canned_line(tool: str, args: dict) -> str:
    if tool == "fetch_item":
        return f"Great choice! Please wait right here while I get your {ITEM_WORDS.get(args.get('item'), 'item')}."
    if tool == "end_conversation":
        return "No problem, enjoy the event!"
    return ""


# ---- system prompt ----------------------------------------------------------------
def situation(robot: RobotState, person: Person, note: str | None) -> str:
    stock = [ITEM_PLURAL.get(i, i) for i in robot.items_in_stock()]
    lines = []
    if person.name:
        lines.append(f"You're talking with {person.name}.")
    else:
        lines.append("A visitor is standing in front of you.")
    if robot.held_item:
        lines.append(f"You're holding the {ITEM_WORDS.get(robot.held_item, robot.held_item)} they asked for.")
    lines.append(f"You have {spoken_list(stock)} to give away." if stock else "You've run out of swag.")
    if note:
        lines.append(note)
    return " ".join(lines)


def system_prompt(robot: RobotState, person: Person, note: str | None, tool_names: list[str]) -> str:
    parts = [PERSONA, situation(robot, person, note)]
    rules = [RULES[n] for n in TOOL_ORDER if n in tool_names]
    if rules:
        heard = SPEECH_NOTE + "\n" if ("wait_for_visitor" in tool_names or "fetch_item" in tool_names) else ""
        parts.append(heard + "Every turn, call exactly one tool:\n" + "\n".join(rules))
    return "\n\n".join(parts)


# ---- tool results: what happened during a fetch ---------------------------------------
def _secs(s: float) -> str:
    return f"{s:.0f} seconds" if s < 90 else f"{s / 60:.0f} minutes"


def fetch_result(call: Event, events: list[Event], now: float, person: Person) -> str:
    """Narrate the state/skill events after a fetch_item call, as the tool's result."""
    item = ITEM_WORDS.get(call.data.get("args", {}).get("item"), "item")
    who = person.name or "the visitor"
    states = [e for e in events if e.kind == "state"]
    if not states:
        return f"Fetching the {item}."
    out = []
    for i, e in enumerate(states):
        s = State(e.data["state"])
        end = states[i + 1].t if i + 1 < len(states) else now
        dur = end - e.t
        slow = s in EXPECTED_S and dur > SLOW_FACTOR * EXPECTED_S[s]
        took = f" ({_secs(dur)}{', slower than usual' if slow else ''})"
        nxt = State(states[i + 1].data["state"]) if i + 1 < len(states) else None
        if s == State.TO_TABLE:
            out.append(f"You walked to your table{took}.")
        elif s == State.PICKING_UP:
            tries = [x for x in events if x.kind == "skill" and x.data.get("skill") == "pickup"
                     and e.t <= x.t <= end]
            fails = sum(1 for x in tries if not x.data.get("success"))
            if any(x.data.get("success") for x in tries):
                out.append(f"You picked up the {item}" + (f" after {fails + 1} tries" if fails else "") + f"{took}.")
            elif nxt is not None:
                out.append(f"You couldn't pick up the {item}: the grasp failed {fails} times{took}.")
            else:
                out.append(f"You're picking up the {item}.")
        elif s == State.TO_PERSON:
            out.append(f"You went back to where {who} was standing{took}.")
        elif s == State.SEARCHING:
            if nxt in (State.GIVING, State.TALKING):
                out.append(f"You looked around and found {who}" + (f" after {_secs(dur)}" if dur > 3 else "")
                           + ", who is right in front of you now.")
            elif nxt is not None:
                out.append(f"You looked around for {_secs(dur)} but couldn't find {who}.")
        elif s == State.RETURNING_ITEM:
            out.append(f"You put the {item} back on your table.")
        elif s in (State.GIVING, State.TALKING) and i > 0:
            prev = State(states[i - 1].data["state"])
            if prev != State.SEARCHING:                    # a missed visitor turned up again later
                out.append(f"{_secs(e.t - states[i - 1].t).capitalize()} later, {who} is in front of you again.")
    return " ".join(out)


# ---- messages -----------------------------------------------------------------------
def join_words(words: list[tuple[float, str]]) -> str:
    text, prev = "", None
    for t, chunk in words:
        if prev is not None:
            text += "... " if t - prev >= PAUSE_MARK_S else " "
        text += chunk
        prev = t
    return text


def render_chat(events: list[Event], registry: Registry, robot: RobotState, pid: str, now: float,
                note: str | None = None, tool_names: list[str] | None = None) -> list[dict]:
    """OpenAI-style messages for the conversation with one person."""
    tool_names = tool_names or []
    person = registry.people[pid]
    messages: list[dict] = [{"role": "system", "content": system_prompt(robot, person, note, tool_names)}]
    words: list[tuple[float, str]] = []
    open_call: tuple[Event, list[Event]] | None = None

    def flush_words():
        if words:
            messages.append({"role": "user", "content": join_words(words)})
            words.clear()

    def close_call():
        nonlocal open_call
        if open_call:
            call, during = open_call
            tool = call.data["tool"]
            result = fetch_result(call, during, now, person) if tool == "fetch_item" else "The conversation is over."
            messages.append({"role": "tool", "tool_call_id": call.data["call_id"], "content": result})
            open_call = None

    for e in events:
        if e.person_id != pid:
            continue
        if e.kind == "heard":
            close_call()
            words.append((e.t, e.data["text"]))
        elif e.kind == "said":
            close_call()
            flush_words()
            tool = e.data.get("tool")
            if tool:                                   # an action: its "say" is what the robot said
                args = {**(e.data.get("args") or {}), "say": e.data.get("text", "")}
                call_id = e.data["call_id"]
            else:                                      # plain speech
                tool, args, call_id = "say_to_visitor", {"text": e.data.get("text", "")}, f"say_{int(e.t * 1000)}"
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]})
            if tool == "say_to_visitor":
                messages.append({"role": "tool", "tool_call_id": call_id, "content": "Said."})
            else:
                open_call = (e, [])
        elif e.kind in ("state", "skill") and open_call:
            open_call[1].append(e)
    close_call()
    flush_words()
    return messages
