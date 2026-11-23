"""Builds the LLM's messages from the event log (see MASTER-PLAN.md, "LLM harness").

Layout, least-changing first, so llama.cpp can reuse its work on the unchanged start:
    system prompt                       fixed
    person snapshot                     first user message; changes rarely
    transcript (state lines, speech)    only grows
    current situation + allowed moves   rebuilt on every call, always last

Robot lines are assistant turns in exactly the format the model outputs. Visitor
speech, state changes and skill results are user turns; consecutive ones merge.
"""

from __future__ import annotations

from .events import Event
from .registry import Registry
from .states import DESCRIPTION, EXPECTED_S, RobotState, State, Trigger

SYSTEM_PROMPT = """You are Stretch, a cheerful robot giving away free swag at an event, one item per visitor. \
Your words are spoken aloud. The visitor's words come from speech recognition and may have errors \
("tea shirt" is t-shirt, "plush e" is plushie).

The transcript shows what you heard and did. "…(1.2s)" marks a pause in the visitor's speech. \
The last line gives the current situation and what you may do.

Reply in exactly one form:
<listening>
<speaking> what you say <name="Jeff"> <transition ...>

- <listening>: the visitor is mid-sentence or still thinking. Say nothing.
- <speaking>: one or two short sentences, under 25 words, no emojis. Speak once they finish a thought or after a long pause.
- <name="..."> only when the visitor just told you their first name. Never guess. Never use it as your own name.
- <transition request_item item="pen">: they clearly chose one item that is in stock. Confirm it and ask them to wait right here while you fetch it.
- <transition decline>: they turned you down. <transition end>: the conversation is over without a choice.
- A question about an item is not a choice: answer it and ask which they want.
- Ask for their name if you don't know it."""

ITEM_WORDS = {"pen": "pen", "shirt": "t-shirt", "plushie": "plushie"}
PAUSE_MARK_S = 0.5      # gaps between heard words at least this long get a "…(0.8s)" mark
SLOW_FACTOR = 1.5


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(round(max(seconds, 0))), 60)
    return f"+{m}:{s:02d}"


def describe(state: State, person_label: str, item: str | None) -> str:
    return DESCRIPTION[state].format(person=person_label, item=ITEM_WORDS.get(item, item or "item"))


def transitions_text(robot: RobotState) -> str:
    parts = []
    for trig in robot.llm_triggers():
        if trig == Trigger.REQUEST_ITEM:
            parts.append(f"request_item item={'|'.join(robot.items_in_stock())}")
        else:
            parts.append(trig.value)
    return ", ".join(parts) if parts else "none"


def render_messages(events: list[Event], registry: Registry, robot: RobotState, pid: str, now: float,
                    mode: str = "tick", reason: str | None = None,
                    silence_s: float | None = None) -> list[dict]:
    """mode "tick": the model may reply <listening>. mode "event": it must speak (reason says why)."""
    person = registry.people[pid]
    label = person.label()
    mine = [e for e in events if e.person_id == pid and e.kind in ("state", "heard", "said", "skill")]
    t0 = mine[0].t if mine else person.first_seen

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_lines: list[str] = [f"[{registry.snapshot_text(pid)}]"]
    words: list[tuple[float, str]] = []           # the visitor's current, still-open line
    others: set[str] = set()                      # other people served since this person's last event
    away_since: float | None = None
    last_state: Event | None = None
    failures: dict[str, int] = {}

    def flush_words():
        if not words:
            return
        text, prev = "", None
        for t, chunk in words:
            if prev is not None:
                gap = t - prev
                text += f" …({gap:.1f}s) " if gap >= PAUSE_MARK_S else " "
            text += chunk
            prev = t
        user_lines.append(f"[visitor: {text}]")
        words.clear()

    def flush_user():
        flush_words()
        messages.append({"role": "user", "content": "\n".join(user_lines) or "[no reply from visitor]"})
        user_lines.clear()

    for e in events:
        if e.person_id != pid or e.kind not in ("state", "heard", "said", "skill"):
            # Time spent serving other people collapses into one summary line.
            if e.kind == "state" and e.person_id and e.t >= t0:
                if not others:
                    away_since = e.t
                others.add(e.person_id)
            continue

        if others:
            flush_words()
            n = len(others)
            minutes = max(1, round((e.t - away_since) / 60))
            user_lines.append(f"[served {n} other visitor{'s' if n > 1 else ''}, about {minutes} min]")
            last_state = None
            others.clear()

        if e.kind == "heard":
            words.append((e.t, e.data["text"]))
        elif e.kind == "said":
            flush_user()
            messages.append({"role": "assistant", "content": e.data["raw"]})
        elif e.kind == "state":
            flush_words()
            state = State(e.data["state"])
            line = f"[state: {describe(state, label, e.data.get('item'))}  {fmt_elapsed(e.t - t0)}"
            if last_state is not None:
                prev = State(last_state.data["state"])
                if prev in EXPECTED_S:
                    took = e.t - last_state.t
                    line += f", {describe(prev, label, last_state.data.get('item'))} took {took:.0f}s"
                    if took > SLOW_FACTOR * EXPECTED_S[prev]:
                        line += ", slower than usual"
            user_lines.append(line + "]")
            last_state = e
            failures.clear()
        elif e.kind == "skill" and not e.data.get("success"):
            flush_words()
            skill = e.data.get("skill", "skill")
            failures[skill] = failures.get(skill, 0) + 1
            n = failures[skill]
            line = f"[{skill} failed]" if n == 1 else f"[{skill} failed {n} times]"
            if n > 1 and user_lines and user_lines[-1].startswith(f"[{skill} failed"):
                user_lines[-1] = line
            else:
                user_lines.append(line)

    flush_words()

    # The current situation: rebuilt every call, always last.
    status = (f"[now {fmt_elapsed(now - t0)} | {describe(robot.state, label, robot.held_item or person.item)}"
              f" | holding: {ITEM_WORDS.get(robot.held_item, robot.held_item) or 'nothing'}"
              f" | in stock: {', '.join(f'{ITEM_WORDS.get(i, i)} {n}' for i, n in robot.stock.items() if n > 0) or 'nothing'}")
    if silence_s is not None:
        status += f" | visitor silent for {silence_s:.1f}s"
    status += "]"
    if mode == "event":
        rule = f"[you must speak now: {reason}. Allowed transitions: {transitions_text(robot)}]"
    else:
        rule = f"[reply <listening> or <speaking>. Allowed transitions: {transitions_text(robot)}]"
    user_lines.extend([status, rule])
    flush_user()
    return messages
