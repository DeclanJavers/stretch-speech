"""The conversation engine: one conversation at a time.

    heard(words)   streaming STT (or a cheat) delivers a few words at a time
    tick()         every ~0.5 s: once the visitor pauses, ask the LLM. It replies with
                   speech and/or a tool call; wait_for_visitor means "they're not done".
                   After a long pause the wait tool is withdrawn, so it must reply.
    respond()      the state machine needs the robot to say something now (e.g. the
                   fetch_item tool result just came in: "found Jeff, hand over the pen")
    say(text)      a fixed line, no LLM (greetings, calling someone's name)

The mic is "muted" while the robot speaks: words heard then are dropped.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Callable, Protocol

from .events import EventLog
from .llm import LLMClient, LLMError, Reply, parse_message, sentences
from .prompt import canned_line, extract_name, render_chat, tool_specs
from .registry import Registry
from .states import RobotState


class Speaker(Protocol):
    def say(self, text: str, person_id: str | None) -> float: ...   # returns the speech's duration (s)
    def done(self, person_id: str | None) -> None: ...             # end of one robot turn


WAIT_NARRATION = re.compile(r"^(i'?ll )?(wait|waiting|keep listening|let(ting)? (them|the visitor) finish)\b")
ASKS_WHICH = ("which", "would you like", "what would you", "what can i get")
WAIT_CUE = re.compile(r"\b(wait|hang on|hold on|be right back|right back|one moment|just a moment|stay)\b", re.I)

# get_context() -> (robot state, a note for the system prompt or None, tool names allowed now)
Context = Callable[[], tuple[RobotState, "str | None", list[str]]]


class Conversation:
    def __init__(self, llm: LLMClient, log: EventLog, registry: Registry, get_context: Context,
                 speaker: Speaker, clock: Callable[[], float] = time.time,
                 llm_log_path: str | pathlib.Path | None = None,
                 min_pause_s: float = 0.5, force_reply_s: float = 2.5, mute_while_speaking: bool = True):
        self.llm, self.log, self.registry = llm, log, registry
        self.get_context, self.speaker, self.clock = get_context, speaker, clock
        self.llm_log = pathlib.Path(llm_log_path) if llm_log_path else None
        self.min_pause_s = min_pause_s          # don't ask the LLM until the visitor pauses this long
        self.force_reply_s = force_reply_s      # after this long a pause, the LLM must reply (no wait tool)
        self.mute_while_speaking = mute_while_speaking   # real mic: yes (don't hear yourself); typed input: no
        self.pid: str | None = None
        self.muted_until = 0.0
        self.last_spoke_end = 0.0               # when the robot's last line finishes playing (estimate)
        self.last_spoke_start = 0.0             # when it started: words after this answer it
        self.last_sig = None
        self.failures = 0

    # ---- lifecycle ---------------------------------------------------------------
    def start(self, pid: str) -> None:
        self.pid = pid
        self.last_spoke_end = self.last_spoke_start = self.clock()
        self.last_sig = None
        self.failures = 0

    def stop(self) -> None:
        self.pid = None

    @property
    def speaking(self) -> bool:
        return self.clock() < self.muted_until

    # ---- input -------------------------------------------------------------------
    def heard(self, text: str, t: float | None = None) -> bool:
        """Returns False if the words were dropped (no conversation, or robot speaking)."""
        t = self.clock() if t is None else t
        if self.pid is None or not text.strip() or (self.mute_while_speaking and t < self.muted_until):
            return False
        self.log.append("heard", self.pid, t=t, text=text.strip())
        return True

    def _heard_since_robot(self):
        # With the mic muted, nothing arrives while the robot speaks anyway; with typed input
        # (no muting), a line typed during the robot's speech is still a reply to it.
        return [e for e in self.log.snapshot()
                if e.kind == "heard" and e.person_id == self.pid and e.t >= self.last_spoke_start]

    def unanswered_for(self) -> float:
        """Seconds since the robot finished speaking with no reply at all (0 if they replied)."""
        if self.pid is None or self.speaking or self._heard_since_robot():
            return 0.0
        return self.clock() - self.last_spoke_end

    # ---- turns -------------------------------------------------------------------
    def tick(self) -> Reply | None:
        if self.pid is None or self.speaking:
            return None
        heard = self._heard_since_robot()
        if not heard:
            return None
        pause = self.clock() - heard[-1].t
        if pause < self.min_pause_s:
            return None
        forced = pause >= self.force_reply_s
        sig = (len(heard), forced)
        if sig == self.last_sig:
            return None                      # nothing new since the last call
        self.last_sig = sig
        return self._call(allow_wait=not forced)

    def respond(self) -> Reply | None:
        return self._call(allow_wait=False) if self.pid else None

    def say(self, text: str, aside: bool = False) -> None:
        """A fixed line (no LLM), logged like any robot turn. aside=True: part of an ongoing action
        (calling someone's name during a search), shown to the LLM inside that action's result."""
        if self.pid is None:
            return
        now = self.clock()
        self._speak(text)
        self.log.append("said", self.pid, t=now, text=text, **({"aside": True} if aside else {}))

    def _speak(self, text: str) -> None:
        self.last_spoke_start = self.clock()
        for sentence in sentences(text):
            duration = self.speaker.say(sentence, self.pid)
            start = max(self.clock(), self.muted_until)
            self.muted_until = start + duration
        self.last_spoke_end = max(self.muted_until, self.clock())
        self.speaker.done(self.pid)

    def _call(self, allow_wait: bool) -> Reply | None:
        pid = self.pid
        robot, note, names = self.get_context()
        names = [n for n in names if allow_wait or n != "wait_for_visitor"]
        now = self.clock()
        events = self.log.snapshot()
        messages = render_chat(events, self.registry, robot, pid, now, note, names)
        tools = tool_specs(names, robot.items_in_stock())
        t0 = time.perf_counter()
        try:
            msg = self.llm.chat(messages, tools or None, require_tool=bool(tools))
        except LLMError as e:
            self.failures += 1
            self.log.append("note", pid, text=f"LLM error: {e}")
            return None
        self.failures = 0
        reply = parse_message(msg, [t["function"]["name"] for t in tools])
        self._validate(reply, robot, events)
        self._learn_name(events)

        if reply.tool != "wait_for_visitor":
            if not reply.speech and reply.tool:
                reply.speech = canned_line(reply.tool, reply.args)
            elif reply.tool == "fetch_item" and not WAIT_CUE.search(reply.speech):
                # The model sometimes says a handover line ("Here is your pen!") as it leaves.
                self.log.append("note", self.pid, text=f"fetch line had no 'wait': {reply.speech!r}")
                reply.speech = canned_line(reply.tool, reply.args)
            if reply.speech:
                self._speak(reply.speech)
            if reply.speech or reply.tool:
                data = {"text": reply.speech}
                if reply.tool:
                    data.update(tool=reply.tool, args=reply.args, call_id=reply.call_id)
                self.log.append("said", pid, t=now, **data)

        if self.llm_log:
            with self.llm_log.open("a") as f:
                f.write(json.dumps({"t": now, "person": pid, "allow_wait": allow_wait, "messages": messages,
                                    "tools": [t["function"]["name"] for t in tools], "output": msg,
                                    "reply": {"speech": reply.speech, "tool": reply.tool, "args": reply.args},
                                    "ms": round((time.perf_counter() - t0) * 1000)}) + "\n")
        return reply

    def _validate(self, reply: Reply, robot: RobotState, events) -> None:
        """Code has the last word on tool calls (each check targets a mistake measured in eval/run_eval.py)."""
        said = reply.speech.lower()
        if reply.tool is None and WAIT_NARRATION.match(said):
            # The model wrote its wait out as text ("Wait for visitor to finish...") instead of calling the tool.
            reply.tool, reply.speech = "wait_for_visitor", ""
            return
        if reply.tool == "fetch_item":
            reason = None
            if reply.args.get("item") not in robot.items_in_stock():
                reason = "item not in stock"
            elif "?" in said and (any(k in said for k in ASKS_WHICH)
                                  or sum(w in said for w in ("pen", "shirt", "plushie")) >= 2):
                reason = "its own words still ask which item"
            if reason:
                self.log.append("note", self.pid, text=f"rejected fetch_item {reply.args}: {reason}")
                reply.tool, reply.args = None, {}          # keep the speech, drop the action

    def _learn_name(self, events) -> None:
        person = self.registry.get(self.pid)
        if not person or person.name:
            return
        mine = [e for e in events if e.person_id == self.pid and e.kind in ("heard", "said")]
        asked, words = False, []
        for e in mine + [None]:
            if e is None or e.kind == "said":
                name = extract_name(" ".join(words), asked) if words else None
                if name:
                    person.name = name
                    self.log.append("person", self.pid, change="name", name=name)
                    return
                words = []
                asked = e is not None and "name" in e.data.get("text", "").lower() and "?" in e.data.get("text", "")
            else:
                words.append(e.data["text"])
