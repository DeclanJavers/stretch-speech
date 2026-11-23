"""The conversation engine: one conversation at a time, driven by ticks and events.

    heard(words)      streaming STT (or a cheat) delivers a few words at a time
    tick()            every ~0.5 s: if something changed, ask the LLM; it replies
                      <listening> (nothing happens) or <speaking> ... (spoken, logged)
    event_turn(why)   the state machine needs the robot to say something now

The mic is "muted" while the robot speaks: words heard then are dropped.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Callable, Protocol

from .events import EventLog
from .llm import LLMClient, LLMError, Reply, SentenceSplitter, build_grammar, parse
from .prompt import render_messages
from .registry import Registry
from .states import RobotState


class Speaker(Protocol):
    def say(self, text: str, person_id: str | None) -> float: ...   # returns the speech's duration (s)
    def done(self, person_id: str | None) -> None: ...             # end of one robot turn


class Conversation:
    def __init__(self, llm: LLMClient, log: EventLog, registry: Registry,
                 get_robot: Callable[[], RobotState], speaker: Speaker,
                 clock: Callable[[], float] = time.time,
                 llm_log_path: str | pathlib.Path | None = None,
                 milestones: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
                 first_prompt_s: float = 4.0):
        self.llm, self.log, self.registry = llm, log, registry
        self.get_robot, self.speaker, self.clock = get_robot, speaker, clock
        self.llm_log = pathlib.Path(llm_log_path) if llm_log_path else None
        self.milestones = milestones
        self.first_prompt_s = first_prompt_s   # robot just spoke and heard nothing: wait this long before asking the LLM
        self.pid: str | None = None
        self.muted_until = 0.0
        self.last_spoke_end = 0.0
        self.started = 0.0
        self.last_sig = None
        self.failures = 0

    # ---- lifecycle ---------------------------------------------------------------
    def start(self, pid: str) -> None:
        self.pid = pid
        self.started = self.last_spoke_end = self.clock()
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
        if self.pid is None or t < self.muted_until or not text.strip():
            return False
        self.log.append("heard", self.pid, t=t, text=text.strip())
        return True

    def _heard_since_robot(self):
        return [e for e in self.log.snapshot()
                if e.kind == "heard" and e.person_id == self.pid and e.t >= self.last_spoke_end]

    def silence(self) -> float:
        heard = self._heard_since_robot()
        last = heard[-1].t if heard else self.last_spoke_end
        return max(0.0, self.clock() - last)

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
        silence = self.silence()
        if not heard and silence < self.first_prompt_s:
            return None                      # they haven't started replying yet
        sig = (len(heard), sum(1 for m in self.milestones if silence >= m))
        if sig == self.last_sig:
            return None                      # nothing new since the last call
        self.last_sig = sig
        return self._call("tick", None, silence)

    def event_turn(self, reason: str) -> Reply | None:
        if self.pid is None:
            return None
        return self._call("event", reason, None)

    def _say(self, text: str) -> None:
        duration = self.speaker.say(text, self.pid)
        start = max(self.clock(), self.muted_until)
        self.muted_until = start + duration
        self.last_spoke_end = self.muted_until

    def _call(self, mode: str, reason: str | None, silence: float | None) -> Reply | None:
        pid = self.pid
        robot = self.get_robot()
        now = self.clock()
        messages = render_messages(self.log.snapshot(), self.registry, robot, pid, now, mode, reason, silence)
        grammar = build_grammar(mode == "tick", robot.items_in_stock(), robot.llm_triggers())
        splitter, text, spoken = SentenceSplitter(), "", []
        t0 = time.perf_counter()
        first_ms = None
        stream = self.llm.stream(messages, grammar)
        try:
            for delta in stream:
                if first_ms is None:
                    first_ms = round((time.perf_counter() - t0) * 1000)
                text += delta
                if text.lstrip().startswith("<listening"):
                    break                    # decided: stop generating
                for sentence in splitter.feed(delta):
                    self._say(sentence)
                    spoken.append(sentence)
        except LLMError as e:
            self.failures += 1
            self.log.append("note", pid, text=f"LLM error: {e}")
            return None
        finally:
            stream.close()
        self.failures = 0
        reply = parse(text)

        if reply.kind == "speaking":
            rest = splitter.flush() if spoken or text.startswith("<speaking>") else [reply.speech]
            for sentence in rest:
                self._say(sentence)
            reply.speech = " ".join(spoken + rest)
            self.speaker.done(pid)
            self.log.append("said", pid, t=now, text=reply.speech, raw=reply.canonical())
            if reply.name:
                person = self.registry.get(pid)
                if person and person.name != reply.name:
                    person.name = reply.name
                    self.log.append("person", pid, change="name", name=reply.name)

        if self.llm_log:
            with self.llm_log.open("a") as f:
                f.write(json.dumps({"t": now, "person": pid, "mode": mode, "reason": reason,
                                    "messages": messages, "output": text, "reply": reply.canonical(),
                                    "first_token_ms": first_ms,
                                    "total_ms": round((time.perf_counter() - t0) * 1000)}) + "\n")
        return reply
