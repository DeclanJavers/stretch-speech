"""The event log: every state change, utterance, skill result and person update,
appended once with a timestamp. Transcripts and prompts are built from it.

Kinds:
    state    data: state, prev, trigger, item        person_id = target at the time
    heard    data: text (a few words from streaming STT)
    said     data: text (what the robot spoke), raw (the LLM output, canonical form)
    skill    data: skill, success, reason
    person   data: change, ... (registry updates: registered, name, outcome)
    note     data: text (anything else worth keeping, e.g. rejected LLM actions)
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    t: float
    kind: str
    person_id: str | None = None
    data: dict = field(default_factory=dict)


class EventLog:
    def __init__(self, path: str | pathlib.Path | None = None):
        self.events: list[Event] = []
        self.path = pathlib.Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, person_id: str | None = None, t: float | None = None, **data) -> Event:
        event = Event(t=time.time() if t is None else t, kind=kind, person_id=person_id, data=data)
        with self._lock:
            self.events.append(event)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(asdict(event), default=str) + "\n")
        return event

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self.events)

    def last(self, kind: str, person_id: str | None = None) -> Event | None:
        with self._lock:
            for e in reversed(self.events):
                if e.kind == kind and (person_id is None or e.person_id == person_id):
                    return e
        return None
