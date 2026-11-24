"""The robot-side person registry: who the robot has met and what happened.

Face embeddings live in the Mac's face service under the same person id; this
registry holds everything else. Memory only: nothing about people is written to
disk except through the event log.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# outcome values and how they read in the prompt
OUTCOMES = {
    None: "just met",
    "talking": "talking with you now",
    "waiting": "waiting for you to bring their item",
    "served": "already received their item",
    "declined": "didn't want anything",
    "no_response": "never answered you",
    "ended": "left the conversation without choosing",
    "missed": "you couldn't find them when you came back",
}


@dataclass
class Person:
    id: str
    first_seen: float
    position: tuple[float, float] | None = None   # where they stood (odom/map frame)
    name: str | None = None
    item: str | None = None                       # what they asked for
    outcome: str | None = None
    face_views: int = 0
    fetch_failures: int = 0                       # times the robot came back without their item
    thumb: str | None = None                      # small JPEG (base64) of their face, for the debug view
    thumb_px: int = 0                             # face size it was cut from (keep the sharpest)
    badge: int | None = None                      # ArUco badge id, when perception is badges

    def label(self) -> str:
        return self.name or "the visitor"


class Registry:
    def __init__(self):
        self.people: dict[str, Person] = {}
        self._lock = threading.Lock()

    def new(self, t: float, position: tuple[float, float] | None = None) -> Person:
        with self._lock:
            pid = f"p{len(self.people) + 1:03d}"
            person = Person(id=pid, first_seen=t, position=position)
            self.people[pid] = person
            return person

    def get(self, pid: str | None) -> Person | None:
        return self.people.get(pid) if pid else None

    def snapshot_text(self, pid: str) -> str:
        p = self.people[pid]
        parts = [f"Visitor {p.id}", f"name: {p.name or 'unknown'}"]
        if p.item:
            parts.append(f"asked for: {p.item}")
        parts.append(f"status: {OUTCOMES.get(p.outcome, p.outcome)}")
        return "; ".join(parts)
