"""The robot's state machine, as data.

Every arrow in MASTER-PLAN.md's state diagram is one (state, trigger) -> state entry
in TRANSITIONS. Code fires triggers; next_state() is the only way to change state,
so an arrow that isn't in the table can't happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    LOOKING = "looking_for_new_person"
    TALKING = "talking_to_person"
    TO_TABLE = "walking_to_table"
    PICKING_UP = "picking_up_item"
    TO_PERSON = "going_to_person"
    SEARCHING = "searching_for_person"
    GIVING = "giving_item"
    RETURNING_ITEM = "returning_item"
    DONE = "done"


class Trigger(str, Enum):
    NEW_PERSON = "new_person_found"              # perception: unknown face in range
    REQUEST_ITEM = "request_item"                # LLM
    DECLINE = "decline"                          # LLM
    END = "end"                                  # LLM (conversation over, no choice)
    NO_RESPONSE = "no_response"                  # timer: visitor never answered
    ARRIVED = "arrived"                          # navigation skill
    PICKED_UP = "picked_up"                      # grasp skill
    PICKUP_GAVE_UP = "pickup_gave_up"            # grasp skill hit its retry cap
    TARGET_FOUND = "target_found"                # perception: verify() matched, holding item
    TARGET_FOUND_NO_ITEM = "target_found_no_item"  # same, but the pickup had failed
    TARGET_NOT_FOUND = "target_not_found"        # timer, holding item
    TARGET_NOT_FOUND_NO_ITEM = "target_not_found_no_item"
    ITEM_GIVEN = "item_given"                    # handover skill
    NO_NEW_PEOPLE = "no_new_people"              # timer
    OUT_OF_STOCK = "out_of_stock"                # stock check


TRANSITIONS: dict[tuple[State, Trigger], State] = {
    (State.LOOKING, Trigger.NEW_PERSON): State.TALKING,
    (State.LOOKING, Trigger.NO_NEW_PEOPLE): State.DONE,
    (State.LOOKING, Trigger.OUT_OF_STOCK): State.DONE,

    (State.TALKING, Trigger.REQUEST_ITEM): State.TO_TABLE,
    (State.TALKING, Trigger.DECLINE): State.LOOKING,
    (State.TALKING, Trigger.END): State.LOOKING,
    (State.TALKING, Trigger.NO_RESPONSE): State.LOOKING,

    (State.TO_TABLE, Trigger.ARRIVED): State.PICKING_UP,

    (State.PICKING_UP, Trigger.PICKED_UP): State.TO_PERSON,
    (State.PICKING_UP, Trigger.PICKUP_GAVE_UP): State.TO_PERSON,   # go back and apologize

    (State.TO_PERSON, Trigger.ARRIVED): State.SEARCHING,

    (State.SEARCHING, Trigger.TARGET_FOUND): State.GIVING,
    (State.SEARCHING, Trigger.TARGET_FOUND_NO_ITEM): State.TALKING,   # offer something else
    (State.SEARCHING, Trigger.TARGET_NOT_FOUND): State.RETURNING_ITEM,
    (State.SEARCHING, Trigger.TARGET_NOT_FOUND_NO_ITEM): State.LOOKING,

    (State.GIVING, Trigger.ITEM_GIVEN): State.LOOKING,

    (State.RETURNING_ITEM, Trigger.ARRIVED): State.LOOKING,

    (State.DONE, Trigger.NEW_PERSON): State.TALKING,
}

# Triggers the LLM may fire, per state. Everything else comes from code.
LLM_TRIGGERS: dict[State, list[Trigger]] = {
    State.TALKING: [Trigger.REQUEST_ITEM, Trigger.DECLINE, Trigger.END],
}

# How each state reads in the LLM's transcript.
DESCRIPTION: dict[State, str] = {
    State.LOOKING: "looking for a new person",
    State.TALKING: "talking with {person}",
    State.TO_TABLE: "walking back to the table",
    State.PICKING_UP: "picking up the {item}",
    State.TO_PERSON: "going back to where {person} was standing",
    State.SEARCHING: "looking around for {person}",
    State.GIVING: "giving the {item} to {person}",
    State.RETURNING_ITEM: "putting the {item} back on the table",
    State.DONE: "nobody new around, waiting at the table",
}

# Typical time a state takes (s). Steps taking over 1.5x this are flagged as slow.
EXPECTED_S: dict[State, float] = {
    State.TO_TABLE: 20, State.PICKING_UP: 15, State.TO_PERSON: 20, State.SEARCHING: 15,
}


class InvalidTransition(ValueError):
    pass


def next_state(state: State, trigger: Trigger) -> State:
    try:
        return TRANSITIONS[(state, trigger)]
    except KeyError:
        raise InvalidTransition(f"no transition from {state.value} on {trigger.value}") from None


@dataclass
class RobotState:
    state: State = State.LOOKING
    since: float = 0.0                       # when the current state started
    target: str | None = None                # person id the robot is serving
    held_item: str | None = None
    stock: dict[str, int] = field(default_factory=dict)

    def items_in_stock(self) -> list[str]:
        return [item for item, n in self.stock.items() if n > 0]

    def llm_triggers(self) -> list[Trigger]:
        triggers = list(LLM_TRIGGERS.get(self.state, []))
        if Trigger.REQUEST_ITEM in triggers and not self.items_in_stock():
            triggers.remove(Trigger.REQUEST_ITEM)
        return triggers
