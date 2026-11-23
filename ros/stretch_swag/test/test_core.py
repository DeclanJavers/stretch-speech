"""Tests for the ROS-free core: states, prompt rendering, reply parsing, the conversation engine,
and the Mac face service's matching. Run:  python3 -m pytest ros/stretch_swag/test -q"""

import importlib.util
import pathlib

import pytest

from stretch_swag.core.conversation import Conversation
from stretch_swag.core.events import EventLog
from stretch_swag.core.llm import Reply, SentenceSplitter, build_grammar, parse
from stretch_swag.core.prompt import render_messages
from stretch_swag.core.registry import Registry
from stretch_swag.core.states import (InvalidTransition, RobotState, State, Trigger, TRANSITIONS,
                                      next_state)


# ---- states ---------------------------------------------------------------------
def test_every_state_has_a_way_out():
    sources = {s for s, _ in TRANSITIONS}
    assert sources == set(State)


def test_invalid_transition_raises():
    assert next_state(State.TALKING, Trigger.REQUEST_ITEM) == State.TO_TABLE
    with pytest.raises(InvalidTransition):
        next_state(State.TO_TABLE, Trigger.REQUEST_ITEM)


def test_request_item_not_offered_when_out_of_stock():
    r = RobotState(state=State.TALKING, stock={"pen": 0})
    assert Trigger.REQUEST_ITEM not in r.llm_triggers()
    assert RobotState(state=State.GIVING).llm_triggers() == []


# ---- reply parsing / grammar ------------------------------------------------------
def test_parse_forms():
    assert parse("<listening>").kind == "listening"
    r = parse('<speaking> A pen, great! Wait here. <name="Jeff"> <transition request_item item="pen">')
    assert (r.kind, r.speech, r.name, r.trigger, r.item) == \
        ("speaking", "A pen, great! Wait here.", "Jeff", Trigger.REQUEST_ITEM, "pen")
    assert r.canonical() == '<speaking> A pen, great! Wait here. <name="Jeff"> <transition request_item item="pen">'
    assert parse("Hello there!").speech == "Hello there!"          # server ignored the grammar
    assert parse("<speaking> Bye. <transition decline>").trigger == Trigger.DECLINE


def test_grammar_lists_only_allowed_moves():
    g = build_grammar(True, ["pen", "plushie"], [Trigger.REQUEST_ITEM, Trigger.DECLINE])
    assert '"<listening>" | speak' in g and '"pen" | "plushie"' in g and '"decline"' in g
    assert '"end"' not in g
    g = build_grammar(False, [], [])
    assert "listening" not in g and "transition" not in g


def test_sentence_splitter_streams_sentences():
    s = SentenceSplitter()
    out = []
    for chunk in ["<speak", "ing> Hi there", "! I'm Stre", "tch. What's your", " name?", ' <name="x">']:
        out += s.feed(chunk)
    out += s.flush()
    assert out == ["Hi there!", "I'm Stretch.", "What's your name?"]
    s = SentenceSplitter()
    assert s.feed("<listening>") == [] and s.flush() == []


# ---- prompt rendering -------------------------------------------------------------
def _setup():
    log, reg = EventLog(), Registry()
    jeff = reg.new(t=100.0)
    robot = RobotState(state=State.TALKING, target=jeff.id, stock={"pen": 2, "shirt": 0, "plushie": 1})
    return log, reg, robot, jeff


def test_transcript_example():
    log, reg, robot, jeff = _setup()
    log.append("state", jeff.id, t=100.0, state="talking_to_person", item=None)
    log.append("said", jeff.id, t=101.0, text="Hi! What's your name?", raw="<speaking> Hi! What's your name?")
    log.append("heard", jeff.id, t=104.0, text="I'm")
    log.append("heard", jeff.id, t=104.3, text="Jeff")
    log.append("heard", jeff.id, t=106.0, text="and")          # 1.7 s pause before this
    jeff.name = "Jeff"
    log.append("said", jeff.id, t=107.0, text="A pen?", raw='<speaking> A pen? <name="Jeff">')
    log.append("state", jeff.id, t=121.0, state="walking_to_table", item="pen")
    log.append("state", jeff.id, t=133.0, state="picking_up_item", item="pen")
    log.append("skill", jeff.id, t=140.0, skill="pickup", success=False)
    log.append("skill", jeff.id, t=150.0, skill="pickup", success=False)
    log.append("state", jeff.id, t=185.0, state="going_to_person", item="pen")
    robot.state, robot.held_item = State.TO_PERSON, "pen"
    msgs = render_messages(log.snapshot(), reg, robot, jeff.id, now=190.0, mode="event", reason="say hi")

    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"].startswith("[Visitor p001; name: Jeff")
    assert "[visitor: I'm Jeff …(1.7s) and]" in msgs[3]["content"]
    assert msgs[4]["content"] == '<speaking> A pen? <name="Jeff">'
    last = msgs[5]["content"]
    assert "[state: walking back to the table  +0:21]" in last
    assert "walking back to the table took 12s" in last
    assert "[pickup failed 2 times]" in last
    assert "picking up the pen took 52s, slower than usual" in last
    assert last.splitlines()[-1].startswith("[you must speak now: say hi")
    assert "in stock: pen 2, plushie 1" in last and "t-shirt" not in last.split("in stock:")[1]


def test_other_visitors_collapse():
    log, reg, robot, jeff = _setup()
    maria = reg.new(t=100.0)
    log.append("state", jeff.id, t=100.0, state="talking_to_person")
    log.append("state", maria.id, t=110.0, state="talking_to_person")
    log.append("said", maria.id, t=111.0, text="Hi Maria", raw="<speaking> Hi Maria")
    log.append("state", maria.id, t=200.0, state="giving_item", item="plushie")
    log.append("state", jeff.id, t=300.0, state="searching_for_person", item="pen")
    msgs = render_messages(log.snapshot(), reg, robot, jeff.id, now=301.0)
    text = "\n".join(m["content"] for m in msgs)
    assert "[served 1 other visitor, about 3 min]" in text
    assert "Hi Maria" not in text


# ---- conversation engine ----------------------------------------------------------
class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def stream(self, messages, grammar=None):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        for i in range(0, len(out), 7):
            yield out[i:i + 7]


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, pid):
        self.said.append(text)
        return 1.0

    def done(self, pid):
        self.said.append("<done>")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _conversation(outputs):
    log, reg, robot, jeff = _setup()
    clock, speaker, llm = Clock(), FakeSpeaker(), FakeLLM(outputs)
    conv = Conversation(llm, log, reg, lambda: robot, speaker, clock=clock)
    conv.start(jeff.id)
    return conv, clock, speaker, llm, log, reg, jeff


def test_ticks_only_call_llm_when_something_changed():
    conv, clock, speaker, llm, log, reg, jeff = _conversation(["<listening>", "<listening>"])
    clock.t += 1
    assert conv.tick() is None and not llm.calls          # nothing heard, robot just spoke
    conv.heard("I'd")
    clock.t += 0.3
    assert conv.tick().kind == "listening" and len(llm.calls) == 1
    clock.t += 0.1
    assert conv.tick() is None and len(llm.calls) == 1    # same words, same pause milestone
    clock.t += 0.3                                        # pause crosses 0.5 s
    assert conv.tick().kind == "listening" and len(llm.calls) == 2


def test_speaking_streams_logs_and_mutes():
    out = '<speaking> A plushie! Please wait here. <name="Jeff"> <transition request_item item="plushie">'
    conv, clock, speaker, llm, log, reg, jeff = _conversation([out])
    conv.heard("plushie please")
    clock.t += 1.2
    reply = conv.tick()
    assert reply.trigger == Trigger.REQUEST_ITEM and reply.item == "plushie"
    assert speaker.said == ["A plushie!", "Please wait here.", "<done>"]
    said = log.last("said")
    assert said.data["raw"] == out and reg.people[jeff.id].name == "Jeff"
    assert conv.speaking and not conv.heard("hello")      # mic muted while speaking (2 x 1.0 s)
    clock.t += 2.5
    assert conv.heard("thanks")


def test_unanswered():
    conv, clock, speaker, llm, log, reg, jeff = _conversation(["<speaking> Hi there!"])
    conv.event_turn("greet")
    clock.t += 1.0 + 20
    assert conv.unanswered_for() == pytest.approx(20.0)


# ---- face service (runs on the Mac; pure Python) ---------------------------------
def _face_service():
    path = pathlib.Path(__file__).resolve().parents[3] / "mac-service" / "face_service.py"
    spec = importlib.util.spec_from_file_location("face_service", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_face_registry_matching():
    import random
    fs = _face_service()
    reg = fs.FaceRegistry(threshold=0.45, margin=0.08)
    rng = random.Random(0)
    base = {n: [rng.gauss(0, 1) for _ in range(128)] for n in ("a", "b")}
    view = lambda n, s=0.5: fs.unit([x + rng.gauss(0, s) for x in base[n]])  # noqa: E731
    assert reg.identify(view("a"))["person_id"] is None    # empty registry
    reg.register("p001", view("a"))
    reg.register("p002", view("b"))
    assert reg.identify(view("a"))["person_id"] == "p001"
    assert reg.identify(view("b"))["person_id"] == "p002"
    assert reg.verify("p001", view("a"))["match"] and not reg.verify("p001", view("b"))["match"]
    stranger = fs.unit([rng.gauss(0, 1) for _ in range(128)])
    assert reg.identify(stranger)["person_id"] is None
