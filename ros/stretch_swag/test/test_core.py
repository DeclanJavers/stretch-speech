"""Tests for the ROS-free core: states, chat rendering, reply parsing, the conversation engine,
and the Mac face service's matching. Run:  python3 -m pytest ros/stretch_swag/test -q"""

import importlib.util
import pathlib

import pytest

from stretch_swag.core.conversation import Conversation
from stretch_swag.core.events import EventLog
from stretch_swag.core.llm import parse_message
from stretch_swag.core.prompt import extract_name, render_chat, system_prompt, tool_specs
from stretch_swag.core.registry import Registry
from stretch_swag.core.states import InvalidTransition, RobotState, State, Trigger, TRANSITIONS, next_state

TALK_TOOLS = ["say_to_visitor", "fetch_item", "wait_for_visitor", "end_conversation"]


# ---- states ---------------------------------------------------------------------
def test_every_state_has_a_way_out():
    assert {s for s, _ in TRANSITIONS} == set(State)


def test_invalid_transition_raises():
    assert next_state(State.TALKING, Trigger.REQUEST_ITEM) == State.TO_TABLE
    with pytest.raises(InvalidTransition):
        next_state(State.TO_TABLE, Trigger.REQUEST_ITEM)


# ---- parsing --------------------------------------------------------------------
def _msg(content=None, tool=None, args="{}"):
    m = {"content": content}
    if tool:
        m["tool_calls"] = [{"id": "c1", "function": {"name": tool, "arguments": args}}]
    return m


def test_parse_message():
    r = parse_message(_msg(None, "fetch_item", '{"item": "pen", "say": "A pen, wait here!"}'), TALK_TOOLS)
    assert (r.speech, r.tool, r.args, r.call_id) == ("A pen, wait here!", "fetch_item", {"item": "pen"}, "c1")
    r = parse_message(_msg(None, "say_to_visitor", '{"text": "Hi Jeff!"}'), TALK_TOOLS)
    assert (r.speech, r.tool) == ("Hi Jeff!", None)                       # plain speech
    assert parse_message(_msg("Waiting for them.", "wait_for_visitor"), TALK_TOOLS).speech == ""
    assert parse_message(_msg("Hi!", "fetch_item", '{"item": "pen"}'), ["say_to_visitor"]).tool is None
    assert parse_message(_msg("Hi!", "fetch_item", "not json"), TALK_TOOLS).args == {}


def test_extract_name():
    assert extract_name("Hey, I'm Sam. What do you have?", False) == "Sam"
    assert extract_name("my name is maria", False) == "Maria"
    assert extract_name("Jordan. What are you giving away?", True) == "Jordan"
    assert extract_name("Jordan. What are you giving away?", False) is None   # not asked: could be anything
    assert extract_name("I'm good, thanks", False) is None
    assert extract_name("I'm just looking", False) is None


def test_tools_follow_stock_and_prompt_follows_tools():
    names = [t["function"]["name"] for t in tool_specs(TALK_TOOLS, [])]
    assert "fetch_item" not in names
    spec = tool_specs(["fetch_item"], ["pen", "plushie"])[0]["function"]
    assert spec["parameters"]["properties"]["item"]["enum"] == ["pen", "plushie"]
    reg = Registry()
    p = reg.new(0)
    robot = RobotState(stock={"pen": 1, "shirt": 0, "plushie": 2})
    full = system_prompt(robot, p, None, TALK_TOOLS)
    assert "pens and plushies" in full and "t-shirt" not in full.split("give away")[0]
    assert "fetch_item" not in system_prompt(robot, p, "Hand them the pen.", ["say_to_visitor"])


# ---- chat rendering ---------------------------------------------------------------
def _setup():
    log, reg = EventLog(), Registry()
    jeff = reg.new(t=100.0)
    robot = RobotState(state=State.TALKING, target=jeff.id, stock={"pen": 2, "shirt": 0, "plushie": 1})
    return log, reg, robot, jeff


def test_fetch_renders_as_a_tool_call_and_its_result():
    log, reg, robot, jeff = _setup()
    log.append("state", jeff.id, t=100.0, state="talking_to_person")
    log.append("said", jeff.id, t=100.5, text="Hi! What's your name?")
    log.append("heard", jeff.id, t=104.0, text="I'm")
    log.append("heard", jeff.id, t=104.3, text="Jeff,")
    log.append("heard", jeff.id, t=106.0, text="pen please")        # 1.7 s pause before this
    log.append("said", jeff.id, t=107.0, text="Wait here, Jeff!", tool="fetch_item", args={"item": "pen"},
               call_id="c1")
    jeff.name = "Jeff"
    log.append("state", jeff.id, t=121.0, state="walking_to_table", item="pen")
    log.append("state", jeff.id, t=128.0, state="picking_up_item", item="pen")
    log.append("skill", jeff.id, t=140.0, skill="pickup", success=False)
    log.append("skill", jeff.id, t=150.0, skill="pickup", success=True)
    log.append("state", jeff.id, t=180.0, state="going_to_person", item="pen")
    log.append("state", jeff.id, t=191.0, state="searching_for_person", item="pen")
    log.append("said", jeff.id, t=191.2, text="Jeff? I have your pen!", aside=True)   # call-out mid-search
    log.append("state", jeff.id, t=192.0, state="giving_item", item="pen")
    robot.state, robot.held_item = State.GIVING, "pen"
    msgs = render_chat(log.snapshot(), reg, robot, jeff.id, now=192.5, note="Hand them the pen.", tool_names=[])

    assert [m["role"] for m in msgs] == ["system", "assistant", "tool", "user", "assistant", "tool"]
    assert "You're talking with Jeff. You're holding the pen" in msgs[0]["content"]
    greet = msgs[1]["tool_calls"][0]["function"]                  # robot speech is a say_to_visitor call
    assert greet["name"] == "say_to_visitor" and "What's your name?" in greet["arguments"]
    assert msgs[2]["content"] == "Said." and msgs[3]["content"] == "I'm Jeff,... pen please"
    call = msgs[4]["tool_calls"][0]
    assert call["id"] == msgs[5]["tool_call_id"] == "c1" and call["function"]["name"] == "fetch_item"
    assert '"say": "Wait here, Jeff!"' in call["function"]["arguments"]
    result = msgs[5]["content"]
    assert "You walked to your table (7 seconds)." in result
    assert "You picked up the pen after 2 tries (52 seconds, slower than usual)." in result
    assert 'You called out: "Jeff? I have your pen!".' in result      # the call-out doesn't end the fetch
    assert "found Jeff, who is right in front of you now" in result


def test_other_people_never_appear():
    log, reg, robot, jeff = _setup()
    maria = reg.new(t=100.0)
    log.append("said", jeff.id, t=101.0, text="Hi Jeff")
    log.append("said", maria.id, t=111.0, text="Hi Maria")
    text = str(render_chat(log.snapshot(), reg, robot, jeff.id, now=120.0))
    assert "Hi Jeff" in text and "Maria" not in text


# ---- conversation engine ----------------------------------------------------------
class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def chat(self, messages, tools=None, require_tool=False):
        self.calls.append((messages, [t["function"]["name"] for t in tools or []]))
        return self.outputs.pop(0)


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, pid):
        self.said.append(text)
        return 1.0

    def done(self, pid):
        self.said.append("<done>")


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def _conversation(outputs):
    log, reg, robot, jeff = _setup()
    clock, speaker, llm = Clock(), FakeSpeaker(), FakeLLM(outputs)
    conv = Conversation(llm, log, reg, lambda: (robot, None, TALK_TOOLS), speaker, clock=clock)
    conv.start(jeff.id)
    return conv, clock, speaker, llm, log, reg, jeff


def test_ticks_wait_for_a_pause_and_only_call_when_something_changed():
    conv, clock, speaker, llm, *_ = _conversation([_msg(None, "wait_for_visitor"), _msg(None, "wait_for_visitor")])
    conv.heard("I'd like the")
    clock.t += 0.3
    assert conv.tick() is None and not llm.calls              # still mid-pause
    clock.t += 0.3
    assert conv.tick().tool == "wait_for_visitor"
    assert "wait_for_visitor" in llm.calls[0][1]
    clock.t += 0.5
    assert conv.tick() is None and len(llm.calls) == 1        # nothing new
    clock.t += 2.0                                            # long pause: must reply now
    conv.tick()
    assert "wait_for_visitor" not in llm.calls[1][1]


def test_fetch_is_spoken_logged_and_mutes_the_mic():
    conv, clock, speaker, llm, log, reg, jeff = _conversation(
        [_msg(None, "fetch_item", '{"item": "plushie", "say": "A plushie! Please wait here."}')])
    conv.heard("I'm Jeff, plushie please")
    clock.t += 1.0
    reply = conv.tick()
    assert reply.tool == "fetch_item" and reply.args == {"item": "plushie"}
    assert reg.people[jeff.id].name == "Jeff"                 # learned from their words, not the model
    assert speaker.said == ["A plushie!", "Please wait here.", "<done>"]
    assert log.last("said").data["call_id"] == reply.call_id
    assert conv.speaking and not conv.heard("ok")
    clock.t += 2.5
    assert conv.heard("thanks")


def test_code_checks_the_models_tool_calls():
    cases = [
        (_msg(None, "fetch_item", '{"item": "pen", "say": "Which would you like?"}'), None),   # still asking
        (_msg(None, "fetch_item", '{"item": "shirt", "say": "A t-shirt! Wait here."}'), None),  # out of stock
        (_msg("Wait for the visitor to finish their sentence."), "wait_for_visitor"),           # narrated wait
        (_msg(None, "fetch_item", '{"item": "pen"}'), "fetch_item"),
    ]
    for out, want in cases:
        conv, clock, speaker, llm, log, *_ = _conversation([out])
        conv.heard("hmm")
        clock.t += 1.0
        reply = conv.tick()
        assert reply.tool == want, out
    assert reply.speech.startswith("Great choice!")           # fixed line when the model said nothing
    assert llm.calls[0][1] == TALK_TOOLS


def test_typed_input_during_robot_speech_counts():
    log, reg, robot, jeff = _setup()
    clock, speaker = Clock(), FakeSpeaker()
    llm = FakeLLM([_msg(None, "say_to_visitor", '{"text": "Hi Alex!"}')])
    conv = Conversation(llm, log, reg, lambda: (robot, None, TALK_TOOLS), speaker, clock=clock,
                        mute_while_speaking=False)
    conv.start(jeff.id)
    conv.say("Hello! What's your name?")              # "speaks" for ~2 s
    clock.t += 0.5
    assert conv.heard("I'm Alex.")                    # typed while the robot is still talking
    clock.t += 3.0
    assert conv.tick().speech == "Hi Alex!"


def test_unanswered():
    conv, clock, *_ = _conversation([])
    conv.say("Hi there!")
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
    assert reg.identify(view("a"))["person_id"] is None
    reg.register("p001", view("a"))
    reg.register("p002", view("b"))
    assert reg.identify(view("a"))["person_id"] == "p001"
    assert reg.identify(view("b"))["person_id"] == "p002"
    assert reg.verify("p001", view("a"))["match"] and not reg.verify("p001", view("b"))["match"]
    assert reg.identify(fs.unit([rng.gauss(0, 1) for _ in range(128)]))["person_id"] is None


# ---- geometry (lidar safety, badge positions) -------------------------------------
def test_scan_obstacle_cone_and_offset():
    import math
    from stretch_swag.core.geometry import scan_obstacle
    ranges = [5.0] * 360                       # 1 deg per beam, starting at -180 deg
    inc, amin = math.radians(1), -math.pi
    assert scan_obstacle(ranges, amin, inc, 0.6) is None
    ranges[180] = 0.5                          # straight ahead (0 deg)
    assert scan_obstacle(ranges, amin, inc, 0.6) == 0.5
    assert scan_obstacle(ranges, amin, inc, 0.6, yaw_offset=math.pi) is None   # lidar mounted backwards
    ranges[180], ranges[90] = 5.0, 0.4         # at -90 deg: beside the robot, outside the cone
    assert scan_obstacle(ranges, amin, inc, 0.6) is None
    ranges[90] = 0.05                          # too close to be real (own body)
    assert scan_obstacle(ranges, amin, inc, 0.6) is None


def test_badges_and_transforms():
    import math
    from stretch_swag.core.geometry import badge_embedding, base_to_world, transform_point
    a, b = badge_embedding(100), badge_embedding(101)
    assert sum(x * y for x, y in zip(a, a)) == 1 and sum(x * y for x, y in zip(a, b)) == 0
    # camera frame -> base: rotate 90 deg about z, then shift
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    x, y, z = transform_point((1.0, 0.0, 1.2), q, (1.0, 0.0, 0.0))
    assert (round(x, 6), round(y, 6), round(z, 6)) == (1.0, 1.0, 1.2)
    wx, wy = base_to_world((2.0, 1.0, math.pi / 2), 1.0, 0.0)  # robot facing +y: 1 m ahead is (2, 2)
    assert (round(wx, 6), round(wy, 6)) == (2.0, 2.0)
