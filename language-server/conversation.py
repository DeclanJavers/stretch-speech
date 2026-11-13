"""talk_to_person(): the function the team's state machine calls.

It runs the whole "Talking to new person" state from the diagram and returns
which transition to take next:

    outcome "item"        -> "Person asks for something"  -> Logging name and item
    outcome "declined"    -> "Person doesn't want anything" -> Logging person in memory
    outcome "unclear"     -> chatted MAX_TURNS without choosing (treat like declined)
    outcome "no_response" -> nobody answered (person walked off?)

Example:
    result = talk_to_person(listener, speaker, brain, available_items=["pen", "plushie"])
    if result["outcome"] == "item":
        go_fetch(result["item"], remember=result["name"])
"""

import random
import time
from typing import Callable

import config

GREETINGS = [
    "Hi there! I'm Stretch, and I'm handing out free swag today. What's your name?",
    "Hello! I'm Stretch, the swag robot. What's your name?",
    "Hey! I'm Stretch, and I've got free goodies to give away. What should I call you?",
]


def _spoken_list(items: list[str]) -> str:
    words = [config.ITEM_DESCRIPTIONS[i].split(" (")[0] for i in items]  # drop "(a little stuffed animal)"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + ", or " + words[-1]


def _print_event(kind: str, text: str, info: dict) -> None:
    timing = "  " + "  ".join(f"{k}={v}" for k, v in info.items()) if info else ""
    label = {"robot": "\033[96mSTRETCH\033[0m", "person": "\033[93mPERSON \033[0m",
             "silence": "\033[90m(silence)\033[0m", "error": "\033[91mERROR  \033[0m",
             "decision": "\033[90m  ->   \033[0m"}[kind]
    print(f"{label}  {text}\033[90m{timing}\033[0m", flush=True)


def talk_to_person(listener, speaker, brain, available_items: list[str] | None = None,
                   on_event: Callable[[str, str, dict], None] = _print_event) -> dict:
    items = available_items or list(config.ALL_ITEMS)
    history: list[dict] = []
    transcript: list[dict] = []
    turn_timings: list[dict] = []
    name = None
    started = time.perf_counter()

    def robot_says(text: str, reply: dict | None = None, info: dict | None = None):
        reply = reply or {"name": name, "item": None, "say": text}
        history.append(brain.assistant_turn(text, reply["name"], reply["item"]))
        transcript.append({"speaker": "robot", "text": text})
        on_event("robot", text, info or {})
        return speaker.say(text)

    def finish(outcome: str, item: str | None):
        return {
            "outcome": outcome,
            "item": item,
            "name": name,
            "turns": sum(1 for t in transcript if t["speaker"] == "person"),
            "duration_s": round(time.perf_counter() - started, 1),
            "transcript": transcript,
            "turn_timings": turn_timings,
        }

    robot_says(random.choice(GREETINGS))
    silent = 0

    for _ in range(config.MAX_TURNS):
        text, stt_info = listener.listen()

        if text is None:
            silent += 1
            on_event("silence", "", {})
            if silent > config.MAX_SILENT_TURNS:
                robot_says("Looks like you're busy. Have a great day!")
                return finish("no_response", None)
            robot_says(f"Are you still there? I have {_spoken_list(items)}.")
            continue

        silent = 0
        history.append({"role": "user", "content": text})
        transcript.append({"speaker": "person", "text": text})
        on_event("person", text, stt_info)

        reply = brain.respond(history, items)
        if reply["error"]:
            on_event("error", reply["error"], {})
        name = reply["name"] or name
        reply["name"] = name

        tts_info = robot_says(reply["say"], reply)
        timing = {**stt_info, "llm_s": reply["llm_s"], **tts_info}
        timing["reply_latency_s"] = round(
            timing.get("stt_s", 0) + timing["llm_s"] + timing.get("tts_first_audio_s", 0), 2)
        turn_timings.append(timing)
        on_event("decision", "", {"item": reply["item"], "name": name,
                               "reply_latency_s": timing["reply_latency_s"]})

        if reply["item"] in items:
            return finish("item", reply["item"])
        if reply["item"] == "none":
            return finish("declined", None)

    robot_says("I'll let you think about it. Come find me any time!")
    return finish("unclear", None)


class TextListener:
    """Type instead of talk. Same interface as stt.Listener, for testing without a mic."""

    def listen(self):
        try:
            text = input("\033[93m  you> \033[0m").strip()
        except EOFError:
            text = ""
        return (text or None), {}


class SilentSpeaker:
    """Prints instead of speaking (for --no-tts)."""

    def say(self, text: str) -> dict:
        return {"tts_first_audio_s": 0.0, "tts_total_s": 0.0}
