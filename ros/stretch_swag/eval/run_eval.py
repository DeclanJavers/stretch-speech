"""Score the conversation LLM on labeled visitor lines, through the robot's real prompt path.

Each case becomes an event log (robot greeting, any earlier turns, the visitor's line,
then a 1 s pause), goes through render_chat -> LLM -> parse -> validate, exactly like
a tick in the robot, and the result is compared with the label:

    expect  pen | shirt | plushie   -> must call fetch_item(item)
            none                    -> must call end_conversation
            unclear                 -> must just reply (no tool)
            wait                    -> must call wait_for_visitor (sentence cut off)

Cases: Declan's 78 from language-server/eval/cases.jsonl + extra_cases.jsonl here.

    python3 eval/run_eval.py                                 # Mac LLM at 192.168.64.1:8081
    python3 eval/run_eval.py --runs 3 --url http://localhost:8081/v1
"""

import argparse
import collections
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from stretch_swag.core.conversation import Conversation  # noqa: E402
from stretch_swag.core.events import EventLog  # noqa: E402
from stretch_swag.core.llm import LLMClient  # noqa: E402
from stretch_swag.core.prompt import GREETINGS  # noqa: E402
from stretch_swag.core.registry import Registry  # noqa: E402
from stretch_swag.core.states import RobotState, State  # noqa: E402

ALL_ITEMS = ["pen", "shirt", "plushie"]
OFFER = "Hi there! I'm Stretch, and I'm giving away free swag today. Would you like a pen, a t-shirt, or a plushie?"
TOOLS = ["say_to_visitor", "fetch_item", "wait_for_visitor", "end_conversation"]


class Quiet:
    def say(self, text, pid):
        return 0.0

    def done(self, pid):
        pass


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def old_say(content: str) -> str:
    """Declan's history stored robot turns as JSON; we only need the spoken text."""
    try:
        return json.loads(content).get("say", content)
    except (json.JSONDecodeError, AttributeError):
        return content


def run_case(llm: LLMClient, case: dict) -> dict:
    items = case.get("items", ALL_ITEMS)
    log, reg, clock = EventLog(), Registry(), Clock()
    person = reg.new(t=clock.t)
    robot = RobotState(state=State.TALKING, target=person.id, stock={i: 5 for i in items})
    conv = Conversation(llm, log, reg, lambda: (robot, None, TOOLS), Quiet(), clock=clock)
    conv.start(person.id)
    log.append("state", person.id, t=clock.t, state=State.TALKING.value)
    log.append("said", person.id, t=clock.t, text=GREETINGS[0] if case.get("after_greeting") else OFFER)
    for turn in case.get("history", []):
        clock.t += 4
        if turn["role"] == "user":
            log.append("heard", person.id, t=clock.t, text=turn["content"])
        else:
            log.append("said", person.id, t=clock.t, text=old_say(turn["content"]))
    clock.t += 4
    log.append("heard", person.id, t=clock.t, text=case["text"])
    conv.last_spoke_end = clock.t - 1          # the robot finished speaking before this line
    clock.t += 1.0                             # a 1 s pause: a normal tick, wait tool still offered
    t0 = time.perf_counter()
    reply = conv.tick()
    ms = (time.perf_counter() - t0) * 1000
    if reply is None:
        got = "error"
    elif reply.tool == "wait_for_visitor":
        got = "wait"
    elif reply.tool == "fetch_item":
        got = reply.args.get("item")
    elif reply.tool == "end_conversation":
        got = "none"
    else:
        got = "unclear"
    return {"got": got, "ms": ms, "say": reply.speech if reply else "", "name": person.name}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.64.1:8081/v1")
    ap.add_argument("--model", default="lfm3-1.7b")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", help="only cases whose tag contains this")
    args = ap.parse_args()

    files = [HERE.parents[2] / "language-server" / "eval" / "cases.jsonl", HERE / "extra_cases.jsonl"]
    cases = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if args.only in c["tag"]]
    llm = LLMClient(args.url, args.model)

    rows = []
    for run in range(args.runs):
        for c in cases:
            r = run_case(llm, c)
            ok = r["got"] == c["expect"]
            name_ok = None
            if "expect_name" in c:
                name_ok = (r["name"] or "").lower() == c["expect_name"].lower()
            rows.append({**c, **r, "ok": ok, "name_ok": name_ok, "run": run})
            mark = "\033[92m ok \033[0m" if ok else "\033[91mFAIL\033[0m"
            print(f"{mark} {r['ms']:5.0f}ms [{c['tag']:<14}] {c['text'][:46]!r:<48} expect={c['expect']:<8} "
                  f"got={r['got']:<8} {('name=' + str(r['name'])) if 'expect_name' in c else ''} | {r['say'][:60]}")

    n = len(rows)
    print("\n" + "=" * 70)
    print(f"Model {args.model}: {sum(r['ok'] for r in rows)}/{n} = {sum(r['ok'] for r in rows) / n:.1%}"
          f"   median {sorted(r['ms'] for r in rows)[n // 2]:.0f} ms")
    by_tag = collections.defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["ok"])
    print("  by tag:  " + "  ".join(f"{t} {sum(v)}/{len(v)}" for t, v in sorted(by_tag.items())))
    conf = collections.Counter((r["expect"], r["got"]) for r in rows if not r["ok"])
    print("  mistakes (expected -> got):  " + ", ".join(f"{e}->{g} x{k}" for (e, g), k in conf.most_common()))
    names = [r["name_ok"] for r in rows if r["name_ok"] is not None]
    if names:
        print(f"  names: {sum(names)}/{len(names)}")
    out = HERE / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"  saved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
