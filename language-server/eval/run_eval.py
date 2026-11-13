"""Score the LLM on labeled utterances (text only, no mic or speakers needed).

    uv run eval/run_eval.py                    # Qwen 3.5 4B, one pass
    uv run eval/run_eval.py --runs 3           # repeat to see run-to-run variance
    uv run eval/run_eval.py --model lfm3-1.7b  # compare another LM Studio model

"expect" labels: pen | shirt | plushie | none (declined) | unclear (robot should keep talking / confirm).
"""

import argparse
import collections
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from brain import Brain  # noqa: E402
from conversation import GREETINGS  # noqa: E402

HERE = pathlib.Path(__file__).parent
OPENER = Brain.assistant_turn(
    "Hi there! I'm Stretch, and I'm handing out free swag today. Would you like a pen, a t-shirt, or a plushie?",
    None, None)
# Real conversations open by asking the visitor's name, so some cases start here instead.
GREETING_OPENER = Brain.assistant_turn(GREETINGS[0], None, None)


def convert(turn: dict) -> dict:
    """Case files store past robot turns as {"name","item","say"}; rewrite to the model's format."""
    if turn["role"] != "assistant":
        return turn
    old = json.loads(turn["content"])
    return Brain.assistant_turn(old["say"], old.get("name"), old.get("item"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    args = ap.parse_args()

    cases = [json.loads(l) for l in pathlib.Path(args.cases).read_text().splitlines() if l.strip()]
    brain = Brain(model=args.model)
    brain.check()

    rows = []
    for run in range(args.runs):
        for c in cases:
            items = c.get("items", config.ALL_ITEMS)
            opener = GREETING_OPENER if c.get("after_greeting") else OPENER
            history = [opener, *map(convert, c.get("history", [])), {"role": "user", "content": c["text"]}]
            r = brain.respond(history, items)
            got = r["item"] if r["item"] is not None else "unclear"
            row = {"run": run, "text": c["text"], "tag": c["tag"], "expect": c["expect"], "got": got,
                   "ok": got == c["expect"], "say": r["say"], "llm_s": r["llm_s"], "error": r["error"]}
            if "expect_name" in c:
                row["name_ok"] = (r["name"] or "").lower() == c["expect_name"].lower()
            rows.append(row)
            mark = "\033[92m ok \033[0m" if row["ok"] else "\033[91mFAIL\033[0m"
            print(f"{mark} {row['llm_s']:>5.2f}s  [{c['tag']:<14}] {c['text']!r:<52} expect={c['expect']:<8} got={got}")

    # ---- summary ---------------------------------------------------------
    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n
    lat = [r["llm_s"] for r in rows]
    by_tag = collections.defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["ok"])
    names = [r["name_ok"] for r in rows if "name_ok" in r]
    errors = [r for r in rows if r["error"]]

    print("\n" + "=" * 60)
    print(f"Model: {args.model}   cases: {len(cases)}   runs: {args.runs}")
    print(f"Item accuracy:  {acc:.1%}  ({sum(r['ok'] for r in rows)}/{n})")
    if names:
        print(f"Name accuracy:  {sum(names) / len(names):.1%}  ({sum(names)}/{len(names)})")
    print(f"LLM latency:    mean {statistics.mean(lat):.2f}s   median {statistics.median(lat):.2f}s   "
          f"p90 {sorted(lat)[int(0.9 * (n - 1))]:.2f}s   max {max(lat):.2f}s")
    if errors:
        print(f"Errors:         {len(errors)} (first: {errors[0]['error']})")
    print("\nBy category:")
    for tag, oks in sorted(by_tag.items()):
        print(f"  {tag:<16} {sum(oks)}/{len(oks)}")

    fails = [r for r in rows if not r["ok"]]
    if fails:
        print("\nFailures:")
        for r in fails:
            print(f"  {r['text']!r}: expected {r['expect']}, got {r['got']}  -> said: {r['say']!r}")

    out = HERE / f"results-{args.model.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": args.model, "accuracy": acc, "rows": rows}, indent=2))
    print(f"\nSaved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
