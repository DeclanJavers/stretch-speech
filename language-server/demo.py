"""Run the speech demo.

    uv run demo.py                 # talk out loud: mic -> Whisper -> Qwen -> Kokoro
    uv run demo.py --auto --loop   # hands-free: start a new conversation each time one ends
    uv run demo.py --text          # type instead of talking (no mic needed)
    uv run demo.py --items pen,plushie   # pretend shirts ran out
"""

import argparse
import json
import os
import pathlib
import time
import warnings

# Quiet the library chatter so the terminal reads cleanly on video.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("CT2_VERBOSE", "-1")
warnings.filterwarnings("ignore")

import config  # noqa: E402
from brain import Brain  # noqa: E402
from conversation import SilentSpeaker, TextListener, talk_to_person  # noqa: E402


def banner(result: dict) -> None:
    color = {"item": "\033[92m", "declined": "\033[93m"}.get(result["outcome"], "\033[91m")
    line = "=" * 58
    print(f"\n{color}{line}")
    print(f"  OUTCOME : {result['outcome']}")
    print(f"  ITEM    : {result['item'] or '-'}")
    print(f"  NAME    : {result['name'] or '-'}")
    lat = [t["reply_latency_s"] for t in result["turn_timings"]]
    if lat:
        print(f"  REPLY LATENCY : avg {sum(lat) / len(lat):.2f}s  (max {max(lat):.2f}s) over {len(lat)} turns")
    print(f"{line}\033[0m\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true", help="type instead of using the microphone")
    ap.add_argument("--no-tts", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--auto", action="store_true", help="don't wait for Enter before each conversation")
    ap.add_argument("--loop", action="store_true", help="keep greeting new people until Ctrl+C")
    ap.add_argument("--items", default=",".join(config.ALL_ITEMS), help="comma list of items in stock")
    ap.add_argument("--voice", default=config.KOKORO_VOICE, help="Kokoro voice, e.g. af_heart, am_michael, bf_emma")
    ap.add_argument("--mic", type=int, default=None, help="input device index (see: python -m sounddevice)")
    ap.add_argument("--model", default=config.LLM_MODEL, help="model id, e.g. qwen/qwen3.5-4b")
    ap.add_argument("--base-url", default=config.LLM_BASE_URL,
                    help="OpenAI-compatible server: LM Studio is :1234, llama-server is :8081")
    args = ap.parse_args()

    items = [i.strip() for i in args.items.split(",") if i.strip() in config.ALL_ITEMS]

    print(f"Checking model server ({args.model} at {args.base_url})...", flush=True)
    brain = Brain(model=args.model, base_url=args.base_url)
    brain.check()

    if args.text:
        listener = TextListener()
    else:
        print(f"Loading Whisper ({config.WHISPER_MODEL}) and voice detection...", flush=True)
        from stt import Listener
        listener = Listener(input_device=args.mic)

    if args.no_tts:
        speaker = SilentSpeaker()
    else:
        print(f"Loading Kokoro (voice {args.voice})...", flush=True)
        from tts import Speaker
        speaker = Speaker(voice=args.voice)

    logs = pathlib.Path(__file__).parent / "logs"
    logs.mkdir(exist_ok=True)
    print(f"\nReady. Items in stock: {', '.join(items)}\n", flush=True)

    try:
        while True:
            if not args.auto:
                input("\033[90m[Press Enter when someone is in front of Stretch: stand-in for face detection]\033[0m")
            result = talk_to_person(listener, speaker, brain, available_items=items)
            banner(result)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (logs / f"conversation-{stamp}.json").write_text(json.dumps(result, indent=2))
            if not args.loop:
                break
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
