"""HTTP API around the speech module, so the ROS side (in the VM) can drive it.

Runs on the Mac, next to the mic, speakers and llama-server. It only wraps
Declan's code (brain.py, conversation.py, stt.py, tts.py) and changes none of it.

    uv run server.py                  # mic + speakers
    uv run server.py --text           # type the visitor's lines into this terminal
    uv run server.py --text --no-tts  # no audio at all

Endpoints (JSON in, JSON out):
    GET  /health              -> {"ok": true, "busy": bool, "model": ..., "items": [...]}
    POST /conversation        {"items": ["pen", ...]}  (optional)
                              -> {"id": "..."}   starts talk_to_person() in the background
                                 409 if a conversation or /say is already using the audio
    GET  /conversation/<id>   -> {"id", "status": "running"|"done"|"error", "events": [...],
                                  "result": <talk_to_person() result> | null, "error": str | null}
    POST /say                 {"text": "Here you go!"}  -> {"ok": true, ...timings}  (blocks until spoken)

The ROS node polls GET /conversation/<id> instead of holding one request open
for a whole conversation.
"""

import argparse
import json
import os
import pathlib
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Same quieting as demo.py so the terminal stays readable.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("CT2_VERBOSE", "-1")
warnings.filterwarnings("ignore")

import config  # noqa: E402
import conversation  # noqa: E402
from brain import Brain  # noqa: E402
from conversation import SilentSpeaker, TextListener, talk_to_person  # noqa: E402

LOGS = pathlib.Path(__file__).parent / "logs"


class SpeechService:
    """Owns the loaded models. Only one thing may use the mic/speakers at a time."""

    def __init__(self, brain, listener, speaker, items: list[str]):
        self.brain, self.listener, self.speaker = brain, listener, speaker
        self.items = items
        self.audio = threading.Lock()            # held for a whole conversation or one /say
        self.conversations: dict[str, dict] = {}
        self.lock = threading.Lock()             # guards self.conversations

    def start_conversation(self, items: list[str] | None) -> str | None:
        if not self.audio.acquire(blocking=False):
            return None
        items = [i for i in (items or self.items) if i in config.ALL_ITEMS] or list(self.items)
        cid = uuid.uuid4().hex[:8]
        record = {"id": cid, "status": "running", "items": items, "events": [],
                  "result": None, "error": None, "started": time.time()}
        with self.lock:
            self.conversations[cid] = record

        def on_event(kind, text, info):
            conversation._print_event(kind, text, info)    # keep demo.py's terminal output
            with self.lock:
                record["events"].append({"kind": kind, "text": text, "info": info, "t": time.time()})

        def run():
            try:
                result = talk_to_person(self.listener, self.speaker, self.brain,
                                        available_items=items, on_event=on_event)
                LOGS.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                (LOGS / f"conversation-{stamp}.json").write_text(json.dumps(result, indent=2))
                with self.lock:
                    record.update(status="done", result=result)
                print(f"\033[92m[server] conversation {cid} done: {result['outcome']} "
                      f"{result['item'] or ''}\033[0m", flush=True)
            except Exception as e:  # noqa: BLE001 - report to the client instead of dying
                with self.lock:
                    record.update(status="error", error=repr(e))
                print(f"\033[91m[server] conversation {cid} failed: {e!r}\033[0m", flush=True)
            finally:
                self.audio.release()

        print(f"\033[90m[server] conversation {cid} started, items={items}\033[0m", flush=True)
        threading.Thread(target=run, daemon=True).start()
        return cid

    def get_conversation(self, cid: str) -> dict | None:
        with self.lock:
            record = self.conversations.get(cid)
            return json.loads(json.dumps(record)) if record else None   # copy

    def say(self, text: str) -> dict | None:
        if not self.audio.acquire(timeout=30):
            return None
        try:
            conversation._print_event("robot", text, {})
            return self.speaker.say(text)
        finally:
            self.audio.release()


def make_handler(service: SpeechService, model: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "busy": service.audio.locked(),
                                        "model": model, "items": service.items})
            if self.path.startswith("/conversation/"):
                record = service.get_conversation(self.path.rsplit("/", 1)[1])
                return self._send(200, record) if record else self._send(404, {"error": "unknown id"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            if self.path == "/conversation":
                cid = service.start_conversation(body.get("items"))
                if cid is None:
                    return self._send(409, {"error": "busy: audio already in use"})
                return self._send(200, {"id": cid})
            if self.path == "/say":
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._send(400, {"error": "text is required"})
                timings = service.say(text)
                if timings is None:
                    return self._send(409, {"error": "busy: audio already in use"})
                return self._send(200, {"ok": True, **timings})
            self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):  # the ROS node polls often; don't flood the terminal
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so the VM can reach it")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--text", action="store_true", help="type the visitor's lines instead of using the mic")
    ap.add_argument("--no-tts", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--items", default=",".join(config.ALL_ITEMS), help="comma list of items in stock")
    ap.add_argument("--voice", default=config.KOKORO_VOICE)
    ap.add_argument("--mic", type=int, default=None, help="input device index (see: python -m sounddevice)")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--base-url", default=config.LLM_BASE_URL)
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

    service = SpeechService(brain, listener, speaker, items)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, args.model))
    print(f"\nLanguage server ready on http://{args.host}:{args.port}  "
          f"(from the VM: http://192.168.64.1:{args.port})  items: {', '.join(items)}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
