"""Speaks text out loud on the Mac with Kokoro (Declan's language-server/tts.py, unchanged).

The robot's speaker_node POSTs each line here. Run it from the language-server folder so
it uses that uv environment (Kokoro is installed there):

    cd language-server && uv run ../mac-service/speech_service.py            # 0.0.0.0:8200
    cd language-server && uv run ../mac-service/speech_service.py --voice am_michael
    python3 mac-service/speech_service.py --print                            # no audio (testing)

    POST /say {"text": "..."}  -> {"ok": true, "seconds": 1.8}   (returns after speaking)
    GET  /health
"""

import argparse
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LANGUAGE_SERVER = pathlib.Path(__file__).resolve().parents[1] / "language-server"


class PrintSpeaker:
    def say(self, text):
        print(f"(say) {text}", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice, e.g. af_heart, am_michael, bf_emma")
    ap.add_argument("--print", action="store_true", help="print instead of speaking (no Kokoro needed)")
    args = ap.parse_args()

    if args.print:
        speaker = PrintSpeaker()
    else:
        sys.path.insert(0, str(LANGUAGE_SERVER))
        print(f"Loading Kokoro (voice {args.voice})...", flush=True)
        from tts import Speaker  # Declan's module
        speaker = Speaker(voice=args.voice)
    lock = threading.Lock()                 # one line at a time, in order

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/say":
                return self._send(404, {"error": "not found"})
            try:
                text = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["text"])
            except (KeyError, ValueError):
                return self._send(400, {"error": "send {\"text\": ...}"})
            with lock:
                t0 = time.perf_counter()
                print(f"STRETCH: {text}", flush=True)
                speaker.say(text)
                self._send(200, {"ok": True, "seconds": round(time.perf_counter() - t0, 2)})

        def log_message(self, *a):
            pass

    print(f"Speech service on http://{args.host}:{args.port}", flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
