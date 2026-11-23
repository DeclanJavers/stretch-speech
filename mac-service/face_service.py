"""Face registry and matching service. Runs on the Mac; the robot calls it over HTTP.

The robot owns person ids and decides when to register a face. This service only
stores face embeddings per person and compares new faces against them.

    python3 face_service.py                 # listens on 0.0.0.0:8100
    python3 face_service.py --threshold 0.5

Endpoints (JSON):
    GET  /health
    POST /faces/reset     {"session": "..."}                 forget everyone (start of a run)
    POST /faces/register  {"person_id": "p003", <face>}      -> {"views": n}
    POST /faces/identify  {<face>}                           -> {"person_id": "p003" | null, "score", "second"}
    POST /faces/verify    {"person_id": "p003", <face>}      -> {"match": bool, "score"}
    GET  /faces                                              -> {"p003": n_views, ...}

<face> is {"embedding": [floats]} or {"image_jpeg_b64": "..."}. Embeddings work now
(the robot's simulator sends synthetic ones). Image encoding needs a face model that
hasn't been chosen and tested yet, so images get HTTP 501 for now.

Memory only: nothing is written to disk.
"""

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_VIEWS = 20          # per person
DUPLICATE_VIEW = 0.97   # a new view this similar to a stored one adds nothing


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    if not n or not all(math.isfinite(x) for x in v):
        raise ValueError("invalid embedding")
    return [x / n for x in v]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class FaceRegistry:
    def __init__(self, threshold: float, margin: float):
        self.threshold, self.margin = threshold, margin
        self.gallery: dict[str, list[list[float]]] = {}
        self.session = None
        self.lock = threading.Lock()

    def embed(self, body: dict) -> list[float]:
        if "embedding" in body:
            return unit([float(x) for x in body["embedding"]])
        if "image_jpeg_b64" in body:
            raise NotImplementedError("image encoding not implemented yet: send an embedding")
        raise ValueError("send 'embedding' or 'image_jpeg_b64'")

    def score(self, pid: str, v: list[float]) -> float:
        return max((cosine(v, g) for g in self.gallery.get(pid, [])), default=-1.0)

    def register(self, pid: str, v: list[float]) -> int:
        with self.lock:
            views = self.gallery.setdefault(pid, [])
            if all(cosine(v, g) < DUPLICATE_VIEW for g in views):
                views.append(v)
                del views[:-MAX_VIEWS]
            return len(views)

    def identify(self, v: list[float]) -> dict:
        with self.lock:
            scores = sorted(((self.score(pid, v), pid) for pid in self.gallery), reverse=True)
        best, pid = scores[0] if scores else (-1.0, None)
        second = scores[1][0] if len(scores) > 1 else -1.0
        match = best >= self.threshold and best - second >= self.margin
        return {"person_id": pid if match else None, "score": round(best, 4), "second": round(second, 4),
                "candidate": pid}

    def verify(self, pid: str, v: list[float]) -> dict:
        with self.lock:
            s = self.score(pid, v)
        return {"match": s >= self.threshold, "score": round(s, 4)}


def make_handler(reg: FaceRegistry, verbose: bool = False):
    def note(text):
        if verbose:
            print(text, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "people": len(reg.gallery), "session": reg.session})
            if self.path == "/faces":
                return self._send(200, {pid: len(v) for pid, v in reg.gallery.items()})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/faces/reset":
                    with reg.lock:
                        reg.gallery.clear()
                        reg.session = body.get("session")
                    note(f"reset (session {reg.session})")
                    return self._send(200, {"ok": True})
                if self.path == "/faces/register":
                    out = {"views": reg.register(str(body["person_id"]), reg.embed(body))}
                    note(f"register {body['person_id']}: {out}")
                    return self._send(200, out)
                if self.path == "/faces/identify":
                    out = reg.identify(reg.embed(body))
                    note(f"identify: {out}")
                    return self._send(200, out)
                if self.path == "/faces/verify":
                    out = reg.verify(str(body["person_id"]), reg.embed(body))
                    note(f"verify {body['person_id']}: {out}")
                    return self._send(200, out)
                self._send(404, {"error": "not found"})
            except NotImplementedError as e:
                self._send(501, {"error": str(e)})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--threshold", type=float, default=0.45, help="min cosine similarity for a match")
    ap.add_argument("--margin", type=float, default=0.08, help="best match must beat the runner-up by this much")
    ap.add_argument("--verbose", action="store_true", help="print every call")
    args = ap.parse_args()
    reg = FaceRegistry(args.threshold, args.margin)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(reg, args.verbose))
    print(f"Face service on http://{args.host}:{args.port}  threshold={args.threshold} margin={args.margin}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
