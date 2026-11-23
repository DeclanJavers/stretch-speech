"""Client for the Mac's face service (mac-service/face_service.py).

The robot sends face cutouts and owns the person ids. Three calls:
    register(pid, face)   "this face belongs to person pid" (no comparison)
    identify(face)        "is this anyone I've registered?"   -> pid or None
    verify(pid, face)     "is this person pid?"               -> True/False

A face is either {"embedding": [...]} (the simulator cheat sends these) or
{"image_jpeg_b64": "..."} (a real camera cutout).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class FaceServiceError(RuntimeError):
    pass


class FaceClient:
    def __init__(self, base_url: str, timeout_s: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise FaceServiceError(f"{path}: HTTP {e.code} {e.read().decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise FaceServiceError(f"{path}: can't reach {self.base_url}: {e}") from e

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise FaceServiceError(f"can't reach {self.base_url}: {e}") from e

    def reset(self, session: str) -> None:
        self._post("/faces/reset", {"session": session})

    def register(self, pid: str, face: dict) -> int:
        return self._post("/faces/register", {"person_id": pid, **face})["views"]

    def identify(self, face: dict) -> tuple[str | None, float]:
        r = self._post("/faces/identify", face)
        return r.get("person_id"), r.get("score", 0.0)

    def verify(self, pid: str, face: dict) -> tuple[bool, float]:
        r = self._post("/faces/verify", {"person_id": pid, **face})
        return bool(r.get("match")), r.get("score", 0.0)
