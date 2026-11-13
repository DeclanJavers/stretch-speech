"""Talks to the language server on the Mac (language-server/server.py) over HTTP.

Standard library only, so nothing extra goes into the VM's system Python.
"""

import json
import urllib.error
import urllib.request


class LanguageServerError(RuntimeError):
    pass


class LanguageClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, body: dict | None = None, timeout_s: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise LanguageServerError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LanguageServerError(f"{method} {path} -> can't reach {self.base_url}: {e}") from e

    def health(self) -> dict:
        return self._request("GET", "/health")

    def start_conversation(self, items: list[str]) -> str:
        return self._request("POST", "/conversation", {"items": items})["id"]

    def get_conversation(self, cid: str) -> dict:
        return self._request("GET", f"/conversation/{cid}")

    def say(self, text: str) -> dict:
        # Blocks until the Mac has finished speaking, so allow for a long sentence.
        return self._request("POST", "/say", {"text": text}, timeout_s=60)
