"""One fetcher for the forward test's live sources, with an on-disk record of every response.

Every client in ``forward/`` reads through a :class:`Fetcher`, so tests can hand them
recorded fixtures instead and never touch the network. Within one fetcher a URL is
requested at most once; each body is also written under ``cache_dir/<run stamp>/`` with
a manifest, so a forward-test run can be audited against exactly what the sources said.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


class Fetcher:
    """GET with an identifying User-Agent, a delay between requests, and a response record."""

    def __init__(self, cache_dir: Path, user_agent: str, *, delay_s: float = 1.0) -> None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(cache_dir) / stamp
        self.user_agent = user_agent
        self.delay_s = delay_s
        self._bodies: dict[str, str] = {}
        self._last_request = 0.0

    def text(self, url: str, headers: dict | None = None) -> str:
        """The body of ``url``; a second call for the same URL reuses the first response.

        Raises:
            requests.HTTPError: on a non-2xx response.
        """
        if url in self._bodies:
            return self._bodies[url]
        wait = self.delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(
            url, headers={"User-Agent": self.user_agent, **(headers or {})}, timeout=30
        )
        self._last_request = time.monotonic()
        resp.raise_for_status()
        self._bodies[url] = resp.text
        self._record(url, resp.text)
        return resp.text

    def json(self, url: str, headers: dict | None = None) -> Any:
        return json.loads(self.text(url, headers))

    def _record(self, url: str, body: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".txt"
        (self.run_dir / name).write_text(body, encoding="utf-8")
        with (self.run_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": url, "file": name}) + "\n")
