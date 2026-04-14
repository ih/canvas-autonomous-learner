"""Append-only JSONL event log so runs/ becomes a queryable history."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, runs_dir: str | Path, session: str | None = None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.path = self.runs_dir / f"events_{self.session}.jsonl"

    def log(self, event: str, **fields: Any) -> None:
        record = {"t": time.time(), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
