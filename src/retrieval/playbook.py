from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_playbook(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("entries", [])
