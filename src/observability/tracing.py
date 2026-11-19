from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TraceEvent:
    step: str
    args: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0
    token_count: int = 0
    status: str = "ok"

    def as_scorecard_dict(self) -> dict[str, str]:
        return {
            "step": self.step,
            "args": _compact(self.args),
            "latency_ms": f"{self.latency_ms:.1f}",
            "token_count": str(self.token_count),
            "status": self.status,
        }


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    @contextmanager
    def step(self, name: str, **args: Any) -> Iterator[TraceEvent]:
        event = TraceEvent(step=name, args=args)
        started = time.perf_counter()
        try:
            yield event
        except Exception:
            event.status = "error"
            raise
        finally:
            event.latency_ms = (time.perf_counter() - started) * 1000
            self.events.append(event)

    def scorecard_trace(self) -> list[dict[str, str]]:
        return [event.as_scorecard_dict() for event in self.events]


def traceable_step(name: str):
    try:
        from langsmith import traceable

        return traceable(name=name)
    except Exception:
        def decorator(func):
            return func

        return decorator


def _compact(value: Any, max_chars: int = 180) -> str:
    text = repr(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
