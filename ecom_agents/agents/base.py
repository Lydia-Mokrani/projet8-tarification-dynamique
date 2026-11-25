"""Brique de base : message échangé entre agents (format JSON standardisé)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Message:
    sender: str
    product_id: int
    week: int
    payload: dict
    confidence: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["payload"] = {k: (float(v) if hasattr(v, "item") else v) for k, v in self.payload.items()}
        return d


class Agent:
    name = "agent"
