# app/models.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyChunk:
    text: str
    section: str
    clause: Optional[str]
    page: int
    content_type: str  # "text" or "table"
