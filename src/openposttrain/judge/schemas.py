from typing import Literal

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]
    reasoning_quality_score: int
    explanation: str
