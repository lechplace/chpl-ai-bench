"""GPT baseline arms: same model with no tools (B) and with web search (C).

These are the two arms a reader can reproduce with their own OpenAI key. The
ChPL.AI (RAG) arm is not included here — its recorded responses are released
under data/results/ for inspection; this repository does not ship the backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI


@dataclass
class ArmResponse:
    arm: str               # "gpt_solo" | "gpt_web"
    model: str
    question_id: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    duration_ms: int = 0
    error: str | None = None


async def gpt_solo(
    oa: AsyncOpenAI, question: dict[str, Any], model: str,
) -> ArmResponse:
    """Arm B: the model with no tools (parametric knowledge only)."""
    start = time.perf_counter()
    try:
        resp = await oa.responses.create(model=model, input=question["question"])
        arm = ArmResponse(arm="gpt_solo", model=model, question_id=question["id"],
                          answer=resp.output_text)
    except Exception as exc:  # noqa: BLE001
        arm = ArmResponse(arm="gpt_solo", model=model, question_id=question["id"],
                          answer="", error=str(exc))
    arm.duration_ms = int((time.perf_counter() - start) * 1000)
    return arm


async def gpt_web(
    oa: AsyncOpenAI, question: dict[str, Any], model: str,
    tool_choice: str = "required",
) -> ArmResponse:
    """Arm C: the model with the hosted web_search tool."""
    start = time.perf_counter()
    try:
        resp = await oa.responses.create(
            model=model,
            input=question["question"],
            tools=[{"type": "web_search"}],
            tool_choice=tool_choice,
        )
        arm = ArmResponse(arm="gpt_web", model=model, question_id=question["id"],
                          answer=resp.output_text)
    except Exception as exc:  # noqa: BLE001
        arm = ArmResponse(arm="gpt_web", model=model, question_id=question["id"],
                          answer="", error=str(exc))
    arm.duration_ms = int((time.perf_counter() - start) * 1000)
    return arm
