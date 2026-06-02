"""LLM-as-judge: blind grading 4 metryk z anonimizacją ramion."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel


class MetricScore(BaseModel):
    score: int
    rationale: str = ""


class SystemVerdict(BaseModel):
    fidelity: MetricScore
    clinical: MetricScore
    traceability: MetricScore
    safety: MetricScore
    is_refusal: bool = False


class JudgeResponse(BaseModel):
    """Stałe pola system_1..system_3 zamiast dict[str, Model].

    OpenAI strict structured output NIE wspiera dict[str, X] (additionalProperties /
    zmienne klucze) — API zwraca 400 mimo że lokalny to_strict_json_schema przechodzi.
    Eksperyment ocenia MAX 3 ramiona (anonimizowane do system_1..3), więc modelujemy je
    jako trzy opcjonalne pola o stałych nazwach. Optional → SDK dodaje null do unii typów
    i zachowuje klucz w `required`, co strict mode akceptuje.
    """

    system_1: Optional[SystemVerdict] = None
    system_2: Optional[SystemVerdict] = None
    system_3: Optional[SystemVerdict] = None


def _coerce_score(v: Any) -> int:
    """Toleruje {score: int}, bare int oraz None/śmieci → 0 (defense in depth)."""
    if isinstance(v, dict):
        v = v.get("score", 0)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _rationale(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("rationale", "") or "")
    return ""


def anonymize_arms(
    arms: dict[str, str], seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Zwraca (labeled, mapping).

    labeled: {"system_1": odpowiedź, ...} w losowej (deterministycznej) kolejności.
    mapping: {"system_1": "chpl_ai", ...} — do deanonimizacji po werdykcie.
    """
    real_arms = list(arms.keys())
    rng = random.Random(seed)
    rng.shuffle(real_arms)
    labeled, mapping = {}, {}
    for i, real in enumerate(real_arms, 1):
        label = f"system_{i}"
        labeled[label] = arms[real]
        mapping[label] = real
    return labeled, mapping


def parse_verdict(raw: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Mapuje werdykt {system_N: {...}} z powrotem na realne ramiona."""
    out: dict[str, Any] = {}
    metrics = ("fidelity", "clinical", "traceability", "safety")
    for label, real in mapping.items():
        v = raw.get(label) or {}
        out[real] = {
            **{m: _coerce_score(v.get(m)) for m in metrics},
            "is_refusal": bool(v.get("is_refusal", False)),
            "rationales": {m: _rationale(v.get(m)) for m in metrics},
        }
    return out


def _build_prompt(
    question: str, gt_fragment: str, labeled: dict[str, str], rubric: dict[str, Any],
) -> str:
    metrics_desc = "\n".join(
        f"- {k} ({m['label']}): 0={m['0']}; 1={m['1']}; 2={m['2']}"
        for k, m in rubric["metrics"].items()
    )
    systems = "\n\n".join(f"### {label}\n{ans}" for label, ans in labeled.items())
    system_keys = ", ".join(labeled.keys())
    return f"""Jesteś ekspertem farmakologii klinicznej oceniającym odpowiedzi systemów AI.

PYTANIE:
{question}

FRAGMENT ChPL (źródło prawdy — oceniaj wierność WZGLĘDEM tego fragmentu):
{gt_fragment}

ODPOWIEDZI DO OCENY (anonimowe):
{systems}

METRYKI (oceniaj każdy system osobno, skala 0-2):
{metrics_desc}

Dodatkowo oznacz is_refusal=true jeśli system odmówił odpowiedzi (twierdzi że brak danych) zamiast podać konkretną odpowiedź kliniczną.

Wypełnij pole NA NAJWYŻSZYM POZIOMIE dla KAŻDEGO ocenianego systemu ({system_keys}) — po jednym kluczu na pokazany system. Każda wartość to {{"fidelity": {{"score": 0-2, "rationale": "..."}}, "clinical": {{...}}, "traceability": {{...}}, "safety": {{...}}, "is_refusal": bool}}. Pola systemów których NIE pokazano pozostaw puste (null)."""


def load_rubric(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


async def judge_question(
    oa: AsyncOpenAI, judge_model: str, question: str, gt_fragment: str,
    arms: dict[str, str], rubric: dict[str, Any], seed: int,
) -> dict[str, Any]:
    """Full pass: anonymize -> call judge (structured output) -> de-anonymize.

    Structured output (responses.parse + text_format=JudgeResponse) guarantees
    well-formed JSON. The whole call+parse is wrapped in try/except as a
    last-resort safety net: a single anomaly returns {} rather than raising.
    """
    labeled, mapping = anonymize_arms(arms, seed=seed)
    prompt = _build_prompt(question, gt_fragment, labeled, rubric)
    try:
        resp = await oa.responses.parse(
            model=judge_model, input=prompt, text_format=JudgeResponse,
        )
        parsed: JudgeResponse | None = resp.output_parsed
        if parsed is None:
            return {}
        raw: dict[str, Any] = {}
        for label in ("system_1", "system_2", "system_3"):
            sv = getattr(parsed, label, None)
            if sv is not None:
                raw[label] = sv.model_dump()
        return parse_verdict(raw, mapping)
    except Exception:
        return {}
