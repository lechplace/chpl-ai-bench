"""Minimal, self-contained reproduction script.

Re-issues the two GPT baseline arms (no-tools and web-search) over the 20 SOR
questions with YOUR OpenAI key, so anyone can verify our GPT results are not
fabricated. Optionally re-runs the blind LLM judge over the produced answers.

Usage:
    export OPENAI_API_KEY=sk-...        # your own key
    python -m bench.reproduce                       # GPT arms only
    python -m bench.reproduce --judge               # GPT arms + re-run the judge
    python -m bench.reproduce --out my_run.jsonl    # write results to a file

The ChPL.AI (RAG) arm is NOT reproduced here — its recorded answers are in
data/results/fazaA-2026-06-02/raw.jsonl for inspection and comparison.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from bench.arms import gpt_solo, gpt_web
from bench.judge import judge_question, load_rubric

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "sor_questions.yaml"
RUBRIC = ROOT / "bench" / "rubric.yaml"
RECORDED = ROOT / "data" / "results" / "fazaA-2026-06-02" / "raw.jsonl"

# The pinned model snapshot used in Phase A. Keep pinned for faithful comparison.
MODEL = "gpt-5.5-2026-04-23"


def load_questions() -> list[dict]:
    data = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    return data["tests"] if isinstance(data, dict) else data


def load_recorded_chpl() -> dict[str, dict]:
    """Map question_id -> our recorded ChPL.AI answer (for the judge's ground truth)."""
    out: dict[str, dict] = {}
    if not RECORDED.exists():
        return out
    for line in RECORDED.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("arm") == "chpl_ai":
            out[r["question_id"]] = r
    return out


def _gt_fragment(chpl_record: dict | None) -> str:
    """Ground-truth ChPL fragment from our recorded citations (for the judge)."""
    if not chpl_record or not chpl_record.get("citations"):
        return "(no ChPL fragment available)"
    parts = []
    for c in chpl_record["citations"]:
        parts.append(
            f"[{c.get('drug_name', '?')} {c.get('section_label', '')}] "
            f"{c.get('text_snippet', '')}"
        )
    return "\n".join(parts)


async def main(run_judge: bool, out_path: Path | None) -> None:
    oa = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    questions = load_questions()
    recorded = load_recorded_chpl()
    rubric = load_rubric(RUBRIC) if run_judge else {}
    records = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ...", flush=True)
        solo, web = await asyncio.gather(
            gpt_solo(oa, q, MODEL),
            gpt_web(oa, q, MODEL),
        )

        verdict = {}
        if run_judge:
            # deterministic blind permutation seed = SHA-256(model:question_id)
            seed = int(hashlib.sha256(f"{MODEL}:{q['id']}".encode()).hexdigest(), 16) % (2**31)
            gt = _gt_fragment(recorded.get(q["id"]))
            arms_text = {"gpt_solo": solo.answer, "gpt_web": web.answer}
            verdict = await judge_question(oa, MODEL, q["question"], gt, arms_text, rubric, seed)

        for arm in (solo, web):
            j = verdict.get(arm.arm, {})
            records.append({
                "question_id": q["id"],
                "arm": arm.arm,
                "model": MODEL,
                "answer": arm.answer,
                "error": arm.error,
                "judge": {k: j.get(k, 0) for k in ("fidelity", "clinical", "traceability", "safety")} if run_judge else None,
            })

    if out_path:
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {len(records)} records to {out_path}")
    else:
        ok = sum(1 for r in records if not r["error"])
        print(f"\nDone: {len(records)} responses ({ok} ok). "
              f"Re-run with --out to save, or compare against "
              f"data/results/fazaA-2026-06-02/raw.jsonl.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="also re-run the blind LLM judge over the GPT answers")
    ap.add_argument("--out", type=Path, default=None,
                    help="write results to this JSONL file")
    args = ap.parse_args()
    asyncio.run(main(args.judge, args.out))
