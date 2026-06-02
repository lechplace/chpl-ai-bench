# ChPL.AI vs ChatGPT — GPT baselines & recorded RAG responses (Phase A)

This repository accompanies a small benchmark comparing a retrieval-augmented
drug-information assistant (**ChPL.AI**, RAG over **997 Polish SmPC / ChPL
documents** — Charakterystyki Produktu Leczniczego / Summaries of Product
Characteristics) against a general-purpose model used two ways: **GPT with no
tools** and **GPT with web search**. The underlying model is pinned to
`gpt-5.5-2026-04-23`. The benchmark uses **20 emergency-department (SOR)
questions**, and each answer is scored by a **blind LLM judge** on four 0–2
metrics (fidelity to source, clinical usefulness, traceability, safe
abstention).

The purpose of this repository is narrow and deliberate:

1. **So our GPT results cannot be taken on faith — anyone can re-run them.** The
   GPT-no-tools and GPT-web arms are reproducible from this repo with your own
   OpenAI key.
2. **So our ChPL.AI answers are open for comparison.** Every ChPL.AI response —
   the full text, the cited ChPL/SmPC section, and the quote — is recorded under
   `data/results/` for inspection.
3. **So the judge is auditable.** Every judge score plus the judge's reasoning is
   in the data, and the judge code and rubric are included.

> This repository is **not** a full pipeline to reproduce an entire publication.
> It does **not** ship our ChPL.AI backend, vector store, or keys. The ChPL.AI
> answers are released as **recorded outputs for inspection**, not as a live
> service.

## Key result (n = 20)

| Metric                                       | ChPL.AI (RAG) | GPT-no-tools | GPT-web |
|----------------------------------------------|:-------------:|:------------:|:-------:|
| Fidelity to source (judge, 0–2)              | **2.00**      | 1.05         | 0.85    |
| Refusal specificity (abstain when uncovered) | **0.60**      | 0.00         | 0.00    |
| Verifiable references to a ChPL document     | **16 / 20**   | 0 / 20       | 0 / 20  |

ChPL.AI is the only arm that grounds answers in citable source documents and
that abstains when the corpus does not cover a question. The two GPT arms never
abstain and never return a verifiable source-document reference.

## 1. Reproduce the GPT baselines (your own OpenAI key)

The model is pinned to `gpt-5.5-2026-04-23` in `bench/reproduce.py`, so you
re-issue the exact same GPT-no-tools and GPT-web queries:

```bash
pip install -e .                 # or: pip install openai pyyaml pydantic
export OPENAI_API_KEY=sk-...      # your own key
python -m bench.reproduce                  # GPT arms only
python -m bench.reproduce --judge          # GPT arms + re-run the blind judge
python -m bench.reproduce --out my_run.jsonl   # save results to compare
```

The 20 prompts are in `data/sor_questions.yaml`; the exact per-arm prompting is
in `bench/arms.py` (`gpt_solo`, `gpt_web`). Compare your output against our
recorded run in `data/results/fazaA-2026-06-02/raw.jsonl`.

## 2. Inspect our ChPL.AI responses (no API key)

Every ChPL.AI answer, with its cited ChPL/SmPC section and quote, is in
`data/results/fazaA-2026-06-02/raw.jsonl`:

```bash
python - <<'PY'
import json
for line in open("data/results/fazaA-2026-06-02/raw.jsonl"):
    r = json.loads(line)
    if r["arm"] == "chpl_ai":
        print(r["question_id"], "| refused:", r["refused"])
        print(r["answer"][:500])
        for c in r.get("citations", []):
            print("  ref:", c["drug_name"], c["section_label"], "->", c["text_snippet"][:80])
        print("-" * 60)
PY
```

OpenAI file handles in the data are anonymised to stable pseudonyms
(`doc_01`, `doc_02`, …).

## 3. The LLM judge — scores and code

Every response in `raw.jsonl` carries the **blind LLM-judge** verdict, including
the judge's written reasoning:

```json
"judge":            {"fidelity": 2, "clinical": 2, "traceability": 1, "safety": 2},
"judge_rationales": {"fidelity": "…", "clinical": "…", "traceability": "…", "safety": "…"}
```

The judge is open, not a black box:

- **Rubric** — the 0–2 criteria for all four metrics: [`bench/rubric.yaml`](bench/rubric.yaml).
- **Code** — [`bench/judge.py`](bench/judge.py). It anonymises the arms to
  `system_1/2/…` with a deterministic per-question permutation
  (`seed = SHA-256(model:question_id)`), so grading is **blind** and
  reproducible, then maps the verdict back. It uses OpenAI structured output
  (`responses.parse` + a Pydantic schema) so scores are well-formed.
- **Re-run it** with `python -m bench.reproduce --judge` (your own key).

> The judge shares the model family with the graded systems, so its scores carry
> known biases — **self-preference**, and a **citation-echo** effect on the
> ChPL.AI arm (the judge sees ChPL.AI's own citation as the reference). Scores
> are released **as-is, with full reasoning, for independent scrutiny** — not as
> ground truth.

## Data

- `data/sor_questions.yaml` — the 20 SOR benchmark questions.
- `data/results/fazaA-2026-06-02/raw.jsonl` — per-(arm, question) responses,
  citations, refusals, and judge verdicts for all three arms.
- `data/results/fazaA-2026-06-02/scores.csv` — a flat table of the same scores.

## Reproducibility notes

- **Model pinned** to `gpt-5.5-2026-04-23` for all arms and the judge.
- **Deterministic blind permutation** — judge seed is `SHA-256(model:question_id)`.
- Small sample (**n = 20**); the judge biases above mean these numbers are a
  pilot signal, not an established result.

## License

- **Code** (`bench/`): MIT — see [`LICENSE`](LICENSE).
- **Data** (`data/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA).

## Citation

```bibtex
@misc{chplai_bench_2026,
  title        = {ChPL.AI vs ChatGPT: Retrieval-Grounded Drug Information vs
                  General-Purpose LLMs on Polish Emergency-Department Questions
                  (Phase A)},
  author       = {ChPL.AI Team},
  year         = {2026},
  howpublished = {Reproducibility package},
  note         = {TODO: add Zenodo DOI}
}
```
