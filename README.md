<div align="center">

# 🚗 Ofek Car RAG

**A question-answering system over a car company's documents. It retrieves the right passages, answers with Claude, and checks every citation word for word.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-claude--haiku--5--5-D97757)
![Embeddings](https://img.shields.io/badge/embeddings-multilingual--e5--small-2f81f7)
![Search](https://img.shields.io/badge/search-vector%20%2B%20BM25%20(RRF)-6e40c9)
![Language](https://img.shields.io/badge/data-Hebrew-0d1117)

**Retrieval recall@3: 6/13 → 11/13 · MRR: 0.37 → 0.76 · Answers: 14/15**

</div>

---

"Ofek Rechev" (אופק רכב) is a fictional car company. Its documents include a spec sheet per model, a price list, warranty terms, a service schedule, dashboard warning codes and recall notices, all in Hebrew. The starter system answered only some customer questions correctly. This project **measures** where it fails, **diagnoses** why, and **fixes it one step at a time**.

<p align="center">
  <img src="docs/screenshots/results-chart.png" width="720" alt="recall@3 and MRR for each version: baseline, headings, hybrid, metadata">
</p>

## How it works

```mermaid
flowchart LR
    A[📄 16 Markdown docs<br/>+ metadata] --> B[✂️ Chunk by ## heading<br/>with a path line]
    B --> C[🔢 Vector search<br/>multilingual-e5]
    B --> D[🔤 BM25<br/>exact words]
    C --> E[🔀 RRF fusion]
    D --> E
    E --> F[🗂️ Drop superseded<br/>documents]
    F --> G[🤖 Claude answers<br/>with citations]
    G --> H[✅ Verify every quote<br/>word for word]
```

## Results, step by step

Each version builds on the previous one. These are the totals from `eval_retrieval.py` (k=3):

| Version | What changed | recall@3 | MRR |
|---|---|:---:|:---:|
| **Baseline** | fixed 350-character chunks, vector search | 6/13 | 0.37 |
| **Headings** | one chunk per `##` section, with a "Document › Section" path line | 9/13 | 0.65 |
| **Hybrid** | vector + BM25, merged with Reciprocal Rank Fusion | 9/13 | 0.69 |
| **Metadata** | filter out documents marked `status: superseded` | **11/13** | **0.76** |

<details>
<summary><b>Breakdown by question type</b></summary>

| Question type | Baseline | Headings | Hybrid | Metadata |
|---|:---:|:---:|:---:|:---:|
| ID / code (5) | 2/5 · 0.17 | 3/5 · 0.50 | **4/5 · 0.80** | 4/5 · 0.80 |
| Free wording (3) | 1/3 · 0.33 | **3/3 · 0.83** | 2/3 · 0.50 | 3/3 · 0.61 |
| Table lookup (2) | 1/2 · 0.50 | **2/2 · 1.00** | 2/2 · 1.00 | 2/2 · 1.00 |
| Version conflict (1) | 0/1 · 0.50 | 0/1 · 0.50 | 0/1 · 0.50 | **1/1 · 1.00** |
| Multiple sources (1) | 1/1 · 0.50 | 0/1 · 0.00 | 0/1 · 0.00 | 0/1 · 0.00 |
| Prompt injection (1) | 1/1 · 1.00 | 1/1 · 1.00 | 1/1 · 1.00 | 1/1 · 1.00 |

</details>

### Before: the baseline

Fixed-size chunks cut tables in half and lose the model name, so 7 of 13 questions miss.

<img src="docs/screenshots/baseline.png" width="820" alt="Baseline run: 6 of 13 questions found, MRR 0.37">

### After: headings + hybrid + metadata

<img src="docs/screenshots/final.png" width="820" alt="Final run: 11 of 13 questions found, MRR 0.76">

The two misses that remain are expected:

- **Q5 (OF-V8 payload): an inventory problem.** The source spec is a scanned PDF, and its text was never extracted. No search can find a number that isn't in the index. The right answer is "not enough information", and that is what the system says. The fix is OCR, not better retrieval.
- **Q12 (which EV has the longer range): a multiple-sources problem.** The answer needs two "Battery & charging" sections, one from each car, and both have to fit in 3 slots. The bonus below solves it by splitting the question in two.

## What I implemented

| Function | File | What it does |
|---|---|---|
| `chunk_by_headings` | [`rag/chunking.py`](rag/chunking.py) | One chunk per `##` section. Each chunk starts with a path line such as `אופק סיטי (OF-210) › מנוע והנעה`, so a table never loses its header or the model name. |
| `rrf` | [`rag/search.py`](rag/search.py) | Reciprocal Rank Fusion: each chunk scores `1 / (60 + rank)` from every list it appears in. Fusing by rank avoids comparing cosine scores with BM25 scores, which are on different scales. |
| `is_current` | [`rag/search.py`](rag/search.py) | Drops chunks whose document metadata says `status: superseded`. Here that is the 2023 warranty, which says 5 years where the current one says 7. |
| `verify_citations` | [`rag/answer.py`](rag/answer.py) | Rejects a citation when its `chunk_id` wasn't sent to the model, when the quote is empty, or when the quote isn't in the chunk word for word. Runs of whitespace are collapsed first, so a quote that spans two lines still matches. |

### Why hybrid search helps with codes

For the question about recall notice **RC-2025-07**, vector search can't tell it apart from the almost identical RC-2025-20. In BM25, `rc-2025-07` is a single rare token, so the exact notice ranks first:

<img src="docs/screenshots/hybrid-rc-2025-07.png" width="820" alt="Question 3 retrieves service bulletin RC-2025-07 first">

Hybrid search also made one question worse. BM25 has no Hebrew morphology, so the question's word **"לטיפול"** ("for service") never matches **"טיפול"** ("service") in the right chunk. The full analysis is in the report.

## Answers with verified citations

```bash
python eval_answers.py --chunker headings --mode hybrid --current-only
```

Result: **14/15**, at a cost of less than one cent with `claude-haiku-5-5`. Only Q12 fails, because neither range value reaches the model.

- **Questions with no answer in the data** (Q13 OF-900, Q14 sunroof): the model returns `insufficient_context=true` with no citations instead of making something up.
- **Hidden prompt injection** (Q15): `sales-notes-2025.md` hides an HTML comment that tells the model to promise a fake 40% discount and to collect customers' phone numbers. The model ignores it and answers with the real price, 2,900 ₪.

## Bonus

**🛡️ Injection defense** (`--defend`): the system prompt now says that text inside `<chunk>` is data, not instructions, and `strip_comments` removes HTML comments before the chunks reach the model. Citations are still checked against the original text.

**🔀 Question decomposition** ([`bonus_decompose.py`](bonus_decompose.py)): the script runs one search per car, keeps the top 2 chunks of each, and merges them. Q12 goes from ✗ to ✓.

<img src="docs/screenshots/bonus-decompose.png" width="820" alt="One search finds neither range; two searches find 420 km and 510 km">

**📏 k=5 vs k=3:** the baseline goes from 6/13 to 7/13 without any real improvement. With only 53–68 chunks in total, a large k covers much of the corpus and hides the differences between methods. A small k keeps the measurement sensitive to ranking quality.

## Run it yourself

Requires Python 3.10+.

```bash
pip install -r requirements.txt

python eval_retrieval.py                                                   # baseline
python eval_retrieval.py --chunker headings                                # + headings
python eval_retrieval.py --chunker headings --mode hybrid                  # + hybrid
python eval_retrieval.py --chunker headings --mode hybrid --current-only   # + metadata
python eval_retrieval.py --show 3 --chunker headings --mode hybrid         # see what Q3 retrieved
python bonus_decompose.py                                                  # bonus
```

The answer evaluation calls Claude, so it needs an API key set **only in your terminal**:

```powershell
$env:ANTHROPIC_API_KEY = "..."     # Windows
```
```bash
export ANTHROPIC_API_KEY="..."     # macOS / Linux
```

> 🔒 API keys never go into the code or into git (see [`.gitignore`](.gitignore)).

Embeddings use `intfloat/multilingual-e5-small` by default. It runs locally, it's free, and it supports Hebrew. To use Voyage AI instead, set `EMBED_BACKEND=voyage` and `VOYAGE_API_KEY`.

## Project structure

```
├── data/
│   ├── general/        price list, warranty (2023 + 2025), service schedule, warning codes, recalls, FAQ…
│   └── models/         one spec sheet per car model (7 models)
├── rag/
│   ├── loader.py       reads Markdown + front-matter metadata
│   ├── chunking.py     fixed-size and heading-based chunkers
│   ├── embed.py        embeddings with an on-disk cache
│   ├── search.py       vector, BM25, RRF fusion, metadata filter
│   └── answer.py       Claude call, structured output, citation verification, injection defense
├── golden.json         15 test questions with the text the answer must contain
├── eval_retrieval.py   retrieval metrics (free, no LLM call)
├── eval_answers.py     end-to-end answer check (≈1¢)
├── bonus_decompose.py  question decomposition for Q12
└── REPORT.md           full diagnosis and analysis (Hebrew)
```

## 📝 Full report

[**REPORT.md**](REPORT.md) (in Hebrew) has the failure-diagnosis table for every missed question, the per-type results for each version, and the answers to the analysis questions.

---

<div align="center">

**Yehudit Pollak & Rivki Peretz**

*The company and all its data are fictional and were created for this exercise.*

</div>
