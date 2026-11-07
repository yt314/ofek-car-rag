"""Bonus: question 12 needs two documents (OF-E5 and OF-E7). Free: no Claude call.

One search has to fit both "battery and charging" chunks into k=3 places.
Decomposing it into one search per car gives each car its own places.

  python bonus_decompose.py
"""
import json
from pathlib import Path

from eval_retrieval import build_index, needles
from rag.search import search

SUB_QUESTIONS = [
    "מה טווח הנסיעה של אופק E5 חשמלי (OF-E5)?",
    "מה טווח הנסיעה של אופק E7 חשמלי (OF-E7)?",
]


def decomposed(index, sub_questions, per_question=2):
    """One search per sub-question, the top `per_question` chunks of each, merged without duplicates."""
    chunks, seen = [], set()
    for sub in sub_questions:
        for c in search(index, sub, "hybrid", per_question, current_only=True):
            if c.id not in seen:
                seen.add(c.id)
                chunks.append(c)
    return chunks


def report(label, q, chunks):
    text = "\n".join(c.text for c in chunks)
    found = [n for n in needles(q) if n in text]
    ok = len(found) == len(needles(q))
    print(f"{label:<22} {'✓' if ok else '✗'}  found {found}   chunks {[c.id for c in chunks]}")


def main():
    golden = json.loads(Path("golden.json").read_text(encoding="utf-8"))
    q = next(q for q in golden if q["id"] == 12)
    index = build_index("headings")
    report("one search, k=3", q, search(index, q["question"], "hybrid", 3, current_only=True))
    report("two searches, 2 each", q, decomposed(index, SUB_QUESTIONS))


if __name__ == "__main__":
    main()
