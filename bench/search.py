"""Search over the corpus: dense (embeddings) or hybrid (embeddings + SQLite FTS5 BM25, fused with RRF).

    uv run python bench/search.py   # retrieval metrics: embedder x dense/hybrid

Document embeddings are cached in results/ per embedder and corpus content, so a server starts without re-encoding.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import documents, questions
from eval_embeddings import MODELS as EMBEDDERS, rank_tables
from hardware import pick

CACHE = Path(__file__).parent / "results"
RRF_K = 60  # standard reciprocal rank fusion constant
WORD = re.compile(r"\w+")


class Index:
    def __init__(self, embedder: str, hybrid: bool, device: str | None = None):
        self.docs = documents()
        self.hybrid = hybrid
        self.qp, dp, kwargs = EMBEDDERS[embedder]
        self.model = SentenceTransformer(embedder, device=device or str(pick()[0]), **kwargs)
        texts = [dp + d["text"] for d in self.docs]
        key = hashlib.sha256(json.dumps([embedder, texts]).encode()).hexdigest()[:16]
        path = CACHE / f"vectors-{embedder.replace('/', '_')}-{key}.npy"
        if path.exists():
            self.vectors = np.load(path)
        else:
            self.vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=16)
            path.parent.mkdir(exist_ok=True)
            np.save(path, self.vectors)
        if hybrid:
            self.fts = sqlite3.connect(":memory:", check_same_thread=False)
            self.fts.execute("CREATE VIRTUAL TABLE docs USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
            # Identifiers are split on "_" so PROCESS_INSTANCE matches "process" and "instance".
            self.fts.executemany("INSERT INTO docs(rowid, text) VALUES (?, ?)",
                                 [(i, d["text"].replace("_", " ")) for i, d in enumerate(self.docs)])

    def scores(self, question: str) -> np.ndarray:
        """A score per document, higher is better."""
        dense = self.vectors @ self.model.encode([self.qp + question], normalize_embeddings=True)[0]
        if not self.hybrid:
            return dense
        fused = np.zeros(len(self.docs))
        fused[np.argsort(-dense)] += 1 / (RRF_K + np.arange(1, len(self.docs) + 1))
        for rank, (rowid,) in enumerate(self.fts.execute(
                "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs)", (_fts_query(question),)), 1):
            fused[rowid] += 1 / (RRF_K + rank)
        return fused

    def tables(self, question: str, limit: int) -> list[str]:
        return rank_tables(self.scores(question), self.docs, {"table", "sql"})[:limit]

    def queries(self, question: str, limit: int) -> list[dict]:
        s = self.scores(question)
        return [self.docs[i] for i in np.argsort(-s) if self.docs[i]["kind"] == "sql"][:limit]


def _fts_query(question: str) -> str:
    """Any of the question's words; long words are cut to a prefix so Russian inflections still match
    ("инцидентов" -> инциден*)."""
    terms = []
    for w in WORD.findall(question.lower()):
        if len(w) < 3:
            continue
        stem = w[:max(4, len(w) - 3)] if len(w) > 5 else w
        terms.append(f'"{stem}"*')
    return " OR ".join(terms) or '""'


def main() -> None:
    qs = questions()
    for embedder in ("intfloat/multilingual-e5-small", "BAAI/bge-m3"):
        for hybrid in (False, True):
            idx = Index(embedder, hybrid)
            ranked = [rank_tables(idx.scores(q["q"]), idx.docs, {"table", "sql"}) for q in qs]
            m = {f"@{k}": round(sum(q["tables"] <= set(r[:k]) for q, r in zip(qs, ranked)) / len(qs), 3) for k in (1, 3, 5)}
            m["mrr"] = round(sum(1 / (min(r.index(t) for t in q["tables"]) + 1) for q, r in zip(qs, ranked)) / len(qs), 3)
            print(f"{embedder:35} {'hybrid' if hybrid else 'dense':6} {m}", flush=True)


if __name__ == "__main__":
    main()
