"""Retrieval quality of embedding models on the Camunda corpus: do the top-k tables contain the gold tables?

    uv run python bench/eval_embeddings.py [model ...]   ->  bench/results/embeddings.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import documents, questions

TASK = "Given a question in Russian about data, retrieve the database tables and SQL queries that answer it"
# model -> (query prefix, document prefix, extra kwargs)
MODELS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("", "", {}),
    "intfloat/multilingual-e5-small": ("query: ", "passage: ", {}),
    "intfloat/multilingual-e5-base": ("query: ", "passage: ", {}),
    "ai-forever/ru-en-RoSBERTa": ("search_query: ", "search_document: ", {}),
    "BAAI/bge-m3": ("", "", {}),
    "deepvk/USER-bge-m3": ("", "", {}),
    "Alibaba-NLP/gte-multilingual-base": ("", "", {"trust_remote_code": True}),
    "Qwen/Qwen3-Embedding-0.6B": (f"Instruct: {TASK}\nQuery: ", "", {}),
}
K = (3, 5, 8)
OUT = Path(__file__).parent / "results" / "embeddings.json"


def rank_tables(sims: np.ndarray, docs: list[dict], kinds: set[str]) -> list[str]:
    """Tables ordered by their best-matching document; a SQL hit counts for every table it reads."""
    best: dict[str, float] = {}
    for i in np.argsort(-sims):
        if docs[i]["kind"] in kinds:
            for t in docs[i]["tables"]:
                best.setdefault(t, float(sims[i]))
    return sorted(best, key=best.get, reverse=True)


def evaluate(name: str) -> dict:
    qp, dp, kwargs = MODELS[name]
    model = SentenceTransformer(name, device="mps", **kwargs)
    docs, qs = documents(), questions()
    t0 = time.perf_counter()
    d = model.encode([dp + x["text"] for x in docs], normalize_embeddings=True, batch_size=16)
    t_docs = time.perf_counter() - t0
    t0 = time.perf_counter()
    q = model.encode([qp + x["q"] for x in qs], normalize_embeddings=True)
    t_query = (time.perf_counter() - t0) / len(qs)
    sims = q @ d.T
    result = {"model": name, "dim": int(d.shape[1]), "index_sec": round(t_docs, 1), "query_ms": round(t_query * 1000, 1)}
    for label, kinds in (("model", {"table"}), ("model+sql", {"table", "sql"})):
        ranked = [rank_tables(sims[i], docs, kinds) for i in range(len(qs))]
        for k in K:
            # all gold tables found in the top k
            result[f"{label}@{k}"] = round(sum(x["tables"] <= set(r[:k]) for x, r in zip(qs, ranked)) / len(qs), 3)
        result[f"{label}_mrr"] = round(sum(1 / (min(r.index(t) for t in x["tables"]) + 1)
                                           for x, r in zip(qs, ranked)) / len(qs), 3)
    return result


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        print(name, flush=True)
        results[name] = evaluate(name)
        print(results[name], flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
