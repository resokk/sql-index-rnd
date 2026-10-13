"""Text-to-SQL quality of local LLMs (transformers) on the Camunda questions, judged by executing on camunda_bench.

    uv run python bench/eval_llm.py <embedding-model> <llm>[+<lora-dir>] [...]  ->  bench/results/llm.json

<llm> is a Hugging Face model id; "+<lora-dir>" adds a LoRA adapter trained by finetune.py.

Context per question: the TOP_TABLES tables ranked best by the embedding model plus any gold table it missed (so the
score measures the LLM, not retrieval), their relations, and the EXAMPLES closest mapper SELECTs as examples.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from corpus import documents, model, questions, tables_of
from eval_embeddings import MODELS as EMBEDDERS, rank_tables
from hardware import pick

DB = "camunda_bench"
OUT = Path(__file__).parent / "results" / os.environ.get("BENCH_OUT", "llm.json")
TOP_TABLES = 3  # retrieved tables per prompt (gold tables are added if missed)
EXAMPLES = 2  # mapper SELECTs per prompt, cut to EXAMPLE_CHARS
EXAMPLE_CHARS = 500
SYSTEM = (
    "Ты эксперт по PostgreSQL. По описанию таблиц и вопросу на русском языке напиши один SQL-запрос SELECT, "
    "отвечающий на вопрос. Используй только перечисленные таблицы и колонки, значения перечислений бери из "
    "описания. Выводи только колонки, о которых спрашивают. Ответ — только SQL в блоке ```sql```, без пояснений."
)


def contexts(embedder: str, qs: list[dict]) -> list[str]:
    """Prompt context per question; each question needs "q" and its gold "tables"."""
    qp, dp, kwargs = EMBEDDERS[embedder]
    st = SentenceTransformer(embedder, device=str(pick()[0]), **kwargs)
    docs, tables = documents(), model()
    d = st.encode([dp + x["text"] for x in docs], normalize_embeddings=True, batch_size=16)
    q = st.encode([qp + x["q"] for x in qs], normalize_embeddings=True)
    out = []
    for i, x in enumerate(qs):
        sims = q[i] @ d.T
        chosen = rank_tables(sims, docs, {"table", "sql"})[:TOP_TABLES]
        chosen += sorted(x["tables"] - set(chosen))
        examples = [docs[j] for j in np.argsort(-sims) if docs[j]["kind"] == "sql"][:EXAMPLES]
        out.append(render(tables, chosen, examples))
    return out


def render(tables: dict[str, dict], chosen: list[str], examples: list[dict]) -> str:
    parts = []
    for t in chosen:
        d = tables[t]
        cols = []
        for c in d["columns"]:
            note = "; ".join(filter(None, [c.get("description"), c.get("values") and "значения: " + ", ".join(c["values"])]))
            cols.append(f"  {c['name']} {c['type']}" + (f" -- {note}" if note else ""))
        parts.append(f"-- {d['entity']}: {d['description']}\nCREATE TABLE {t} (\n" + ",\n".join(cols) + "\n);")
    joins = sorted({r["on"] for t in chosen for r in tables[t]["relations"] if r["table"] in chosen})
    if joins:
        parts.append("-- Связи:\n" + "\n".join(f"-- {j}" for j in joins))
    if examples:
        parts.append("-- Примеры запросов из кода приложения:\n" + "\n".join(f"-- {e['text'][:EXAMPLE_CHARS]}" for e in examples))
    return "\n\n".join(parts)


def messages(context: str, question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{context}\n\nВопрос: {question}"}]


# Models trained on their own prompt format instead of a chat template: prompt and the text that ends the answer.
RAW_PROMPTS = {
    "PipableAI/pip-sql-1.3b": ("<schema>{context}</schema>\n<question>{question}</question>\n<sql>", "</sql>"),
}


# Models whose tokenizer must be read from tokenizer.json as is: transformers 5 rebuilds their LlamaTokenizer as a
# SentencePiece one and drops the byte-level pre-tokenizer and decoder, which loses spaces in both directions.
FILE_TOKENIZER = {"PipableAI/pip-sql-1.3b"}


def prompt(name: str, tok, context: str, question: str) -> tuple[str, str | None]:
    """Model input text and the stop string (None: the chat template's end of turn)."""
    if name in RAW_PROMPTS:
        template, stop = RAW_PROMPTS[name]
        return template.format(context=context, question=question), stop
    return tok.apply_chat_template(messages(context, question), add_generation_prompt=True, tokenize=False), None


class LLM:
    """A causal LM (optionally with a LoRA adapter) on the best available device, greedy decoding."""

    def __init__(self, spec: str):
        name, _, adapter = spec.partition("+")
        self.name = name
        self.device, dtype = pick()
        self.tok = (PreTrainedTokenizerFast if name in FILE_TOKENIZER else AutoTokenizer).from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device)
        self.model = (PeftModel.from_pretrained(model, adapter) if adapter else model).eval()

    @torch.inference_mode()
    def ask(self, context: str, question: str) -> tuple[str, float]:
        text, stop = prompt(self.name, self.tok, context, question)
        # a chat template writes its own special tokens; a raw prompt needs the tokenizer's (BOS)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=bool(stop)).to(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**ids, max_new_tokens=384, do_sample=False, pad_token_id=self.tok.eos_token_id,
                                  stop_strings=stop, tokenizer=self.tok if stop else None)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if stop:
            text = text.split(stop)[0]
        return extract_sql(text), time.perf_counter() - t0


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


def run(sql: str) -> list[list[str]] | str:
    """Rows as strings (numbers rounded), or the error text."""
    r = subprocess.run(["psql", "-d", DB, "-X", "-q", "--csv", "-v", "ON_ERROR_STOP=1",
                        "-c", "SET statement_timeout = '10s'", "-c", sql], capture_output=True, text=True)
    if r.returncode:
        return r.stderr.strip().splitlines()[0] if r.stderr.strip() else "error"
    rows = list(csv.reader(io.StringIO(r.stdout)))[1:]
    return [[_norm(v) for v in row] for row in rows]


def _norm(v: str) -> str:
    try:
        return f"{float(v):.2f}"
    except ValueError:
        return v


def same_result(gold: list[list[str]], pred: list[list[str]]) -> bool:
    """Same number of rows and every gold column appears among the predicted columns (extra columns allowed)."""
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    pred_cols = [sorted(c) for c in zip(*pred)]
    return all(sorted(c) in pred_cols for c in zip(*gold))


def main() -> None:
    embedder, llms = sys.argv[1], sys.argv[2:]
    qs = questions()
    ctx = contexts(embedder, qs)
    gold = [run(q["sql"]) for q in qs]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for llm in llms:
        model = LLM(llm)
        model.ask("", "SELECT 1")  # warm-up, so the first question's latency is not one-off setup
        items, correct, valid, total_t = [], 0, 0, 0.0
        for q, c, g in zip(qs, ctx, gold):
            sql, t = model.ask(c, q["q"])
            res = run(sql)
            ok_exec = not isinstance(res, str)
            ok = ok_exec and same_result(g, res)
            valid += ok_exec
            correct += ok
            total_t += t
            items.append({"q": q["q"], "sql": sql, "ok": ok, "error": None if ok_exec else res,
                          "tables_ok": tables_of(sql) == q["tables"], "sec": round(t, 2)})
            print(f"{'+' if ok else '-'} {t:5.1f}s {q['q'][:60]} | {sql[:120]!r}", flush=True)
        results[llm] = {"embedder": embedder, "device": str(model.device), "accuracy": round(correct / len(qs), 3),
                        "executes": round(valid / len(qs), 3),
                        "tables_ok": round(sum(i["tables_ok"] for i in items) / len(qs), 3),
                        "avg_sec": round(total_t / len(qs), 2), "items": items}
        print(llm, {k: v for k, v in results[llm].items() if k != "items"}, flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
        del model


if __name__ == "__main__":
    main()
