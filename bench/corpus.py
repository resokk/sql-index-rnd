"""Benchmark corpus: index documents (logical model + mapper SQL) and questions with their gold tables."""

from __future__ import annotations

import json
from pathlib import Path

import sqlglot
from sqlglot import exp

DIR = Path(__file__).parent / "camunda"


def tables_of(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return set()
    ctes = {c.alias_or_name.upper() for c in tree.find_all(exp.CTE)}
    return {t.name.upper() for t in tree.find_all(exp.Table)} - ctes


def model() -> dict[str, dict]:
    return {d["table"]: d for d in (json.loads(p.read_text()) for p in sorted((DIR / "model").glob("*.json")))}


def table_text(d: dict) -> str:
    cols = ", ".join(c["name"] + (f" ({c['description']})" if c.get("description") else "") for c in d["columns"])
    return f"{d['entity']} ({d['table']}). {d['description']}. Колонки: {cols}"


def documents() -> list[dict]:
    """Index documents: one per table, one per mapper SELECT; `tables` says which tables a hit points to."""
    docs = [{"id": t, "kind": "table", "text": table_text(d), "tables": {t}} for t, d in model().items()]
    for s in json.loads((DIR / "statements.json").read_text()):
        if s["kind"] == "select":
            tables = tables_of(s["sql"]) & set(model())
            if tables:
                docs.append({"id": s["id"], "kind": "sql", "text": f"{s['id']}: {s['sql'][:1000]}", "tables": tables})
    return docs


def questions() -> list[dict]:
    qs = json.loads((DIR / "questions.json").read_text())
    for q in qs:
        q["tables"] = tables_of(q["sql"])
    return qs
