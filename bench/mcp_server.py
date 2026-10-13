"""MCP server (stdio) giving Claude retrieval over the Camunda corpus and static SQL checks; no LLM inside.

    uv run python bench/mcp_server.py

Tools search the same index the benchmark uses: tables of the logical model (JSON) and SELECTs from the code.
T2S_EMBEDDER (default intfloat/multilingual-e5-small) and T2S_HYBRID=1 (add FTS5 BM25) pick the search variant.
"""

from __future__ import annotations

import os

import sqlglot
from mcp.server.mcpserver import MCPServer
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

from corpus import model, tables_of
from eval_llm import render
from search import Index

TABLES = model()
# Query embeddings on CPU: one short query per call, and several servers may run at once.
INDEX = Index(os.environ.get("T2S_EMBEDDER", "intfloat/multilingual-e5-small"), os.environ.get("T2S_HYBRID") == "1",
              device="cpu")
# Column types for sqlglot's qualifier, so unknown tables and columns are reported.
SCHEMA = {t.lower(): {c["name"].lower(): "TEXT" for c in d["columns"]} for t, d in TABLES.items()}  # Postgres folds case

mcp = MCPServer(
    "text-to-sql",
    instructions=(
        "Database: Camunda 8 RDBMS schema in PostgreSQL. To answer a question with SQL: call search_schema with the "
        "question, look at similar queries from the application code with find_queries, describe_table for details, "
        "then check the statement with validate_sql before giving it."
    ),
)


@mcp.tool()
def search_schema(question: str, limit: int = 5) -> str:
    """Tables most relevant to a question (Russian or English): columns with descriptions, allowed enum values
    and join conditions between the returned tables."""
    return render(TABLES, INDEX.tables(question, limit), [])


@mcp.tool()
def find_queries(question: str, limit: int = 3) -> list[dict[str, str]]:
    """SELECT statements from the application code (MyBatis mappers) most similar to a question."""
    return [{"id": d["id"], "sql": d["text"].split(": ", 1)[1]} for d in INDEX.queries(question, limit)]


@mcp.tool()
def describe_table(table: str) -> str:
    """Full description of one table (exact name, e.g. PROCESS_INSTANCE)."""
    if table.upper() not in TABLES:
        raise ValueError(f"unknown table {table!r}; tables: {', '.join(sorted(TABLES))}")
    return render(TABLES, [table.upper()], [])


@mcp.tool()
def validate_sql(sql: str) -> dict[str, object]:
    """Checks a PostgreSQL statement without running it: it parses, is a single SELECT, and uses only existing
    tables and columns. Returns {"ok": true} or {"ok": false, "error": ...}."""
    try:
        trees = [t for t in sqlglot.parse(sql, read="postgres") if t is not None]
        if len(trees) != 1:
            return {"ok": False, "error": "expected exactly one statement"}
        tree = trees[0]
        if not isinstance(tree, exp.Query):
            return {"ok": False, "error": "only SELECT queries are allowed"}
        unknown = tables_of(sql) - set(TABLES)
        if unknown:
            return {"ok": False, "error": f"unknown tables: {', '.join(sorted(unknown))}"}
        qualify(tree, schema=SCHEMA, dialect="postgres", validate_qualify_columns=True, identify=False)
    except Exception as e:  # sqlglot reports both syntax and unknown names as exceptions
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True}


if __name__ == "__main__":
    mcp.run("stdio")
