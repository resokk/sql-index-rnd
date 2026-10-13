"""Text-to-SQL quality of Claude (via `claude -p`) using the text-to-sql MCP server, on the Camunda questions.

    uv run python bench/eval_claude.py <embedder> <dense|hybrid> [model]  ->  bench/results/llm_claude.json

Unlike eval_llm.py, no context is given: Claude finds tables itself with the MCP tools (search_schema, find_queries,
describe_table, validate_sql). Built-in tools, user settings and hooks are off, so only those tools are available.
Answers are judged the same way: executed on camunda_bench and compared with the gold result.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus import questions, tables_of
from eval_llm import SYSTEM, extract_sql, run, same_result

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / ".venv" / "bin" / "python"
OUT = HERE / "results" / "llm_claude.json"
TOOLS = ["search_schema", "find_queries", "describe_table", "validate_sql"]
PARALLEL = 3


def ask(question: str, model: str | None, workdir: str, mcp_config: str) -> dict:
    cmd = ["claude", "-p", f"Вопрос: {question}", "--output-format", "json", "--no-session-persistence",
           "--append-system-prompt", SYSTEM + " Схему БД и примеры запросов ищи инструментами text-to-sql; "
           "перед ответом проверь запрос через validate_sql.",
           "--tools", "", "--strict-mcp-config", "--mcp-config", mcp_config, "--setting-sources", "project",
           "--allowedTools", *[f"mcp__t2s__{t}" for t in TOOLS]]
    if model:
        cmd += ["--model", model]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, timeout=600)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"text": "", "error": (r.stderr or r.stdout)[-500:]}
    return {"text": out.get("result", ""), "error": None if not out.get("is_error") else out.get("result"),
            "sec": out.get("duration_ms", 0) / 1000, "cost": out.get("total_cost_usd", 0),
            "turns": out.get("num_turns"), "models": list(out.get("modelUsage", {}))}


def main() -> None:
    embedder, mode = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else None
    label = f"{model or 'default'} | {embedder} {mode}"
    qs = questions()
    with tempfile.TemporaryDirectory() as workdir:  # empty cwd: no project CLAUDE.md or settings
        config = Path(workdir) / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"t2s": {
            "command": str(PYTHON), "args": [str(HERE / "mcp_server.py")], "env": {
            "HF_HUB_OFFLINE": "1", "T2S_EMBEDDER": embedder, "T2S_HYBRID": "1" if mode == "hybrid" else "0"}}}}))
        with ThreadPoolExecutor(PARALLEL) as pool:
            answers = list(pool.map(lambda q: ask(q["q"], model, workdir, str(config)), qs))
    items = []
    for q, a in zip(qs, answers):
        sql = extract_sql(a["text"]) if a["text"] else ""
        gold, res = run(q["sql"]), run(sql) if sql else (a["error"] or "no answer")
        ok_exec = not isinstance(res, str)
        ok = ok_exec and same_result(gold, res)
        items.append({"q": q["q"], "sql": sql, "ok": ok, "error": None if ok_exec else res,
                      "tables_ok": tables_of(sql) == q["tables"], **{k: a.get(k) for k in ("sec", "cost", "turns")}})
        print(f"{'+' if ok else '-'} {a.get('sec', 0):5.1f}s {q['q'][:60]} | {sql[:120]!r}", flush=True)
    n = len(qs)
    summary = {"models": sorted({m for a in answers for m in a.get("models", [])}),
               "accuracy": round(sum(i["ok"] for i in items) / n, 3),
               "executes": round(sum(i["error"] is None for i in items) / n, 3),
               "tables_ok": round(sum(i["tables_ok"] for i in items) / n, 3),
               "avg_sec": round(sum(i["sec"] or 0 for i in items) / n, 2),
               "avg_turns": round(sum(i["turns"] or 0 for i in items) / n, 1),
               "total_cost_usd": round(sum(i["cost"] or 0 for i in items), 3)}
    print(label, summary, flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[label] = {**summary, "items": items}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
