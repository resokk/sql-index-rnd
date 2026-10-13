# sql-index-rnd

Research for an MCP server that turns a question in Russian into SQL. The database is described by a logical model
(JSON per table) and by application code with native SQL; an index over both feeds the SQL generator. The benchmark
uses the Camunda 8 RDBMS layer: 51 tables from the Liquibase changesets, 388 statements from the MyBatis mappers.

## Layout

| Path | Contents |
|---|---|
| `bench/camunda_source.py`, `bench/build_camunda_model.py` | Camunda checkout → logical model (`camunda/model/*.json`, Russian descriptions, enum values, relations), mapper SQL (`statements.json`), Postgres DDL (`schema.sql`) |
| `bench/gen_data.py` | synthetic, reference-consistent data for the `camunda_bench` database |
| `bench/camunda/questions.json` | 40 Russian questions with gold SQL |
| `bench/corpus.py`, `bench/search.py` | index documents (tables + mapper SELECTs); dense or hybrid (embeddings + FTS5 BM25, RRF) search |
| `bench/eval_embeddings.py` | retrieval quality of embedding models |
| `bench/eval_llm.py` | local LLMs (transformers, CUDA / MPS / CPU via `hardware.py`), judged by executing on `camunda_bench` |
| `bench/gen_train.py`, `bench/finetune.py` | synthetic training pairs, LoRA fine-tuning (peft) |
| `bench/mcp_server.py`, `bench/eval_claude.py` | MCP server without an LLM (`search_schema`, `find_queries`, `describe_table`, `validate_sql`) and Claude via `claude -p` using it |
| `bench/results/` | all results (JSON), training pairs, LoRA adapters |

## Reproduce

```sh
uv sync
git clone --depth 1 https://github.com/camunda/camunda.git /tmp/camunda
uv run python bench/build_camunda_model.py /tmp/camunda
createdb camunda_bench && psql -d camunda_bench -f bench/camunda/schema.sql
(cd bench && uv run python gen_data.py | psql -q -d camunda_bench)

cd bench
uv run python eval_embeddings.py                                     # embeddings
uv run python search.py                                              # dense vs hybrid
uv run python gen_train.py                                           # training pairs
uv run python finetune.py Qwen/Qwen2.5-Coder-0.5B-Instruct results/lora-coder-0.5b
BENCH_OUT=llm_hf.json uv run python eval_llm.py intfloat/multilingual-e5-small \
    Qwen/Qwen2.5-Coder-0.5B-Instruct Qwen/Qwen2.5-Coder-0.5B-Instruct+results/lora-coder-0.5b
uv run python eval_claude.py intfloat/multilingual-e5-small dense    # needs the `claude` CLI
```

Data dates are relative to the day `gen_data.py` runs; gold and generated SQL are compared on the same data.
A generated query is correct when it runs and returns the gold rows (extra columns allowed).

## Results

Hardware: Apple M4, 16 GB. Questions: 40.

### Embeddings (`results/embeddings.json`)

All gold tables in the top 5, searching tables of the logical model plus mapper SELECTs:

| Model | @5 | MRR | Query, ms |
|---|---|---|---|
| BAAI/bge-m3 | **1.0** | **0.89** | 66 |
| intfloat/multilingual-e5-small | 0.975 | 0.77 | 2.5 |
| ai-forever/ru-en-RoSBERTa | 0.975 | 0.78 | 19 |
| deepvk/USER-bge-m3 | 0.975 | 0.87 | 41 |
| intfloat/multilingual-e5-base | 0.95 | 0.78 | 17 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.825 | 0.67 | 7 |

Indexing SQL from the code helps every model. Not evaluated: `gte-multilingual-base` (its remote code fails on
transformers 5) and `Qwen3-Embedding-0.6B` (the run stopped at gte).

Hybrid search (`search.py`): e5-small @1 0.45 → 0.525, @5 0.975 → 0.95; bge-m3 @5 1.0 → 0.95, MRR 0.89 → 0.90.
BM25 favours short link tables; it helps on exact names (`order-process`, `demo`), which this set rarely has.

### Local LLMs, transformers on MPS (`results/llm_hf.json`)

Prompt: 3 retrieved tables plus any gold table the retriever missed, relations, 2 mapper SELECTs (~1100 tokens).

| Model | Correct | Runs | Tables | s / question |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 20% | 72.5% | 62.5% | 1.4 |
| **Qwen2.5-Coder-0.5B-Instruct + LoRA** | **52.5%** | 80% | 80% | 1.7 |
| Qwen2.5-Coder-1.5B-Instruct | 42.5% | 90% | 70% | 4.9 |
| Qwen2.5-Coder-1.5B-Instruct + LoRA | 47.5% | 87.5% | 85% | 4.4 |
| PipableAI/pip-sql-1.3b | 32.5% | 77.5% | 80% | 5.4 |

LoRA: one epoch over 570 synthetic pairs (`results/train.jsonl`, pairs with the same tables and columns as a test
question removed), r=16, all linear layers; 51 min for 0.5B, 2 h for 1.5B. Remaining errors come from the uniform
templates: copied join aliases, single-condition queries, `SELECT *` for lists.

Earlier stage through Ollama (Q4, other context size, `results/llm.json`, `results/llm_small.json`): qwen2.5-coder
0.5b 15%, 1.5b 45%, 3b 55%.

### Claude + RAG over MCP (`results/llm_claude.json`)

`claude -p`, Claude Opus 5.5, only the four `mcp_server.py` tools; no gold tables are given, Claude searches itself.

| Search | Correct | Tables | Turns | s / question | $ / 40 |
|---|---|---|---|---|---|
| e5-small dense | **95%** | 97.5% | 3.2 | 5.6 | 1.50 |
| e5-small hybrid | 90% | 97.5% | 3.2 | 5.7 | 1.45 |
| bge-m3 dense | 92.5% | 97.5% | 3.1 | 5.6 | 1.51 |
| bge-m3 hybrid | 92.5% | 97.5% | 3.1 | 5.8 | 1.38 |

The misses are the same few disputable cases in every run (`COUNT(DISTINCT)` on duplicated synthetic pairs, an extra
`OWNER_TYPE = 'USER'` filter on random synthetic values, `DATE_TRUNC` vs a date, excluding tasks without assignee),
not retrieval failures: the set is saturated for Claude.

## Conclusions

- Claude with retrieval over MCP: 95% against 52.5% for the best local model, at ~$0.04 and ~6 s per question;
  questions, schema descriptions and code fragments leave the network.
- Fully local: Qwen2.5-Coder-0.5B + LoRA with e5-small; better training data diversity is the next lever.
- Index: e5-small is enough for 51 tables; bge-m3 and hybrid search need a larger schema and harder questions
  (multi-hop joins, business terms, exact values) to show a difference.
