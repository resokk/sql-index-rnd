"""Synthetic training pairs (Russian question -> SQL) generated from the logical model and checked on camunda_bench.

    uv run python bench/gen_train.py  ->  bench/results/train.jsonl

Every SQL must run and return rows. Pairs that touch the same tables and columns as a test question are dropped,
so the benchmark measures generalisation, not memorised answers.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import sqlglot
from sqlglot import exp

from corpus import model, questions, tables_of

DB = "camunda_bench"
OUT = Path(__file__).parent / "results" / "train.jsonl"

# Genitive plural: "Сколько <...>?"
NOUNS = {
    "PROCESS_INSTANCE": "экземпляров процессов", "PROCESS_DEFINITION": "определений процессов",
    "INCIDENT": "инцидентов", "USER_TASK": "пользовательских задач", "JOB": "заданий", "VARIABLE": "переменных",
    "FLOW_NODE_INSTANCE": "экземпляров элементов процесса", "DECISION_INSTANCE": "вычислений решений",
    "DECISION_DEFINITION": "определений решений", "BATCH_OPERATION": "пакетных операций",
    "BATCH_OPERATION_ITEM": "элементов пакетных операций", "AUDIT_LOG": "записей журнала аудита",
    "MESSAGE_SUBSCRIPTION": "подписок на сообщения", "USER_": "пользователей", "GROUP_": "групп",
    "TENANT": "тенантов", "FORM": "форм", "AUTHORIZATIONS": "разрешений", "AGENT_INSTANCE": "запусков ИИ-агентов",
    "JOB_METRICS_BATCH": "записей метрик заданий", "DEPLOYED_RESOURCE": "развёрнутых ресурсов",
    "CORRELATED_MESSAGE_SUBSCRIPTION": "сопоставленных сообщений", "WAIT_STATE": "состояний ожидания",
    "USAGE_METRIC": "метрик использования", "TENANT_MEMBER": "участников тенантов", "ROLE_MEMBER": "назначений ролей",
    "AGENT_HISTORY": "записей истории ИИ-агентов", "SEQUENCE_FLOW": "пройденных переходов",
}
# Genitive plural adjective for an enum value: "Сколько <активных> инцидентов?"
ADJ = {"ACTIVE": "активных", "COMPLETED": "завершённых", "CANCELED": "отменённых", "FAILED": "упавших",
       "CREATED": "созданных", "RESOLVED": "решённых", "TERMINATED": "прерванных", "CORRELATED": "сопоставленных",
       "DELETED": "удалённых", "MIGRATED": "мигрированных", "PENDING": "ожидающих", "RUNNING": "выполняющихся",
       "SUCCESS": "успешных", "FAIL": "неуспешных", "EVALUATED": "успешно вычисленных",
       "PARTIALLY_COMPLETED": "частично завершённых", "TIMED_OUT": "просроченных по таймауту"}
# Participle for a date column: "Сколько инцидентов, <созданных> за последние 10 дней?"
DATE_VERB = {"START_DATE": "начатых", "CREATION_DATE": "созданных", "CREATION_TIME": "созданных",
             "END_DATE": "завершённых", "COMPLETION_DATE": "завершённых", "END_TIME": "завершённых",
             "EVALUATION_DATE": "выполненных", "TIMESTAMP": "зарегистрированных", "DUE_DATE": "со сроком",
             "CORRELATION_TIME": "сопоставленных", "DATE_TIME": "созданных", "PROCESSED_DATE": "обработанных"}
GROUPABLE = {"STATE", "TYPE", "ERROR_TYPE", "TENANT_ID", "PROCESS_DEFINITION_ID", "ASSIGNEE", "WORKER", "JOB_TYPE",
             "MODEL", "ACTOR_ID", "OPERATION_TYPE", "MESSAGE_NAME", "ENTITY_TYPE", "RESOURCE_TYPE", "KIND", "RESULT",
             "MESSAGE_SUBSCRIPTION_STATE", "STATUS", "PERMISSION_TYPE", "OWNER_ID", "FLOW_NODE_ID", "ELEMENT_ID",
             "PRIORITY", "VAR_NAME", "PROVIDER", "CATEGORY", "VERSION"}
NUMERIC = {"INPUT_TOKENS", "OUTPUT_TOKENS", "RETRIES", "PRIORITY", "FAILED_COUNT", "COMPLETED_COUNT", "CREATED_COUNT",
           "OPERATIONS_FAILED_COUNT", "OPERATIONS_TOTAL_COUNT", "OPERATIONS_COMPLETED_COUNT", "NUM_INCIDENTS",
           "EVENT_VALUE", "DURATION_MS", "MODEL_CALLS", "TOOL_CALLS", "LONG_VALUE", "DOUBLE_VALUE"}
# Parent tables a child can be grouped by: (dative phrase, column to show)
PARENTS = {"PROCESS_DEFINITION": ("определению процесса", "NAME"), "DECISION_DEFINITION": ("решению", "NAME"),
           "GROUP_": ("группе", "NAME"), "FORM": ("форме", "FORM_ID"), "TENANT": ("тенанту", "NAME"),
           "BATCH_OPERATION": ("пакетной операции", "OPERATION_TYPE"), "PROCESS_INSTANCE": ("экземпляру процесса", "PROCESS_DEFINITION_ID")}
AGG = {"SUM": ("Сумма", "суммарное"), "AVG": ("Среднее", "среднее"), "MAX": ("Максимальное", "максимальное")}


def generate(rnd: random.Random) -> list[tuple[str, str]]:
    tables = model()
    pairs = []
    for t, noun in NOUNS.items():
        d = tables[t]
        cols = {c["name"]: c for c in d["columns"]}
        key = d["columns"][0]["name"]
        desc = lambda c: cols[c].get("description", c.lower().replace("_", " "))  # noqa: E731
        pairs.append((rnd.choice([f"Сколько всего {noun}?", f"Общее количество {noun}", f"Посчитай число {noun}"]),
                      f"SELECT COUNT(*) FROM {t}"))
        for c, col in cols.items():
            for v in col.get("values", []):
                adj = ADJ.get(v)
                label = f"{adj} {noun}" if adj else f"{noun} со значением {desc(c)} = {v}"
                pairs.append((rnd.choice([f"Сколько {label}?", f"Количество {label}"]),
                              f"SELECT COUNT(*) FROM {t} WHERE {c} = '{v}'"))
                pairs.append((rnd.choice([f"Выведи ключи {label}", f"Список {label} (только ключи)"]),
                              f"SELECT {key} FROM {t} WHERE {c} = '{v}'"))
            if c in GROUPABLE and c != key:
                d_ = desc(c)
                pairs.append((rnd.choice([f"Сколько {noun} в разрезе: {d_}?", f"Количество {noun} по каждому значению поля «{d_}»",
                                          f"Сгруппируй {noun} по {c} и посчитай количество"]),
                              f"SELECT {c}, COUNT(*) FROM {t} GROUP BY {c}"))
                k = rnd.choice([3, 5, 10])
                pairs.append((f"Топ-{k} значений «{d_}» по числу {noun}",
                              f"SELECT {c}, COUNT(*) AS cnt FROM {t} GROUP BY {c} ORDER BY cnt DESC LIMIT {k}"))
                if not col.get("values") and c not in ("PRIORITY", "VERSION"):
                    for sv in _sample_values(t, c, rnd, 1):
                        if rnd.random() < 0.5:
                            pairs.append((f"Сколько {noun} с {c} = '{sv}'?", f"SELECT COUNT(*) FROM {t} WHERE {c} = '{sv}'"))
                        else:
                            pairs.append((f"Покажи ключи {noun}, где {d_} = {sv}", f"SELECT {key} FROM {t} WHERE {c} = '{sv}'"))
            if c in DATE_VERB and "TIMESTAMP" in col["type"]:
                n = rnd.choice([3, 10, 14, 30])
                verb = DATE_VERB[c]
                pairs.append((f"Сколько {noun}, {verb} за последние {n} дней?",
                              f"SELECT COUNT(*) FROM {t} WHERE {c} >= NOW() - INTERVAL '{n} days'"))
                pairs.append((f"Количество {noun} по дням ({desc(c)})",
                              f"SELECT CAST({c} AS DATE), COUNT(*) FROM {t} GROUP BY CAST({c} AS DATE)"))
            if c in NUMERIC:
                f, (title, _) = rnd.choice(list(AGG.items()))
                pairs.append((f"{title} значение «{desc(c)}» по всем {noun.split()[0]}{' ' + ' '.join(noun.split()[1:]) if len(noun.split()) > 1 else ''}",
                              f"SELECT {f}({c}) FROM {t}"))
                g = rnd.choice([x for x in cols if x in GROUPABLE and x != c] or [None])
                if g:
                    pairs.append((f"{title} «{desc(c)}» для {noun} в разрезе: {desc(g)}",
                                  f"SELECT {g}, {f}({c}) FROM {t} GROUP BY {g}"))
        for r in d["relations"]:
            parent = r["table"]
            if parent not in PARENTS or parent == t:
                continue
            dat, show = PARENTS[parent]
            left, right = r["on"].split(" = ")
            on = f"c.{left.split('.')[1]} = p.{right.split('.')[1]}"
            pk = tables[parent]["columns"][0]["name"]
            pairs.append((f"Сколько {noun} по каждому {dat}? Выведи {show}",
                          f"SELECT p.{show}, COUNT(*) FROM {t} c JOIN {parent} p ON {on} GROUP BY p.{pk}, p.{show}"))
            state = cols.get("STATE")
            if state and state.get("values"):
                v = rnd.choice([x for x in state["values"] if x in ADJ] or state["values"])
                pairs.append((f"Сколько {ADJ.get(v, v)} {noun} по каждому {dat}? Выведи {show}",
                              f"SELECT p.{show}, COUNT(*) FROM {t} c JOIN {parent} p ON {on} WHERE c.STATE = '{v}' GROUP BY p.{pk}, p.{show}"))
    return pairs


def _sample_values(table: str, col: str, rnd: random.Random, n: int) -> list[str]:
    rows = _run(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")
    vals = [r[0] for r in rows or [] if r and "'" not in r[0]]
    return rnd.sample(vals, min(n, len(vals)))


def _run(sql: str) -> list[list[str]] | None:
    r = subprocess.run(["psql", "-d", DB, "-X", "-q", "-A", "-t", "-F", "\t", "-c", sql], capture_output=True, text=True)
    return None if r.returncode else [line.split("\t") for line in r.stdout.splitlines()]


def signature(sql: str) -> tuple[frozenset, frozenset]:
    tree = sqlglot.parse_one(sql, read="postgres")
    return frozenset(tables_of(sql)), frozenset(c.name.upper() for c in tree.find_all(exp.Column))


def main() -> None:
    rnd = random.Random(7)
    test = {signature(q["sql"]) for q in questions()}
    seen, kept, dropped_test, dropped_empty = set(), [], 0, 0
    for q, sql in generate(rnd):
        if sql in seen:
            continue
        seen.add(sql)
        if signature(sql) in test:
            dropped_test += 1
            continue
        if not _run(f"SELECT 1 FROM ({sql}) t LIMIT 1"):
            dropped_empty += 1
            continue
        kept.append({"q": q, "sql": sql})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
    print(f"{len(kept)} pairs -> {OUT}; dropped: {dropped_test} overlap with test, {dropped_empty} failing/empty")


if __name__ == "__main__":
    main()
