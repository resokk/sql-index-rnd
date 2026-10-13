"""Synthetic, reference-consistent data for the Camunda benchmark database (deterministic, seed 42).

    python bench/gen_data.py > data.sql   # dates are relative to today, so questions like "за неделю" have answers
"""

from __future__ import annotations

import json
import random
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODEL = Path(__file__).parent / "camunda" / "model"
ROWS = {"TENANT": 3, "PROCESS_DEFINITION": 12, "DECISION_REQUIREMENTS": 4, "DECISION_DEFINITION": 8, "FORM": 5,
        "USER_": 15, "GROUP_": 5, "ROLES": 4, "PROCESS_INSTANCE": 300, "FLOW_NODE_INSTANCE": 1500, "INCIDENT": 60,
        "USER_TASK": 200, "JOB": 400, "VARIABLE": 900, "DECISION_INSTANCE": 150}
DEFAULT_ROWS = 40
PROCESS_IDS = ["order-process", "invoice-approval", "loan-application", "onboarding", "claim-handling",
               "payment-retry", "shipment", "kyc-check"]
NAMES = ["Обработка заказа", "Согласование счёта", "Заявка на кредит", "Онбординг", "Урегулирование претензии",
         "Повтор платежа", "Доставка", "Проверка клиента"]
PEOPLE = ["ivanov", "petrova", "sidorov", "smirnova", "kuznetsov", "popova", "demo", "admin"]
TENANTS = ["<default>", "tenant-a", "tenant-b"]
WORKERS = ["payment-worker", "email-worker", "scoring-worker", "crm-worker"]
JOB_TYPES = ["charge-card", "send-email", "score-client", "update-crm"]
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def main() -> None:
    rnd = random.Random(42)
    docs = {d["table"]: d for d in (json.loads(p.read_text()) for p in MODEL.glob("*.json"))}
    keys: dict[tuple[str, str], list] = {}  # (table, column) -> generated values, for references
    out = []
    definitions: dict[int, str] = {}
    for table in _order(docs):
        doc = docs[table]
        refs = {r["on"].split(" = ")[0].split(".")[1]: tuple(r["on"].split(" = ")[1].split(".")) for r in doc["relations"]}
        n = ROWS.get(table, DEFAULT_ROWS)
        cols = [c["name"] for c in doc["columns"]]
        rows = []
        for i in range(n):
            row = {}
            for c in doc["columns"]:
                name = c["name"]
                ref = keys.get(refs.get(name))
                if ref and isinstance(ref[0], str) == _is_text(c["type"]):
                    row[name] = rnd.choice(ref)
                else:
                    row[name] = _value(rnd, table, c, i)
            _fix(rnd, table, row)
            if table != "PROCESS_DEFINITION" and row.get("PROCESS_DEFINITION_KEY") in definitions:
                row["PROCESS_DEFINITION_ID"] = definitions[row["PROCESS_DEFINITION_KEY"]]
            rows.append(row)
        for c in cols:
            keys[table, c] = [r[c] for r in rows]
        if table == "PROCESS_DEFINITION":
            definitions = {r["PROCESS_DEFINITION_KEY"]: r["PROCESS_DEFINITION_ID"] for r in rows}
        out.append(f"INSERT INTO {table} ({', '.join(cols)}) VALUES\n" +
                   ",\n".join("(" + ", ".join(_sql(r[c]) for c in cols) + ")" for r in rows) + ";")
    print("\n".join(out))


def _order(docs: dict) -> list[str]:
    done: list[str] = []
    def visit(t: str, path: tuple = ()) -> None:
        if t in done or t in path:
            return
        for r in docs[t]["relations"]:
            visit(r["table"], path + (t,))
        done.append(t)
    for t in sorted(docs):
        visit(t)
    return done


def _is_text(ty: str) -> bool:
    return ty.upper().startswith(("VARCHAR", "TEXT"))


def _value(rnd: random.Random, table: str, col: dict, i: int):
    name, ty = col["name"], col["type"].upper()
    if col.get("values"):
        return rnd.choice(col["values"])
    if name.endswith(("_KEY", "_ID")) and name.startswith(table.rstrip("_").removesuffix("S")):  # the table's own key
        return 1000 * (zlib.crc32(table.encode()) % 97 + 1) + i if not _is_text(ty) else f"{table.lower()}-{i}"
    if name == "TENANT_ID":
        return rnd.choice(TENANTS)
    if name in ("ASSIGNEE", "USERNAME", "ACTOR_ID", "OWNER_ID", "ENTITY_ID", "CANDIDATE_USER"):
        return rnd.choice(PEOPLE)
    if name in ("WORKER",):
        return rnd.choice(WORKERS)
    if name in ("JOB_TYPE",) or (table in ("JOB",) and name == "TYPE"):
        return rnd.choice(JOB_TYPES)
    if name == "PROCESS_DEFINITION_ID":
        return rnd.choice(PROCESS_IDS)
    if name == "CANDIDATE_GROUP":
        return rnd.choice(["accounting", "sales", "support", "risk"])
    if "TIMESTAMP" in ty or ty == "DATETIME" or (_is_text(ty) is False and ty.startswith("DATE")):
        return NOW - timedelta(days=rnd.uniform(0, 60))
    if ty.startswith(("BIGINT", "INT", "SMALLINT", "NUMBER")):
        return rnd.randint(0, 5) if name in ("RETRIES", "NUM_INCIDENTS", "PRIORITY", "VERSION") else rnd.randint(1, 1000)
    if ty.startswith("DOUBLE"):
        return round(rnd.uniform(0, 10000), 2)
    if ty == "BOOLEAN":
        return rnd.random() < 0.2
    if ty == "BYTEA":
        return None
    if name == "EMAIL":
        return f"user{i}@example.com"
    return f"{name.lower()}-{rnd.randint(1, 20)}"


def _fix(rnd: random.Random, table: str, row: dict) -> None:
    """Keeps rows plausible: finished things have end dates, versions start at 1, names are readable."""
    if table == "PROCESS_DEFINITION":
        k = rnd.randrange(len(PROCESS_IDS))
        row.update(PROCESS_DEFINITION_ID=PROCESS_IDS[k], NAME=NAMES[k], VERSION=rnd.randint(1, 4), STATE="ACTIVE")
    if "START_DATE" in row and "END_DATE" in row:
        row["START_DATE"] = NOW - timedelta(days=rnd.uniform(1, 60))
        active = row.get("STATE") in ("ACTIVE", "CREATED")
        row["END_DATE"] = None if active else row["START_DATE"] + timedelta(hours=rnd.uniform(0.1, 72))
    if table == "USER_TASK":
        row["COMPLETION_DATE"] = None if row["STATE"] == "CREATED" else row["CREATION_DATE"] + timedelta(hours=rnd.uniform(1, 48))
        row["DUE_DATE"] = row["CREATION_DATE"] + timedelta(days=rnd.randint(1, 10))
        row["ASSIGNEE"] = None if rnd.random() < 0.25 else row["ASSIGNEE"]
    if table == "INCIDENT":
        row["ERROR_MESSAGE"] = rnd.choice(["No more retries left", "Failed to evaluate expression", "Timeout calling service",
                                           "Variable 'amount' not found"])
    if table == "USER_":
        row["USERNAME"] = PEOPLE[row["USER_KEY"] % len(PEOPLE)] + str(row["USER_KEY"] % 100)
        row["NAME"] = row["USERNAME"].capitalize()


def _sql(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, datetime):
        return f"'{v.isoformat()}'"
    return "'" + str(v).replace("'", "''") + "'"


if __name__ == "__main__":
    main()
