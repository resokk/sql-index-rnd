"""Reads the Camunda 8 RDBMS schema (Liquibase changesets) and SQL statements (MyBatis mappers)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

LB = "{http://www.liquibase.org/xml/ns/dbchangelog}"
CHANGESETS = "db/rdbms-schema/src/main/resources/db/changelog/rdbms-exporter/changesets"
MAPPERS = "db/rdbms/src/main/resources/mapper"
# Liquibase properties as set for Postgres (vendor-properties/postgresql.properties).
SIZES = {"userCharColumnSize": "256", "errorMessageSize": "4000", "treePathSize": "8191"}
PG_TYPES = {"NUMBER": "BIGINT", "CLOB": "TEXT", "BLOB": "BYTEA", "DATETIME": "TIMESTAMP",
            "DOUBLE": "DOUBLE PRECISION", "TIMESTAMP WITH TIME ZONE(3)": "TIMESTAMP(3) WITH TIME ZONE"}


def _version(path: Path) -> tuple[int, ...]:
    return tuple(int(p) for p in path.stem.split("."))


def read_tables(repo: Path) -> dict[str, dict[str, str]]:
    """Table -> {column: Postgres type}, after applying all changesets in version order."""
    tables: dict[str, dict[str, str]] = {}
    for path in sorted((repo / CHANGESETS).glob("*.xml"), key=_version):
        for el in ET.parse(path).iter():
            if el.tag not in (LB + "createTable", LB + "addColumn", LB + "dropTable", LB + "dropColumn"):
                continue
            table = el.get("tableName").replace("${prefix}", "")
            if el.tag == LB + "dropTable":
                tables.pop(table, None)
            elif el.tag == LB + "dropColumn":
                tables[table].pop(el.get("columnName"), None)
            else:
                cols = tables.setdefault(table, {})
                for col in el.iter(LB + "column"):
                    cols[col.get("name")] = _pg_type(col.get("type"))
    return tables


def _pg_type(t: str) -> str:
    t = re.sub(r"\$\{(\w+)\}", lambda m: SIZES[m.group(1)], t)
    return PG_TYPES.get(t, t.replace("NVARCHAR", "VARCHAR"))


def read_statements(repo: Path) -> list[dict[str, str]]:
    """Mapper statements as plain SQL: includes inlined, optional (<if>, <foreach>, <choose>) parts dropped."""
    mappers = {}
    for path in sorted((repo / MAPPERS).glob("*.xml")):
        text = path.read_text()
        text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.S)
        root = ET.fromstring(text)
        mappers[root.get("namespace")] = root
    fragments = {f"{ns}.{el.get('id')}": el for ns, root in mappers.items() for el in root.iter("sql")}
    out = []
    for ns, root in mappers.items():
        for el in root:
            if el.tag not in ("select", "insert", "update", "delete"):
                continue
            sql = _flatten(el, ns, fragments)
            sql = re.sub(r"#\{[^}]*\}", "?", sql)
            sql = sql.replace("${prefix}", "")
            sql = re.sub(r"\$\{[^}]*\}", "", sql)
            sql = re.sub(r"\s+", " ", sql).strip()
            if sql:
                out.append({"id": f"{ns.rsplit('.', 1)[-1]}.{el.get('id')}", "kind": el.tag, "sql": sql})
    return out


def _flatten(el: ET.Element, ns: str, fragments: dict[str, ET.Element]) -> str:
    parts = [el.text or ""]
    for child in el:
        if child.tag == "include":
            ref = child.get("refid")
            frag = fragments.get(ref) or fragments.get(f"{ns}.{ref}")
            if frag is not None:
                parts.append(_flatten(frag, ns, fragments))
        elif child.tag == "where":
            inner = _flatten(child, ns, fragments).strip()
            if inner:
                parts.append(" WHERE " + re.sub(r"^(AND|OR)\s+", "", inner, flags=re.I))
        elif child.tag in ("set", "trim"):
            parts.append(_flatten(child, ns, fragments))
        # <if>, <foreach>, <choose>, <bind>: optional or repeated parts, left out
        parts.append(child.tail or "")
    return " ".join(parts)
