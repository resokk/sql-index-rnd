"""Builds the benchmark corpus from a Camunda checkout: logical model (JSON per table), mapper SQL, Postgres DDL.

    python bench/build_camunda_model.py <camunda-repo>  ->  bench/camunda/{model/*.json, statements.json, schema.sql}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from camunda_source import read_statements, read_tables

OUT = Path(__file__).parent / "camunda"

# Logical names and descriptions, as an analyst would write them.
TABLES = {
    "EXPORTER_POSITION": ("Позиция экспортёра", "Последняя выгруженная позиция журнала по каждой партиции"),
    "PROCESS_DEFINITION": ("Определение процесса", "Развёрнутая BPMN-модель процесса: версия, имя, XML"),
    "PROCESS_INSTANCE": ("Экземпляр процесса", "Запуск процесса: состояние, даты старта и завершения, число инцидентов"),
    "PROCESS_INSTANCE_TAG": ("Тег экземпляра процесса", "Метки, присвоенные экземплярам процессов"),
    "VARIABLE": ("Переменная процесса", "Переменные экземпляров процессов: имя, тип, значение"),
    "FLOW_NODE_INSTANCE": ("Экземпляр элемента процесса", "Выполнение шага (узла BPMN) в экземпляре процесса"),
    "USER_TASK": ("Пользовательская задача", "Задача для человека: исполнитель, срок, приоритет, состояние"),
    "CANDIDATE_USER": ("Кандидат-пользователь задачи", "Пользователи, которые могут взять задачу"),
    "CANDIDATE_GROUP": ("Кандидат-группа задачи", "Группы, которые могут взять задачу"),
    "USER_TASK_TAG": ("Тег пользовательской задачи", "Метки пользовательских задач"),
    "INCIDENT": ("Инцидент", "Ошибка при выполнении процесса, требующая вмешательства"),
    "DECISION_DEFINITION": ("Определение решения", "DMN-решение (таблица решений): версия, имя"),
    "DECISION_REQUIREMENTS": ("Модель требований решений", "DRD/DMN-файл, содержащий решения"),
    "DECISION_INSTANCE": ("Вычисление решения", "Результат вычисления DMN-решения"),
    "DECISION_INSTANCE_INPUT": ("Вход вычисления решения", "Входные значения вычисления решения"),
    "DECISION_INSTANCE_OUTPUT": ("Выход вычисления решения", "Выходные значения и сработавшие правила решения"),
    "USER_": ("Пользователь", "Учётная запись пользователя: логин, имя, email"),
    "FORM": ("Форма", "Форма пользовательской задачи: схема, версия"),
    "TENANT": ("Арендатор (тенант)", "Тенант — изолированное пространство данных"),
    "TENANT_MEMBER": ("Участник тенанта", "Пользователи, группы и др., входящие в тенант"),
    "ROLES": ("Роль", "Роль доступа"),
    "ROLE_MEMBER": ("Участник роли", "Кому назначена роль"),
    "GROUP_": ("Группа", "Группа пользователей"),
    "GROUP_MEMBER": ("Участник группы", "Членство в группе"),
    "AUTHORIZATIONS": ("Разрешение", "Права доступа владельца на ресурс"),
    "BATCH_OPERATION": ("Пакетная операция", "Массовая операция над экземплярами (отмена, миграция и т.п.)"),
    "BATCH_OPERATION_ITEM": ("Элемент пакетной операции", "Отдельный объект, обрабатываемый пакетной операцией"),
    "BATCH_OPERATION_ERROR": ("Ошибка пакетной операции", "Ошибки пакетной операции по партициям"),
    "JOB": ("Задание (джоб)", "Задание для воркера: тип, воркер, повторы, состояние, ошибка"),
    "SEQUENCE_FLOW": ("Пройденный переход", "Пройденные стрелки (sequence flow) в экземпляре процесса"),
    "USAGE_METRIC": ("Метрика использования", "Счётчики использования (запуски процессов, вычисления решений)"),
    "USAGE_METRIC_TU": ("Метрика задач по исполнителям", "Уникальные исполнители задач за период"),
    "MAPPING_RULES": ("Правило сопоставления", "Сопоставление claim из токена OIDC с сущностями"),
    "MESSAGE_SUBSCRIPTION": ("Подписка на сообщение", "Ожидание сообщения процессом"),
    "CORRELATED_MESSAGE_SUBSCRIPTION": ("Сопоставленное сообщение", "Сообщения, доставленные в подписки"),
    "CLUSTER_VARIABLE": ("Переменная кластера", "Глобальные переменные кластера или тенанта"),
    "AUDIT_LOG": ("Журнал аудита", "Кто, когда и какую операцию выполнил над сущностью"),
    "HISTORY_DELETION": ("Удаление истории", "Очередь удаления исторических данных"),
    "WEB_SESSION": ("Веб-сессия", "Сессии пользователей веб-приложения"),
    "JOB_METRICS_BATCH": ("Метрики заданий", "Счётчики созданных, завершённых и упавших заданий по типу и воркеру"),
    "GLOBAL_LISTENER": ("Глобальный слушатель", "Слушатели событий, применяемые ко всем процессам"),
    "GLOBAL_LISTENER_EVENT_TYPE": ("Событие глобального слушателя", "Типы событий глобального слушателя"),
    "DEPLOYED_RESOURCE": ("Развёрнутый ресурс", "Файл, развёрнутый в кластер (BPMN, DMN, форма)"),
    "RDBMS_SCHEMA_VERSION": ("Версия схемы", "Версия схемы БД"),
    "AGENT_INSTANCE": ("Экземпляр ИИ-агента", "Запуск ИИ-агента в процессе: модель, токены, вызовы"),
    "AGENT_INSTANCE_ELEMENT_INSTANCE": ("Шаг ИИ-агента", "Связь агента с экземплярами элементов процесса"),
    "AGENT_DEFINITION": ("Определение ИИ-агента", "ИИ-агент, описанный в процессе"),
    "WAIT_STATE": ("Состояние ожидания", "Где и чего ждёт экземпляр процесса"),
    "PROCESS_DEF_VAR_NAME_LOOKUP": ("Имена переменных процесса", "Справочник имён переменных по определению процесса"),
    "AGENT_HISTORY": ("История ИИ-агента", "Сообщения и вызовы ИИ-агента"),
    "CLUSTER_VARIABLE_METADATA": ("Метаданные переменной кластера", "Ключ-значение метаданных переменной кластера"),
}

COLUMNS = {
    "TENANT_ID": "идентификатор тенанта", "PARTITION_ID": "номер партиции",
    "PROCESS_INSTANCE_KEY": "ключ экземпляра процесса",
    "ROOT_PROCESS_INSTANCE_KEY": "ключ корневого экземпляра процесса",
    "PARENT_PROCESS_INSTANCE_KEY": "ключ родительского экземпляра процесса",
    "PROCESS_DEFINITION_KEY": "ключ определения процесса",
    "PROCESS_DEFINITION_ID": "идентификатор процесса (BPMN id)",
    "PROCESS_DEFINITION_VERSION": "версия определения процесса",
    "NAME": "наименование", "DESCRIPTION": "описание", "VERSION": "номер версии", "VERSION_TAG": "метка версии",
    "STATE": "состояние", "START_DATE": "дата начала", "END_DATE": "дата окончания",
    "CREATION_DATE": "дата создания", "COMPLETION_DATE": "дата завершения", "DUE_DATE": "срок выполнения",
    "FOLLOW_UP_DATE": "дата напоминания", "ASSIGNEE": "исполнитель (логин)", "PRIORITY": "приоритет",
    "ERROR_MESSAGE": "текст ошибки", "ERROR_TYPE": "тип ошибки", "ERROR_CODE": "код ошибки",
    "FLOW_NODE_ID": "идентификатор элемента BPMN", "FLOW_NODE_NAME": "название элемента BPMN",
    "ELEMENT_ID": "идентификатор элемента BPMN", "ELEMENT_INSTANCE_KEY": "ключ экземпляра элемента",
    "FLOW_NODE_INSTANCE_KEY": "ключ экземпляра элемента", "INCIDENT_KEY": "ключ инцидента",
    "NUM_INCIDENTS": "число инцидентов", "JOB_KEY": "ключ задания", "USER_TASK_KEY": "ключ пользовательской задачи",
    "WORKER": "имя воркера", "RETRIES": "оставшиеся повторы", "DEADLINE": "крайний срок",
    "TYPE": "тип", "VAR_NAME": "имя переменной", "VAR_VALUE": "значение переменной (превью)",
    "USERNAME": "логин", "EMAIL": "электронная почта", "BUSINESS_ID": "бизнес-идентификатор",
    "EVALUATION_DATE": "дата вычисления", "EVALUATION_FAILURE_MESSAGE": "текст ошибки вычисления",
    "DECISION_DEFINITION_KEY": "ключ определения решения", "DECISION_DEFINITION_ID": "идентификатор решения",
    "DECISION_REQUIREMENTS_KEY": "ключ модели требований", "DECISION_INSTANCE_ID": "идентификатор вычисления решения",
    "BATCH_OPERATION_KEY": "ключ пакетной операции", "OPERATION_TYPE": "тип операции",
    "OPERATIONS_TOTAL_COUNT": "всего элементов", "OPERATIONS_FAILED_COUNT": "упавших элементов",
    "OPERATIONS_COMPLETED_COUNT": "завершённых элементов", "ACTOR_ID": "кто выполнил", "ACTOR_TYPE": "тип исполнителя",
    "MESSAGE_NAME": "имя сообщения", "CORRELATION_KEY": "ключ корреляции", "TIMESTAMP": "время события",
    "ENTITY_TYPE": "тип сущности", "ENTITY_ID": "идентификатор участника", "OWNER_ID": "владелец разрешения",
    "PERMISSION_TYPE": "тип разрешения", "RESOURCE_TYPE": "тип ресурса", "RESOURCE_ID": "идентификатор ресурса",
    "CANDIDATE_GROUP": "группа-кандидат", "CANDIDATE_USER": "пользователь-кандидат", "TAG_VALUE": "значение тега",
    "INPUT_TOKENS": "входные токены", "OUTPUT_TOKENS": "выходные токены", "MODEL": "модель LLM",
    "STATUS": "статус", "JOB_TYPE": "тип задания", "FAILED_COUNT": "число упавших", "COMPLETED_COUNT": "число завершённых",
    "CREATED_COUNT": "число созданных", "CREATION_TIME": "время создания", "END_TIME": "время окончания",
    "START_TIME": "начало периода", "EVENT_TYPE": "тип события", "EVENT_VALUE": "значение счётчика",
    "GROUP_ID": "идентификатор группы", "ROLE_ID": "идентификатор роли", "FORM_KEY": "ключ формы",
    "RESOURCE_NAME": "имя файла ресурса", "DEPLOYMENT_KEY": "ключ развёртывания", "SUSPENDED_DATE": "дата приостановки",
    "IS_SUSPENDED": "задача приостановлена", "KIND": "вид", "RESULT": "результат", "DURATION_MS": "длительность, мс",
}

VALUES = {
    ("PROCESS_INSTANCE", "STATE"): ["ACTIVE", "COMPLETED", "CANCELED"],
    ("PROCESS_DEFINITION", "STATE"): ["ACTIVE", "DELETED"],
    ("FLOW_NODE_INSTANCE", "STATE"): ["ACTIVE", "COMPLETED", "TERMINATED"],
    ("FLOW_NODE_INSTANCE", "TYPE"): ["START_EVENT", "END_EVENT", "SERVICE_TASK", "USER_TASK", "EXCLUSIVE_GATEWAY",
                                     "CALL_ACTIVITY", "SUB_PROCESS", "INTERMEDIATE_CATCH_EVENT"],
    ("USER_TASK", "STATE"): ["CREATED", "COMPLETED", "CANCELED", "FAILED"],
    ("INCIDENT", "STATE"): ["ACTIVE", "RESOLVED", "MIGRATED", "PENDING"],
    ("INCIDENT", "ERROR_TYPE"): ["JOB_NO_RETRIES", "EXTRACT_VALUE_ERROR", "CALLED_ELEMENT_ERROR", "UNHANDLED_ERROR_EVENT",
                                 "DECISION_EVALUATION_ERROR", "IO_MAPPING_ERROR"],
    ("JOB", "STATE"): ["CREATED", "COMPLETED", "FAILED", "RETRIES_UPDATED", "TIMED_OUT", "CANCELED", "ERROR_THROWN"],
    ("JOB", "KIND"): ["BPMN_ELEMENT", "EXECUTION_LISTENER", "TASK_LISTENER"],
    ("DECISION_INSTANCE", "STATE"): ["EVALUATED", "FAILED"],
    ("BATCH_OPERATION", "STATE"): ["CREATED", "ACTIVE", "COMPLETED", "PARTIALLY_COMPLETED", "CANCELED", "FAILED"],
    ("BATCH_OPERATION", "OPERATION_TYPE"): ["CANCEL_PROCESS_INSTANCE", "MIGRATE_PROCESS_INSTANCE",
                                           "MODIFY_PROCESS_INSTANCE", "RESOLVE_INCIDENT", "DELETE_PROCESS_INSTANCE"],
    ("BATCH_OPERATION_ITEM", "STATE"): ["ACTIVE", "COMPLETED", "CANCELED", "FAILED"],
    ("MESSAGE_SUBSCRIPTION", "MESSAGE_SUBSCRIPTION_STATE"): ["CREATED", "CORRELATED", "DELETED", "MIGRATED"],
    ("VARIABLE", "TYPE"): ["STRING", "LONG", "DOUBLE", "BOOLEAN", "JSON"],
    ("AUDIT_LOG", "RESULT"): ["SUCCESS", "FAIL"],
    ("TENANT_MEMBER", "ENTITY_TYPE"): ["USER", "GROUP", "ROLE", "CLIENT", "MAPPING_RULE"],
    ("GROUP_MEMBER", "ENTITY_TYPE"): ["USER", "CLIENT", "MAPPING_RULE"],
    ("ROLE_MEMBER", "ENTITY_TYPE"): ["USER", "GROUP", "CLIENT", "MAPPING_RULE"],
    ("AGENT_INSTANCE", "STATUS"): ["RUNNING", "COMPLETED", "FAILED"],
}


def build(repo: Path) -> None:
    tables = read_tables(repo)
    # The first column ending in _KEY (or the first column) identifies a row; other tables refer to it by name.
    pk = {t: next((c for c in cols if c.endswith("_KEY")), next(iter(cols))) for t, cols in tables.items()}
    owners = {}
    for t, col in pk.items():
        owners.setdefault(col, t)
    for col, t in [("PROCESS_INSTANCE_KEY", "PROCESS_INSTANCE"), ("FLOW_NODE_INSTANCE_KEY", "FLOW_NODE_INSTANCE"),
                   ("ELEMENT_INSTANCE_KEY", "FLOW_NODE_INSTANCE"), ("ROOT_PROCESS_INSTANCE_KEY", "PROCESS_INSTANCE"),
                   ("PARENT_PROCESS_INSTANCE_KEY", "PROCESS_INSTANCE"), ("DECISION_INSTANCE_ID", "DECISION_INSTANCE"),
                   ("GROUP_ID", "GROUP_"), ("ROLE_ID", "ROLES"), ("TENANT_ID", "TENANT"),
                   ("GLOBAL_LISTENER_ID", "GLOBAL_LISTENER"), ("CLUSTER_VARIABLE_ID", "CLUSTER_VARIABLE")]:
        owners[col] = t
    for col in ("PARTITION_ID", "CORRELATION_KEY", "USAGE_METRIC_KEY"):  # shared names, not references
        owners.pop(col, None)
    owners["RESOURCE_KEY"] = "DEPLOYED_RESOURCE"
    target_col = {"ELEMENT_INSTANCE_KEY": "FLOW_NODE_INSTANCE_KEY", "ROOT_PROCESS_INSTANCE_KEY": "PROCESS_INSTANCE_KEY",
                  "PARENT_PROCESS_INSTANCE_KEY": "PROCESS_INSTANCE_KEY", "GLOBAL_LISTENER_ID": "ID"}

    model_dir = OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    ddl = []
    for t, cols in tables.items():
        entity, description = TABLES[t]
        relations = []
        for c in cols:
            owner = owners.get(c)
            if owner and owner != t:
                relations.append({"table": owner, "on": f"{t}.{c} = {owner}.{target_col.get(c, c)}"})
        doc = {
            "table": t, "entity": entity, "description": description,
            "columns": [{"name": c, "type": ty, **({"description": COLUMNS[c]} if c in COLUMNS else {}),
                         **({"values": VALUES[t, c]} if (t, c) in VALUES else {})} for c, ty in cols.items()],
            "relations": relations,
        }
        (model_dir / f"{t.lower()}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        ddl.append(f"CREATE TABLE {t} (\n  " + ",\n  ".join(f"{c} {ty}" for c, ty in cols.items()) + "\n);")
    (OUT / "schema.sql").write_text("\n\n".join(ddl) + "\n")
    stmts = read_statements(repo)
    (OUT / "statements.json").write_text(json.dumps(stmts, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(tables)} tables, {len(stmts)} statements -> {OUT}")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
