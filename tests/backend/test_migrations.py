"""
Alembic migration chain — regression tests for the "DuplicateColumn on a fresh
database" defect.

0001 and 0002 used to call `Base.metadata.create_all()`, which builds the
schema from the models as they are *today*; every column a later revision
added was therefore already there on a fresh database and `alembic upgrade
head` failed in 0002 (`components.cad_node_id`) and 0003 (`models_3d.scale`).

The first group of tests needs no database: it renders the whole upgrade as
SQL (Alembic's offline mode) and checks that every table, column and type is
created exactly once and that the result is the schema the models describe.
The second group runs real DDL and is skipped unless
MIGRATION_TEST_DATABASE_URL points at a PostgreSQL server; each test creates
and drops its own database there, so it never touches existing data.
"""

import ast
import io
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import make_url

import app.models  # noqa: F401 - populates Base.metadata
from app.core.config import get_settings
from app.core.database import Base

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
VERSIONS_DIR = BACKEND_DIR / "alembic" / "versions"

# Tables each revision creates. Downgrading a revision must remove these and
# nothing else.
TABLES_BY_REVISION = {
    "0001": {
        "assets", "components", "states", "procedures", "procedure_revisions", "steps", "validation_rules",
        "inspection_runs", "inspection_steps", "observations", "evidence", "users", "audit_logs",
    },
    "0002": {"models_3d"},
    "0003": set(),
    "0004": set(),
    "0005": {"marker_tags"},
}


def _alembic_config(url: str, output: io.StringIO | None = None) -> Config:
    # No ini file: env.py then skips logging.fileConfig, which would otherwise
    # disable the loggers of every test that runs after this one.
    cfg = Config(output_buffer=output) if output is not None else Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def _revisions_base_to_head() -> list[str]:
    script = ScriptDirectory.from_config(_alembic_config("postgresql+psycopg2://"))
    return [rev.revision for rev in reversed(list(script.walk_revisions()))]


# ---------------------------------------------------------------- no database


def test_history_is_one_line_with_one_head():
    script = ScriptDirectory.from_config(_alembic_config("postgresql+psycopg2://"))
    assert script.get_heads() == ["0005"]
    assert script.get_bases() == ["0001"]
    assert _revisions_base_to_head() == sorted(TABLES_BY_REVISION)


@pytest.mark.parametrize("path", sorted(VERSIONS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_migration_does_not_build_schema_from_live_models(path: Path):
    """A migration that imports the models or calls create_all/drop_all
    applies today's schema instead of the one it was written for."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            imported = []
        assert not [name for name in imported if name == "app" or name.startswith("app.")], (
            f"{path.name} imports application code"
        )
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"create_all", "drop_all"}, f"{path.name} calls {node.attr}()"


def _offline_upgrade_sql() -> str:
    output = io.StringIO()
    command.upgrade(_alembic_config("postgresql+psycopg2://", output), "head", sql=True)
    return output.getvalue()


def _columns_from_sql(sql: str) -> tuple[dict[str, list[str]], list[str]]:
    """({table: [column, ...]}, [enum type, ...]) in the order the SQL creates
    them, duplicates kept so the caller can detect them."""
    columns: dict[str, list[str]] = {}
    for table, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\n\);", sql, flags=re.S):
        assert table not in columns or table == "alembic_version", f"{table} is created twice"
        names = [m.group(1) for line in body.splitlines() if (m := re.match(r"\s+(\w+) ", line))]
        columns[table] = [n for n in names if n not in {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}]
    for table, column in re.findall(r"ALTER TABLE (\w+) ADD COLUMN (\w+) ", sql):
        columns[table].append(column)
    types = re.findall(r"CREATE TYPE (\w+) AS ENUM", sql)
    return columns, types


def test_upgrade_creates_every_table_column_and_type_exactly_once():
    columns, types = _columns_from_sql(_offline_upgrade_sql())
    for table, names in columns.items():
        duplicated = sorted({n for n in names if names.count(n) > 1})
        assert not duplicated, f"{table}: created more than once: {duplicated}"
    assert len(types) == len(set(types)), f"enum type created more than once: {types}"


def test_upgraded_schema_has_the_tables_and_columns_of_the_models():
    columns, _ = _columns_from_sql(_offline_upgrade_sql())
    columns.pop("alembic_version")
    expected = {table.name: {c.name for c in table.columns} for table in Base.metadata.sorted_tables}
    assert {table: set(names) for table, names in columns.items()} == expected


# -------------------------------------------------------------- real database

_server_url = get_settings().migration_test_database_url
needs_database = pytest.mark.skipif(
    not _server_url, reason="set MIGRATION_TEST_DATABASE_URL to a PostgreSQL server to run real migrations"
)


@pytest.fixture
def database_url():
    """A brand-new, empty database on the test server, dropped afterwards."""
    server = make_url(_server_url)
    name = f"tvasta_migtest_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    try:
        yield server.set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _inspect(url: str, fn):
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return fn(conn)
    finally:
        engine.dispose()


def _tables(url: str) -> set[str]:
    return _inspect(url, lambda c: set(sa.inspect(c).get_table_names()) - {"alembic_version"})


def _enum_types(url: str) -> set[str]:
    return _inspect(url, lambda c: {e["name"] for e in sa.inspect(c).get_enums()})


def _current(url: str) -> str | None:
    return _inspect(url, lambda c: MigrationContext.configure(c).get_current_revision())


def _execute(url: str, statement: str, **params) -> list:
    engine = sa.create_engine(url)
    try:
        with engine.begin() as conn:
            result = conn.execute(sa.text(statement), params)
            return list(result) if result.returns_rows else []
    finally:
        engine.dispose()


@needs_database
def test_fresh_database_upgrades_to_head_and_matches_the_models(database_url):
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")
    assert _current(database_url) == "0005"
    assert _tables(database_url) == set().union(*TABLES_BY_REVISION.values())

    # What `alembic check` runs: no difference between the migrated schema
    # and Base.metadata (tables, columns, types, nullability, FKs, uniques).
    diff = _inspect(
        database_url,
        lambda c: compare_metadata(MigrationContext.configure(c, opts={"compare_type": True}), Base.metadata),
    )
    assert diff == []

    command.upgrade(cfg, "head")  # second run: nothing left to apply
    assert _current(database_url) == "0005"


@needs_database
def test_existing_data_survives_the_upgrade_and_new_columns_get_their_defaults(database_url):
    cfg = _alembic_config(database_url)
    asset_id, component_id, model_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    command.upgrade(cfg, "0001")
    _execute(database_url, "INSERT INTO assets (id, name, created_at) VALUES (:id, 'Pump', now())", id=asset_id)
    _execute(
        database_url,
        "INSERT INTO components (id, asset_id, component_id_str, name, class_label) "
        "VALUES (:id, :asset, 'PCB-1', 'PCB', 'pcb')",
        id=component_id,
        asset=asset_id,
    )
    command.upgrade(cfg, "0002")
    _execute(
        database_url,
        "INSERT INTO models_3d (id, asset_id, name, format, storage_key, created_at) "
        "VALUES (:id, :asset, 'pump', 'glb', 'pump.glb', now())",
        id=model_id,
        asset=asset_id,
    )
    command.upgrade(cfg, "head")

    assert _execute(database_url, "SELECT name, cad_node_id FROM components") == [("PCB", None)]
    assert _execute(
        database_url, "SELECT name, scale, anchor_offset_x, anchor_rotation_w FROM models_3d"
    ) == [("pump", 1.0, 0.0, 1.0)]


@needs_database
def test_each_downgrade_removes_only_what_its_revision_created(database_url):
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "0001")
    asset_id = uuid.uuid4()
    _execute(database_url, "INSERT INTO assets (id, name, created_at) VALUES (:id, 'Pump', now())", id=asset_id)
    command.upgrade(cfg, "head")

    revisions = _revisions_base_to_head()
    remaining = set().union(*TABLES_BY_REVISION.values())
    for revision, previous in zip(reversed(revisions[1:]), reversed(revisions[:-1])):
        command.downgrade(cfg, previous)
        remaining -= TABLES_BY_REVISION[revision]
        assert _current(database_url) == previous
        assert _tables(database_url) == remaining
        # The Phase 1 row predates every revision being reverted here.
        assert _execute(database_url, "SELECT name FROM assets") == [("Pump",)]

    command.downgrade(cfg, "base")
    assert _tables(database_url) == set()
    assert _enum_types(database_url) == set()

    command.upgrade(cfg, "head")  # and the whole chain still applies afterwards
    assert _current(database_url) == "0005"


@needs_database
def test_upgrade_refuses_a_database_that_already_has_unversioned_tables(database_url):
    """Tables but no alembic_version means Alembic cannot know what is there.
    That has to fail loudly and change nothing, not be skipped over."""
    _execute(database_url, "CREATE TABLE assets (id integer PRIMARY KEY, note text)")
    _execute(database_url, "INSERT INTO assets VALUES (1, 'keep me')")

    with pytest.raises(sa.exc.ProgrammingError):
        command.upgrade(_alembic_config(database_url), "head")

    assert _execute(database_url, "SELECT note FROM assets") == [("keep me",)]
    assert _tables(database_url) == {"assets"}  # the failed upgrade rolled back completely
    assert _enum_types(database_url) == set()


@needs_database
def test_concurrent_upgrades_do_not_race(database_url):
    """Three `alembic upgrade head` processes at once, as when several
    containers start together: env.py's advisory lock makes them take turns."""
    # Through the real CLI and alembic.ini, which read the URL from Settings.
    env = {**os.environ, "DATABASE_URL": database_url}
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=BACKEND_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(3)
    ]
    outputs = [p.communicate(timeout=120)[0] for p in processes]

    assert [p.returncode for p in processes] == [0, 0, 0], "\n".join(outputs)
    assert _current(database_url) == "0005"
    # Exactly one of them did the work.
    assert sum("Running upgrade  -> 0001" in out for out in outputs) == 1
