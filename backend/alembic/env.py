from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from app.core.config import get_settings
from app.core.database import Base
import app.models  # noqa: F401 - populates Base.metadata

config = context.config
# A caller that built the Config itself (tests/backend/test_migrations.py
# points it at a throwaway database) wins; alembic.ini leaves the URL empty,
# so the CLI always falls through to Settings.
if not config.get_main_option("sqlalchemy.url"):
    # "%" is configparser interpolation syntax, and a URL-encoded password has one.
    config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Arbitrary constant shared by every process migrating this schema ("TVASTA" in ASCII).
_MIGRATION_LOCK_KEY = 0x545641535441


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        # Two `alembic upgrade` processes started together (two containers,
        # or a developer and a deploy) would both read the same revision and
        # both run the same DDL. A session-level advisory lock makes the
        # second wait, then find nothing left to do. It is released when the
        # connection closes, so a crashed run cannot leave it held.
        if connection.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
            # End the transaction the SELECT opened: Alembic only commits a
            # transaction it began itself, so migrating inside this one would
            # silently roll everything back.
            connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
