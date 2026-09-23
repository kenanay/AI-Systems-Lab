"""Additive, versioned migrations; legacy rows remain administrator-only."""
from sqlalchemy import inspect, text


def migrate(engine):
    from backend.models import Base
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"))
        version = conn.execute(text("SELECT MAX(version) FROM schema_migrations")).scalar() or 0
        if version < 1:
            for table in Base.metadata.sorted_tables:
                if "owner_id" not in table.c:
                    continue
                columns = {c["name"] for c in inspect(conn).get_columns(table.name)}
                if "owner_id" not in columns:
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN owner_id VARCHAR(50)'))
                    conn.execute(text(f'CREATE INDEX "ix_{table.name}_owner_id" ON "{table.name}" (owner_id)'))
            conn.execute(text("INSERT INTO schema_migrations (version) VALUES (1)"))

        if version < 2:
            # Original schema implemented global hash uniqueness as an index.
            indexes = inspect(conn).get_indexes("files")
            for index in indexes:
                if index.get("unique") and index["column_names"] == ["sha256"]:
                    name = index["name"]
                    conn.execute(text(f'DROP INDEX "{name}"'))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_sha256 ON files (sha256)"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_files_owner_hash ON files (owner_id, sha256)"))
            conn.execute(text("INSERT INTO schema_migrations (version) VALUES (2)"))
