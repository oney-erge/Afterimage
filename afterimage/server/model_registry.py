"""Durable model and job records for the local Afterimage application."""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sqlite3
import threading
import time
from typing import Any

from afterimage.cli import DEFAULT_STORE_ROOT


def default_database_path() -> pathlib.Path:
    configured = os.environ.get("AFTERIMAGE_STATE_DB")
    if configured:
        return pathlib.Path(configured).expanduser()
    return DEFAULT_STORE_ROOT.parent / "state" / "afterimage.sqlite3"


class ModelRegistry:
    """Small SQLite registry with additive schema initialization."""

    def __init__(self, path: pathlib.Path | None = None):
        self.path = pathlib.Path(path or default_database_path())
        self._initialized = False
        self._init_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        """A fresh connection for one call. Callers must wrap it in
        ``contextlib.closing`` in addition to using it as its own context
        manager: ``sqlite3.Connection``'s context-manager protocol only
        commits or rolls back a transaction, it does not close the
        connection, so a bare ``with self._connect() as connection:``
        silently leaks one OS file handle per call. Every registry method
        in this class used to do exactly that, in a server meant to run for
        days; use ``with contextlib.closing(self._connect()) as connection,
        connection:`` instead, which closes on top of the same commit/
        rollback behavior."""
        self._ensure_schema()
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30)
            try:
                connection.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS models (
                      model_id TEXT PRIMARY KEY,
                      revision TEXT,
                      state TEXT NOT NULL,
                      stage TEXT,
                      source_kind TEXT NOT NULL DEFAULT 'huggingface',
                      source_ref TEXT,
                      local_snapshot TEXT,
                      store_path TEXT,
                      bytes_done INTEGER NOT NULL DEFAULT 0,
                      bytes_total INTEGER,
                      compatibility TEXT,
                      metadata_json TEXT NOT NULL DEFAULT '{}',
                      error TEXT,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS jobs (
                      id TEXT PRIMARY KEY,
                      model_id TEXT,
                      kind TEXT NOT NULL,
                      lane TEXT NOT NULL,
                      status TEXT NOT NULL,
                      progress_json TEXT NOT NULL DEFAULT '{}',
                      result_json TEXT,
                      error TEXT,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS jobs_model_id ON jobs(model_id);
                    CREATE INDEX IF NOT EXISTS jobs_updated_at ON jobs(updated_at DESC);
                    CREATE TABLE IF NOT EXISTS runtime_profiles (
                      id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      model_id TEXT NOT NULL,
                      config_json TEXT NOT NULL,
                      source_run_id TEXT,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS runtime_profiles_model_id
                      ON runtime_profiles(model_id);
                    """
                )
                connection.execute(
                    "UPDATE jobs SET status='interrupted', updated_at=? "
                    "WHERE status IN ('queued','running','pause_requested','pausing',"
                    "'cancelling')",
                    (time.time(),),
                )
                connection.execute(
                    "UPDATE models SET state='interrupted', stage='interrupted', "
                    "updated_at=? WHERE state IN "
                    "('queued','downloading','preparing','verifying')",
                    (time.time(),),
                )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    @staticmethod
    def _model_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json") or "{}")
        return value

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["progress"] = json.loads(value.pop("progress_json") or "{}")
        result = value.pop("result_json")
        value["result"] = json.loads(result) if result else None
        return value

    def upsert_model(self, model_id: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        existing = self.get_model(model_id)
        values = {
            "revision": None,
            "state": "remote",
            "stage": None,
            "source_kind": "huggingface",
            "source_ref": model_id,
            "local_snapshot": None,
            "store_path": None,
            "bytes_done": 0,
            "bytes_total": None,
            "compatibility": None,
            "metadata": {},
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        if existing:
            values.update(existing)
        values.update(fields)
        values["updated_at"] = now
        metadata_json = json.dumps(values.pop("metadata", {}), ensure_ascii=False)
        columns = [
            "model_id", "revision", "state", "stage", "source_kind", "source_ref",
            "local_snapshot", "store_path", "bytes_done", "bytes_total",
            "compatibility", "metadata_json", "error", "created_at", "updated_at",
        ]
        parameters = [
            model_id,
            values["revision"], values["state"], values["stage"],
            values["source_kind"], values["source_ref"], values["local_snapshot"],
            values["store_path"], values["bytes_done"], values["bytes_total"],
            values["compatibility"], metadata_json, values["error"],
            values["created_at"], values["updated_at"],
        ]
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{name}=excluded.{name}" for name in columns[1:])
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                f"INSERT INTO models ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(model_id) DO UPDATE SET {updates}",
                parameters,
            )
        return self.get_model(model_id) or {}

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM models WHERE model_id=?", (model_id,)
            ).fetchone()
        return self._model_row(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM models ORDER BY updated_at DESC"
            ).fetchall()
        return [self._model_row(row) for row in rows]

    def delete_model(self, model_id: str) -> bool:
        """Remove one registry row after its local files were handled."""

        with contextlib.closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM models WHERE model_id=?", (model_id,)
            )
        return cursor.rowcount > 0

    def create_job(self, job_id: str, kind: str, lane: str, model_id: str | None) -> None:
        now = time.time()
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO jobs "
                "(id, model_id, kind, lane, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, model_id, kind, lane, now, now),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "progress", "result", "error"}
        assignments: list[str] = []
        parameters: list[Any] = []
        for name, value in fields.items():
            if name not in allowed:
                continue
            column = f"{name}_json" if name in {"progress", "result"} else name
            assignments.append(f"{column}=?")
            parameters.append(
                json.dumps(value, ensure_ascii=False, default=str)
                if name in {"progress", "result"} and value is not None
                else value
            )
        assignments.append("updated_at=?")
        parameters.extend([time.time(), job_id])
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                f"UPDATE jobs SET {','.join(assignments)} WHERE id=?", parameters
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """Remove one job row. Callers must have already confirmed the job
        is not active (queued/running/paused/pause_requested/cancelling) --
        this method itself does not check, matching delete_model's own
        contract of "the caller handled the live state, this just clears
        the durable record"."""
        with contextlib.closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE id=?", (job_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _profile_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json") or "{}")
        return value

    def save_runtime_profile(
        self,
        profile_id: str,
        *,
        name: str,
        model_id: str,
        config: dict[str, Any],
        source_run_id: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO runtime_profiles "
                "(id,name,model_id,config_json,source_run_id,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name,model_id=excluded.model_id,"
                "config_json=excluded.config_json,source_run_id=excluded.source_run_id,"
                "updated_at=excluded.updated_at",
                (profile_id, name, model_id, json.dumps(config, ensure_ascii=False),
                 source_run_id, now, now),
            )
        return self.get_runtime_profile(profile_id) or {}

    def get_runtime_profile(self, profile_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM runtime_profiles WHERE id=?", (profile_id,)
            ).fetchone()
        return self._profile_row(row) if row else None

    def list_runtime_profiles(self, model_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM runtime_profiles"
        parameters: tuple[Any, ...] = ()
        if model_id:
            query += " WHERE model_id=?"
            parameters = (model_id,)
        query += " ORDER BY updated_at DESC"
        with contextlib.closing(self._connect()) as connection, connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._profile_row(row) for row in rows]

    def delete_runtime_profile(self, profile_id: str) -> bool:
        with contextlib.closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM runtime_profiles WHERE id=?", (profile_id,)
            )
        return cursor.rowcount > 0


model_registry = ModelRegistry()
