from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.schema import ParsedContract, RiskScorecard


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS contracts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'default',
                    source_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'default',
                    contract_id TEXT,
                    verdict TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS party_memory (
                    party_key TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Safe schema migration for pre-existing SQLite databases
            try:
                conn.execute("ALTER TABLE contracts ADD COLUMN session_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN session_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass

    def save_contract_analysis(
        self,
        parsed: ParsedContract,
        scorecard: RiskScorecard,
        session_id: str = "default",
    ) -> str:
        payload = {"contract": parsed.model_dump(mode="json"), "scorecard": scorecard.model_dump(mode="json")}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO contracts
                (id, session_id, source_name, created_at, summary, risk_score, risk_level, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parsed.id,
                    session_id,
                    parsed.source_name,
                    parsed.created_at.isoformat(),
                    scorecard.contract_summary,
                    scorecard.overall_risk_score,
                    scorecard.risk_level,
                    json.dumps(payload),
                ),
            )
        return parsed.id

    def get_contract(self, contract_id: str) -> dict[str, Any] | None:
        """Retrieve full saved contract payload for reload."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            try:
                data["payload"] = json.loads(data["payload"])
            except Exception:
                pass
            return data

    def delete_contract(self, contract_id: str) -> bool:
        """Safely delete contract analysis and its feedback."""
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback WHERE contract_id = ?", (contract_id,))
            cursor = conn.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))
            return cursor.rowcount > 0

    def save_feedback(self, contract_id: str, verdict: str, note: str, session_id: str = "default") -> str:
        feedback_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (id, session_id, contract_id, verdict, note) VALUES (?, ?, ?, ?, ?)",
                (feedback_id, session_id, contract_id, verdict, note),
            )
        return feedback_id

    def save_preferences(self, text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (id, text) VALUES (1, ?)",
                (text,),
            )

    def get_preferences(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT text FROM preferences WHERE id = 1").fetchone()
        return row["text"] if row else ""

    def list_contracts(
        self,
        limit: int = 25,
        query: str = "",
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            clauses = []
            params: list[Any] = []

            if session_id:
                # Isolate to current session or shared demo contracts
                clauses.append("(session_id = ? OR session_id = 'default' OR session_id = 'demo')")
                params.append(session_id)

            if query.strip():
                clauses.append("(source_name LIKE ? OR summary LIKE ?)")
                pattern = f"%{query.strip()}%"
                params.extend([pattern, pattern])

            where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"""
            SELECT id, source_name, created_at, risk_score, risk_level, summary
            FROM contracts
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(sql, tuple(params)).fetchall()

        formatted = []
        for index, row in enumerate(rows, start=1):
            d = dict(row)
            d["display_id"] = f"REV-{index:03d}"
            try:
                dt = datetime.fromisoformat(d["created_at"])
                d["formatted_date"] = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                d["formatted_date"] = d["created_at"]
            formatted.append(d)
        return formatted

    def list_feedback(self, limit: int = 20, session_id: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if session_id:
                sql = """
                SELECT id, contract_id, verdict, note, created_at
                FROM feedback
                WHERE session_id = ? OR session_id = 'default'
                ORDER BY created_at DESC
                LIMIT ?
                """
                rows = conn.execute(sql, (session_id, limit)).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, contract_id, verdict, note, created_at
                    FROM feedback
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def feedback_context(self, limit: int = 5) -> str:
        rows = self.list_feedback(limit=limit)
        if not rows:
            return ""
        return "\n".join(f"- {row['verdict']}: {row['note']}" for row in rows if row["note"])

    def save_party_summary(self, party_key: str, summary: str) -> None:
        if not party_key.strip():
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO party_memory (party_key, summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (party_key.strip().lower(), summary),
            )

    def get_party_summary(self, party_key: str) -> str:
        if not party_key.strip():
            return ""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM party_memory WHERE party_key = ?",
                (party_key.strip().lower(),),
            ).fetchone()
        return row["summary"] if row else ""

    def save_chat_message(self, session_id: str, role: str, content: str) -> str:
        message_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (message_id, session_id, role, content),
            )
        return message_id

    def list_chat_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return list(reversed([dict(row) for row in rows]))
