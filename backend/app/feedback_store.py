"""
feedback_store.py
-----------------
Handles all MySQL operations for the RLHF-style feedback system.
Creates the `feedback` table on startup, provides:
  - save_feedback()       : persist a user thumbs-up/down record
  - get_good_examples()   : fetch top-N "good" Q&A pairs for few-shot prompting
  - get_analytics()       : return aggregate good/bad counts
"""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_MODE = os.environ.get("DB_MODE", "local")


def _build_connect_kwargs() -> dict:
    base = {
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
        "charset":  "utf8mb4",
        "connection_timeout": 10,
    }

    if DB_MODE == "socket":
        base["unix_socket"] = os.environ.get(
            "MYSQL_SOCKET_PATH", "/var/run/mysqld/mysqld.sock"
        )
    else:
        base["host"] = os.environ.get(
            "DB_HOST",
            "host.docker.internal" if DB_MODE == "local" else None,
        )
        base["port"] = int(os.environ.get("DB_PORT", 3306))

    return base


class FeedbackStore:
    def __init__(self):
        self.conn = None
        try:
            kwargs = _build_connect_kwargs()
            db_name = kwargs.pop("database")

            self.conn = mysql.connector.connect(**kwargs)
            cursor = self.conn.cursor()

            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{db_name}`")

            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    session_id  VARCHAR(255),
                    question    TEXT        NOT NULL,
                    answer      TEXT        NOT NULL,
                    feedback    VARCHAR(10) NOT NULL,
                    created_at  TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

            self.conn.database = db_name
            print(f"[feedback_store] Connected (mode={DB_MODE}). Feedback logging active.")

        except KeyError as e:
            print(f"[feedback_store] Missing required env var: {e}. Feedback disabled.")
            self.conn = None
        except Exception as e:
            print(f"[feedback_store] Failed to connect (mode={DB_MODE}): {e}. Feedback disabled.")
            self.conn = None

    # ------------------------------------------------------------------
    def save_feedback(self, session_id: str, question: str, answer: str, feedback: str):
        """Persist one feedback record to MySQL."""
        if not self.conn:
            return
        try:
            # Reconnect if connection dropped
            if not self.conn.is_connected():
                self.conn.reconnect(attempts=3, delay=1)

            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (session_id, question, answer, feedback) "
                "VALUES (%s, %s, %s, %s)",
                (session_id, question, answer, feedback),
            )
            self.conn.commit()
            print(f"[feedback_store] Saved '{feedback}' feedback for session={session_id}")
        except Exception as e:
            print(f"[feedback_store] Error saving feedback: {e}")

    # ------------------------------------------------------------------
    def get_good_examples(self, limit: int = 5) -> list:
        """
        Return the most recent `limit` Q&A pairs that received a 'good' rating.
        Used for few-shot injection into the RAG prompt.
        Returns list of dicts: [{"question": ..., "answer": ...}, ...]
        """
        if not self.conn:
            return []
        try:
            if not self.conn.is_connected():
                self.conn.reconnect(attempts=3, delay=1)

            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT question, answer FROM feedback "
                "WHERE feedback = 'good' "
                "ORDER BY created_at DESC "
                "LIMIT %s",
                (limit,),
            )
            rows = cursor.fetchall()
            return [{"question": r["question"], "answer": r["answer"]} for r in rows]
        except Exception as e:
            print(f"[feedback_store] Error fetching good examples: {e}")
            return []

    # ------------------------------------------------------------------
    def get_analytics(self) -> dict:
        """Return counts of good vs bad, plus recent bad questions."""
        if not self.conn:
            return {"good": 0, "bad": 0, "recent_bad": []}
        try:
            if not self.conn.is_connected():
                self.conn.reconnect(attempts=3, delay=1)

            cursor = self.conn.cursor(dictionary=True)

            # Count per feedback type
            cursor.execute(
                "SELECT feedback, COUNT(*) AS cnt FROM feedback GROUP BY feedback"
            )
            rows = cursor.fetchall()
            counts = {"good": 0, "bad": 0}
            for row in rows:
                counts[row["feedback"]] = row["cnt"]

            # Recent bad questions (last 10)
            cursor.execute(
                "SELECT question, created_at FROM feedback "
                "WHERE feedback = 'bad' "
                "ORDER BY created_at DESC LIMIT 10"
            )
            bad_rows = cursor.fetchall()
            recent_bad = [
                {"question": r["question"], "created_at": str(r["created_at"])}
                for r in bad_rows
            ]

            return {
                "good": counts.get("good", 0),
                "bad": counts.get("bad", 0),
                "recent_bad": recent_bad,
            }
        except Exception as e:
            print(f"[feedback_store] Error fetching analytics: {e}")
            return {"good": 0, "bad": 0, "recent_bad": []}

    # ------------------------------------------------------------------
    def close(self):
        if self.conn:
            self.conn.close()


# Singleton — imported by routes/feedback.py and ai/rag_pipeline.py
feedback_store = FeedbackStore()
