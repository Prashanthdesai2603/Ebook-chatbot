import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_MODE = os.environ.get("DB_MODE", "local")  # local | socket | remote


def _build_connect_kwargs() -> dict:
    """
    Return the correct keyword-args for mysql.connector.connect()
    based on DB_MODE.

    MODE 1 – local:  TCP to host.docker.internal (or any explicit DB_HOST)
    MODE 2 – socket: Unix socket mounted into the container
    MODE 3 – remote: TCP to a remote host / RDS endpoint
    """
    base = {
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
        "charset":  "utf8mb4",
        "connection_timeout": 10,
    }

    if DB_MODE == "socket":
        # Unix socket — MySQL sees connection as 'localhost'; no % grant needed.
        base["unix_socket"] = os.environ.get(
            "MYSQL_SOCKET_PATH", "/var/run/mysqld/mysqld.sock"
        )
    else:
        # local  → DB_HOST defaults to host.docker.internal
        # remote → DB_HOST must be set explicitly in .env
        base["host"] = os.environ.get(
            "DB_HOST",
            "host.docker.internal" if DB_MODE == "local" else None,
        )
        base["port"] = int(os.environ.get("DB_PORT", 3306))

    return base


class MySQLChatLogger:
    def __init__(self):
        self.conn = None
        try:
            kwargs = _build_connect_kwargs()
            db_name = kwargs.pop("database")          # connect without DB first

            self.conn = mysql.connector.connect(**kwargs)
            cursor = self.conn.cursor()

            # Ensure database + table exist
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{db_name}`")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    session_id  VARCHAR(255),
                    query       TEXT        NOT NULL,
                    response    TEXT        NOT NULL,
                    timestamp   DATETIME    DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

            # Re-select the DB so subsequent queries don't need USE
            self.conn.database = db_name
            print(f"[mysql_logger] Connected (mode={DB_MODE}). Chat history logging active.")

        except KeyError as e:
            print(f"[mysql_logger] Missing required env var: {e}. Logging disabled.")
            self.conn = None
        except Exception as e:
            print(f"[mysql_logger] Failed to connect (mode={DB_MODE}): {e}. Logging disabled.")
            self.conn = None

    def log_chat(self, query: str, response: str):
        self.save_chat("legacy", query, response)

    def save_chat(self, session_id: str, query: str, response: str):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            # Check if column exists (for backward compatibility if table exists)
            cursor.execute("SHOW COLUMNS FROM chat_history LIKE 'session_id'")
            if not cursor.fetchone():
                 cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id VARCHAR(255) AFTER id")
            
            cursor.execute(
                "INSERT INTO chat_history (session_id, query, response) VALUES (%s, %s, %s)",
                (session_id, query, response),
            )
            self.conn.commit()
        except Exception as e:
            print(f"[mysql_logger] Error saving chat: {e}")

    def close(self):
        if self.conn:
            self.conn.close()


# Singleton — imported by main.py
chat_logger = MySQLChatLogger()
