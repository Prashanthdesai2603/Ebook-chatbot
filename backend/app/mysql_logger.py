import os
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MySQL configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ebook_chatbot")

class MySQLChatLogger:
    def __init__(self):
        self.conn = None
        try:
            self.conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD
            )
            cursor = self.conn.cursor()
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
            cursor.execute(f"USE {MYSQL_DATABASE}")
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            print("Connected to MySQL for chat history logging.")
        except Exception as e:
            print(f"Failed to connect to MySQL: {e}")
            self.conn = None

    def log_chat(self, query: str, response: str):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            query_sql = "INSERT INTO chat_history (query, response) VALUES (%s, %s)"
            cursor.execute(query_sql, (query, response))
            self.conn.commit()
        except Exception as e:
            print(f"Error logging chat to MySQL: {e}")

    def close(self):
        if self.conn:
            self.conn.close()

# Singleton instance
chat_logger = MySQLChatLogger()
