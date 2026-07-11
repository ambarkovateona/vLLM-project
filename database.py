import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "chat_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            title      TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT NOT NULL,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str) -> bool:
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(username: str, password: str) -> bool:
    conn = get_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    return user is not None


def create_conversation(username: str) -> int:
    title = "New Chat"
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO conversations (username, title) VALUES (?, ?)",
        (username, title)
    )
    conv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conv_id


def update_conversation_title(conversation_id: int, title: str):
    conn = get_connection()
    conn.execute(
        "UPDATE conversations SET title=? WHERE id=?",
        (title, conversation_id)
    )
    conn.commit()
    conn.close()


def get_user_conversations(username: str) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title FROM conversations WHERE username=? ORDER BY created_at DESC",
        (username,)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]


def get_conversation_messages(conversation_id: int) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def save_message(username: str, conversation_id: int, role: str, content: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (username, conversation_id, role, content) VALUES (?, ?, ?, ?)",
        (username, conversation_id, role, content)
    )
    conn.commit()
    conn.close()


def clear_conversation(conversation_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    conn.commit()
    conn.close()

def init_usage_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            username          TEXT NOT NULL,
            conversation_id   INTEGER,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_token_usage(username, conversation_id, prompt_tokens, completion_tokens):
    conn = get_connection()
    conn.execute(
        "INSERT INTO token_usage (username, conversation_id, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?)",
        (username, conversation_id, prompt_tokens, completion_tokens)
    )
    conn.commit()
    conn.close()


def get_user_token_usage(username):
    conn = get_connection()
    row = conn.execute(
        """SELECT COALESCE(SUM(prompt_tokens), 0),
                  COALESCE(SUM(completion_tokens), 0),
                  COUNT(*)
           FROM token_usage WHERE username=?""",
        (username,)
    ).fetchone()
    conn.close()
    return {"prompt": row[0], "completion": row[1], "total": row[0] + row[1], "requests": row[2]}


def get_all_users_usage():
    conn = get_connection()
    rows = conn.execute(
        """SELECT username,
                  COALESCE(SUM(prompt_tokens), 0),
                  COALESCE(SUM(completion_tokens), 0),
                  COUNT(*)
           FROM token_usage GROUP BY username
           ORDER BY SUM(prompt_tokens) + SUM(completion_tokens) DESC"""
    ).fetchall()
    conn.close()
    return [{"username": r[0], "prompt": r[1], "completion": r[2], "total": r[1] + r[2], "requests": r[3]} for r in rows]