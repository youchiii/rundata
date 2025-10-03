import bcrypt
import sqlite3
from .database import get_db_connection

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed one."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_user(username: str, password: str) -> bool:
    """Creates a new student user with a pending status."""
    hashed_password = hash_password(password)
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, status) VALUES (?, ?, ?, ?)",
            (username, hashed_password, 'student', 'pending')
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # This happens if the username is already taken
        return False
    finally:
        conn.close()

def get_user(username: str):
    """Fetches a single user by their username."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def get_pending_users():
    """Fetches all users with a 'pending' status."""
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, created_at FROM users WHERE status = 'pending' ORDER BY created_at DESC").fetchall()
    conn.close()
    return users

def update_user_status(user_id: int, status: str):
    """Updates a user's status to 'active' or 'rejected'."""
    if status not in ['active', 'rejected']:
        raise ValueError("Status must be either 'active' or 'rejected'.")
    
    conn = get_db_connection()
    conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    conn.commit()
    conn.close()

def get_all_users():
    """Fetches all users from the database (excluding password hash)."""
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role, status, created_at FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return users

def reset_password(user_id: int, new_password: str):
    """Resets a user's password."""
    new_hashed_password = hash_password(new_password)
    conn = get_db_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hashed_password, user_id))
    conn.commit()
    conn.close()
