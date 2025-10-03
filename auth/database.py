import sqlite3

DATABASE_NAME = 'users.db'

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database, creates tables, and seeds initial data if necessary."""
    # Moved import here to prevent circular dependency
    from .auth_utils import hash_password

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL, 
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if the table is empty
    cursor.execute("SELECT COUNT(id) FROM users")
    user_count = cursor.fetchone()[0]
    
    # If empty, create the initial teacher account
    if user_count == 0:
        print("First run: Creating initial accounts...")
        from .auth_utils import hash_password

        # Original Teacher
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, status) VALUES (?, ?, ?, ?)",
            ("teacher", hash_password("admin"), 'teacher', 'active')
        )
        print("  - Created: username='teacher', password='admin'")

        # Test Teacher
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, status) VALUES (?, ?, ?, ?)",
            ("teacher2", hash_password("teacherpass"), 'teacher', 'active')
        )
        print("  - Created: username='teacher2', password='teacherpass'")

        # Test Student (active for easier testing)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, status) VALUES (?, ?, ?, ?)",
            ("student1", hash_password("studentpass"), 'student', 'active')
        )
        print("  - Created: username='student1', password='studentpass'")

        print("Please change default passwords after your first login.")

    conn.commit()
    conn.close()
