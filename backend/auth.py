import sqlite3, hashlib, os, hmac
from contextlib import contextmanager
from pathlib import Path
DB_PATH = Path(__file__).resolve().parents[1] / "database" / "users.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
@contextmanager
def _connect():
    connection=sqlite3.connect(DB_PATH)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
def init_db():
    with _connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt BLOB NOT NULL, password_hash BLOB NOT NULL, district TEXT DEFAULT '', state TEXT DEFAULT 'Karnataka')")
        con.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, username TEXT, module TEXT, result TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
def _hash(password, salt): return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
def register(username, password):
    username=username.strip().lower()
    if len(username)<3 or len(password)<8: return False,"Username needs 3+ characters and password needs 8+ characters."
    salt=os.urandom(16)
    try:
        with _connect() as con: con.execute("INSERT INTO users(username,salt,password_hash) VALUES(?,?,?)",(username,salt,_hash(password,salt)))
        return True,"Account created. Please log in."
    except sqlite3.IntegrityError: return False,"Username already exists."
def login(username,password):
    with _connect() as con: row=con.execute("SELECT salt,password_hash FROM users WHERE username=?",(username.strip().lower(),)).fetchone()
    return bool(row and hmac.compare_digest(_hash(password,row[0]),row[1]))
def get_profile(username):
    with _connect() as con: row=con.execute("SELECT district,state FROM users WHERE username=?",(username,)).fetchone()
    return row or ("","Karnataka")
def update_profile(username,district,state):
    with _connect() as con: con.execute("UPDATE users SET district=?,state=? WHERE username=?",(district,state,username))
def add_history(username,module,result):
    with _connect() as con: con.execute("INSERT INTO history(username,module,result) VALUES(?,?,?)",(username,module,result[:2000]))
def get_history(username,limit=30):
    with _connect() as con: return con.execute("SELECT id,module,result,created_at FROM history WHERE username=? ORDER BY id DESC LIMIT ?",(username,limit)).fetchall()
def delete_history(username,history_id):
    with _connect() as con:
        cursor=con.execute("DELETE FROM history WHERE id=? AND username=?",(history_id,username))
        return cursor.rowcount==1
