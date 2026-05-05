import sqlite3
import hashlib
from datetime import datetime

def connect_db():
    return sqlite3.connect("database.db")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email    TEXT DEFAULT '',
            age      INTEGER DEFAULT 0,
            gender   TEXT DEFAULT ''
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            filename    TEXT,
            mse         REAL,
            prediction  TEXT,
            risk_level  TEXT DEFAULT '-',
            zcr         REAL DEFAULT 0,
            energy      REAL DEFAULT 0,
            mfcc_mean   REAL DEFAULT 0,
            snr_db      REAL DEFAULT 0,
            timestamp   TEXT
        )
    """)
    for col, coltype in [
        ("email","TEXT DEFAULT ''"),("age","INTEGER DEFAULT 0"),("gender","TEXT DEFAULT ''"),
    ]:
        try: cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")
        except: pass
    for col, coltype in [
        ("risk_level","TEXT DEFAULT '-'"),("zcr","REAL DEFAULT 0"),
        ("energy","REAL DEFAULT 0"),("mfcc_mean","REAL DEFAULT 0"),("snr_db","REAL DEFAULT 0"),
        ("ds_label","TEXT DEFAULT ''"),("ds_pd_prob","REAL DEFAULT 0"),
    ]:
        try: cursor.execute(f"ALTER TABLE results ADD COLUMN {col} {coltype}")
        except: pass
    conn.commit()
    conn.close()

def register_user(username, password, email="", age=0, gender=""):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email, age, gender) VALUES (?, ?, ?, ?, ?)",
            (username, hash_password(password), email, age, gender)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=? AND password=?",
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

def insert_result(user_id, filename, mse, prediction,
                  risk_level="-", zcr=0, energy=0, mfcc_mean=0, snr_db=0,
                  ds_label="", ds_pd_prob=0.0):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results
            (user_id,filename,mse,prediction,risk_level,zcr,energy,mfcc_mean,snr_db,timestamp,ds_label,ds_pd_prob)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (user_id, filename, mse, prediction, risk_level,
          zcr, energy, mfcc_mean, snr_db, str(datetime.now()),
          ds_label, ds_pd_prob))
    conn.commit()
    conn.close()

def get_user_results(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, mse, prediction, timestamp, risk_level, zcr, energy, mfcc_mean, snr_db,
               COALESCE(ds_label,''), COALESCE(ds_pd_prob,0)
        FROM results WHERE user_id=? ORDER BY timestamp DESC
    """, (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data

def get_user_results_with_ids(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, mse, prediction, timestamp, risk_level, zcr, energy, mfcc_mean, snr_db,
               COALESCE(ds_label,''), COALESCE(ds_pd_prob,0)
        FROM results WHERE user_id=? ORDER BY timestamp DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "filename": row[1],
            "mse": row[2],
            "prediction": row[3],
            "timestamp": row[4],
            "risk_level": row[5],
            "zcr": row[6],
            "energy": row[7],
            "mfcc_mean": row[8],
            "snr_db": row[9],
            "ds_label": row[10],
            "ds_pd_prob": row[11],
        }
        for row in rows
    ]

def get_username(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else "Unknown"

def get_user_profile(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, age, gender FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {"username": row[0], "email": row[1], "age": row[2], "gender": row[3]} if row else {}

def get_stats(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*),
            SUM(CASE WHEN prediction NOT LIKE '%Abnormal%' THEN 1 ELSE 0 END),
            SUM(CASE WHEN prediction LIKE '%Abnormal%' THEN 1 ELSE 0 END),
            AVG(mse)
        FROM results WHERE user_id=?
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {"total": row[0] or 0, "normal": row[1] or 0,
            "abnormal": row[2] or 0, "avg_mse": round(row[3], 6) if row[3] else 0}

def delete_result(result_id, user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM results WHERE id=? AND user_id=?", (result_id, user_id))
    conn.commit()
    conn.close()

def get_result_ids(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename FROM results WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data
