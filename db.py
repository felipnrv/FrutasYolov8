import sqlite3

conn = sqlite3.connect('example.db')

cursor = conn.cursor()

cursor.execute('SELECT SQLITE_VERSION()')

data = cursor.fetchone()
print(f"SQLite version: {data[0]}")