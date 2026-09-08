"""One-time export: pull hold coordinates and grade names out of tension.db
into holds.json so the app runs without the 92MB database.

    python export_holds.py ../TensionBoardProject/tension.db
"""
import json
import sqlite3
import sys

from board import load_holds

conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)

holds = load_holds(conn)[['placement_id', 'x', 'y']].sort_values('placement_id')
grades = dict(conn.execute(
    "SELECT difficulty, boulder_name FROM difficulty_grades WHERE is_listed = 1"
).fetchall())

with open('holds.json', 'w') as f:
    json.dump({'holds': holds.to_dict(orient='records'), 'grades': grades}, f)

print(len(holds), 'holds,', len(grades), 'grades ->', 'holds.json')
