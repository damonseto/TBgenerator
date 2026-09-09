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

# Board image calibration: 12 high x 16 wide (product_size 10) with all four
# hold sets. The edges say which board x/y range the image spans;
# static/img/board.png is built to exactly these edges by make_board_image.py.
PRODUCT_SIZE = 10
left, right, bottom, top = conn.execute(
    "SELECT edge_left, edge_right, edge_bottom, edge_top FROM product_sizes WHERE id = ?",
    (PRODUCT_SIZE,)).fetchone()
layers = ['img/board.png']

with open('holds.json', 'w') as f:
    json.dump({
        'holds': holds.to_dict(orient='records'),
        'grades': grades,
        'board': {'left': left, 'right': right, 'bottom': bottom, 'top': top,
                  'layers': layers},
    }, f)

print(len(holds), 'holds,', len(grades), 'grades ->', 'holds.json')
