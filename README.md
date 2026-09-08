# TBGenerator

A small web UI around the [Tension Board climb generator](https://github.com/damonseto/tensionboardproject):
pick a board angle and a grade, click Generate, see a Tension Board 2 climb drawn on the board.

The model (`ClimbGPT`, a ~1M parameter decoder-only transformer) and its constrained
decoder are copied verbatim from that repo's notebook into `model.py`. `board.py` is the
same file as in the model repo. `best_model.pt` is the trained checkpoint.

## Run it

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn main:app --reload
```

Open http://localhost:8000, pick 40° and grade 20 (V5), click Generate.

## API

`POST /generate` with JSON `{"angle": 40, "grade": 20, "temperature": 0.8}`
(temperature optional). Returns:

```json
{
  "holds": [{"placement_id": 466, "role": 8, "x": -24, "y": 4}, ...],
  "frames": "p466r8p477r6...",
  "warning": "..."   // only present if no climbable candidate was found
}
```

Roles: 5 = start, 6 = hand, 7 = finish, 8 = foot. Grade is on the Aurora scale
(16 = V3, 20 = V5, 23 = V7). Angle is 0–70 in steps of 5.

The server generates up to 10 candidates and returns the first one where no start or
hand hold sits above the finish hold. If all 10 fail it returns the last one with a
`warning` field.

`GET /holds` returns the 690 placements with x/y coordinates and the grade-name table.

## Files

- `main.py` — FastAPI server. Loads the model and hold table once at startup.
- `model.py` — model classes and `generate()`, moved out of the notebook.
- `board.py` — tokenizer and constants, unchanged from the model repo.
- `static/index.html` — the page. Vanilla JS, one canvas, no build step.
- `holds.json` — hold coordinates and grade names, exported from `tension.db` by
  `export_holds.py` so the app does not need the 92MB database.

## Caveats

The training data is concentrated at 40–45° and roughly V2–V7. Other angles and
grades are offered but the model has seen little data there, so expect worse
climbs. The model knows nothing about hold type or orientation, so a climb can be
unclimbable for reasons the validity check cannot catch.
