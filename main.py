"""FastAPI wrapper around ClimbGPT.

    uvicorn main:app --reload
"""
import json

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from board import build_tokenizer
from model import ClimbGPT, generate

START, HAND, FINISH = 5, 6, 7
MAX_ATTEMPTS = 10

# --- load once at startup ---------------------------------------------------

with open('holds.json') as f:
    HOLD_DATA = json.load(f)

holds = pd.DataFrame(HOLD_DATA['holds'])
COORDS = {h['placement_id']: (h['x'], h['y']) for h in HOLD_DATA['holds']}
hold_to_token, token_to_hold, VOCAB_SIZE = build_tokenizer(holds)

model = ClimbGPT(VOCAB_SIZE)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.eval()

# --- api --------------------------------------------------------------------

app = FastAPI()


class GenerateRequest(BaseModel):
    angle: int = Field(ge=0, le=70)
    grade: float = Field(ge=0, le=40)
    temperature: float = Field(default=0.8, gt=0)


def is_climbable(climb):
    """No start or hand hold may sit above the (highest) finish hold."""
    finish_y = [COORDS[p][1] for p, r in climb if r == FINISH]
    if not finish_y:
        return False
    top = max(finish_y)
    return all(COORDS[p][1] <= top for p, r in climb if r in (START, HAND))


@app.get('/holds')
def get_holds():
    return HOLD_DATA


@app.post('/generate')
def post_generate(req: GenerateRequest):
    for _ in range(MAX_ATTEMPTS):
        climb = generate(model, token_to_hold, req.angle, req.grade, req.temperature)
        if is_climbable(climb):
            break

    out = {
        'holds': [{'placement_id': p, 'role': r, 'x': COORDS[p][0], 'y': COORDS[p][1]}
                  for p, r in climb],
        'frames': ''.join(f'p{p}r{r}' for p, r in climb),
    }
    if not is_climbable(climb):
        out['warning'] = (f'No climbable candidate in {MAX_ATTEMPTS} attempts: '
                          'a start or hand hold is above the finish.')
    return out


app.mount('/', StaticFiles(directory='static', html=True))
