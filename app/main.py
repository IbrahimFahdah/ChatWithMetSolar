import json
import os
import pathlib
import pickle
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent.parent
MODEL_PATH = HERE / "model" / "model.pkl"

app = FastAPI(title="MetSolar Carport Estimator")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

_bundle: dict | None = None


def get_bundle():
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def predict_from_params(params: dict) -> dict:
    bundle = get_bundle()
    row = [params[col] for col in bundle["feature_cols"]]
    results = {}
    for model, target in zip(bundle["models"], bundle["target_cols"]):
        results[target] = float(model.predict(np.array([row]))[0])
    return results


EXTRACTION_SYSTEM = """You are a solar carport parameter extractor.
The user will describe a solar carport installation. Extract these fields and return ONLY valid JSON with no extra text:
{
  "carport_type": <int 1=SingleMonoIncline, 2=SingleMonoDecline, 3=DuoMono, 4=GullWing>,
  "panel_length_mm": <float, typically 1500-4000>,
  "panel_width_mm": <float, typically 750-2500>,
  "panel_power_w": <float, typically 200-1200>,
  "area_length_mm": <float, the total carport length in mm, typically 9000-300000>,
  "delivery_area": <string, the UK location name provided by the user, or "unknown" if not specified>,
  "delivery_artic_cost_gbp": <float, based on UK delivery area — use these reference costs: London=450, Birmingham=550, Manchester=600, Leeds=620, Glasgow=900, Edinburgh=950, Cardiff=650, Bristol=500, Exeter=594, Falkirk=1290, Aberdeen=1400, default for unknown UK locations=450 (same as London)>
}
Rules:
- number_row is derived automatically (3 for type 1 or 2, 6 for type 3 or 4) — do NOT include it
- If the user gives area in metres, convert to mm (multiply by 1000)
- If carport type is ambiguous, default to 1 (SingleMonoIncline)
- If panel size is not specified, use common defaults: length=2333, width=1134, power=600
- Always return valid JSON with all 7 fields"""

RESPONSE_SYSTEM = """You are a friendly solar carport estimator assistant for MetSolar.
Given the estimation results, write a short (3-5 sentence) conversational response that:
1. States the key numbers clearly: power in kW, weight in kg, cost in £
2. If any values were not provided by the user and defaults were assumed (panel length, width, power, or carport type), explicitly call them out — e.g. "I've assumed standard 600W panels (2333×1134mm) as none were specified."
3. Adds one brief helpful note (e.g. about installation, grid connection, or savings)
Keep it professional but warm. Do not use bullet points."""


class NLRequest(BaseModel):
    message: str
    history: Optional[list] = None


class PredictRequest(BaseModel):
    carport_type: int
    panel_length_mm: float
    panel_width_mm: float
    panel_power_w: float
    area_length_mm: float
    number_row: int
    delivery_artic_cost_gbp: float


@app.get("/")
def root():
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    params = req.model_dump()
    results = predict_from_params(params)
    return {"params_used": params, **results}


@app.post("/estimate")
def estimate(req: NLRequest):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    # Step 1: extract structured params from natural language
    extraction_resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": req.message},
        ],
        temperature=0,
        max_tokens=300,
    )
    raw = extraction_resp.choices[0].message.content.strip()

    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"Could not parse parameters from message. Raw: {raw}")

    # derive number_row from carport_type
    params["number_row"] = 6 if params["carport_type"] in (3, 4) else 3
    delivery_area = params.pop("delivery_area", "unknown")

    # detect which fields Groq defaulted (server-side, not inferred by LLM)
    DEFAULTS = {
        "panel_length_mm": 2333,
        "panel_width_mm": 1134,
        "panel_power_w": 600,
        "carport_type": 1,
    }
    DEFAULT_LABELS = {
        "panel_length_mm": "panel length 2333mm",
        "panel_width_mm": "panel width 1134mm",
        "panel_power_w": "panel power 600W",
        "carport_type": "carport type SingleMonoIncline",
    }
    assumed = [
        DEFAULT_LABELS[k] for k, v in DEFAULTS.items() if params.get(k) == v
    ]
    if delivery_area.lower() == "unknown":
        assumed.append("delivery location (not specified, London delivery cost used as default)")

    # Step 2: run XGBoost prediction
    try:
        predictions = predict_from_params(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Step 3: generate friendly response
    assumed_note = (
        f"The following values were NOT provided by the user and defaults were used: {', '.join(assumed)}. "
        f"You MUST mention these assumptions in your response."
        if assumed else "All values were explicitly provided by the user."
    )
    context = (
        f"User asked: {req.message}\n"
        f"{assumed_note}\n"
        f"Final params: {json.dumps(params)}\n"
        f"Predictions: power={predictions['total_power_kw']:.1f} kW, "
        f"weight={predictions['total_weight_kg']:.0f} kg, "
        f"cost=£{predictions['total_cost_gbp']:.0f}"
    )
    response_resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": RESPONSE_SYSTEM},
            {"role": "user", "content": context},
        ],
        temperature=0.4,
        max_tokens=250,
    )
    friendly = response_resp.choices[0].message.content.strip()

    carport_names = {1: "SingleMonoIncline", 2: "SingleMonoDecline", 3: "DuoMono", 4: "GullWing"}

    return {
        "response": friendly,
        "params_used": {**params, "carport_type_name": carport_names.get(params["carport_type"], "Unknown"), "delivery_area": delivery_area},
        "total_power_kw": round(predictions["total_power_kw"], 2),
        "total_weight_kg": round(predictions["total_weight_kg"], 1),
        "total_cost_gbp": round(predictions["total_cost_gbp"], 2),
    }
