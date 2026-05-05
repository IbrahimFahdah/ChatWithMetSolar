import json
import os
import pathlib
import pickle

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
_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


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


DELIVERY_COSTS = {
    "london": 450,
    "birmingham": 550,
    "manchester": 600,
    "leeds": 620,
    "glasgow": 900,
    "edinburgh": 950,
    "cardiff": 650,
    "bristol": 500,
    "exeter": 594,
    "falkirk": 1290,
    "aberdeen": 1400,
}
DEFAULT_DELIVERY_AREA = "london"
DEFAULT_DELIVERY_COST = DELIVERY_COSTS[DEFAULT_DELIVERY_AREA]

FIELD_BOUNDS = {
    "panel_length_mm": (1500, 4000),
    "panel_width_mm": (750, 2500),
    "panel_power_w": (200, 1200),
    "area_length_mm": (9000, 300000),
}


def validate_model_inputs(params: dict):
    out_of_range = [
        f"{field}={params[field]} (valid range {lo}–{hi})"
        for field, (lo, hi) in FIELD_BOUNDS.items()
        if field in params and not (lo <= params[field] <= hi)
    ]
    if out_of_range:
        raise HTTPException(status_code=422, detail=f"Input(s) outside model training range: {', '.join(out_of_range)}")

EXTRACTION_SYSTEM = """You are a solar carport parameter extractor.
The user will describe a solar carport installation. Extract these fields and return ONLY valid JSON with no extra text:
{
  "carport_type": <int 1=SingleMonoIncline, 2=SingleMonoDecline, 3=DuoMono, 4=GullWing>,
  "panel_length_mm": <float, typically 1500-4000>,
  "panel_width_mm": <float, typically 750-2500>,
  "panel_power_w": <float, typically 200-1200>,
  "area_length_mm": <float, the total carport length in mm, typically 9000-300000>,
  "delivery_area": <string, the UK location name provided by the user, or "unknown" if not specified>
}
Rules:
- number_row is derived automatically (3 for type 1 or 2, 6 for type 3 or 4) — do NOT include it
- If the user gives area in metres, convert to mm (multiply by 1000)
- If carport type is not specified or ambiguous, return null for carport_type (do NOT pick a default)
- panel_length_mm: return null UNLESS the user explicitly states a panel length dimension — do NOT guess, infer, or use any default value
- panel_width_mm: return null UNLESS the user explicitly states a panel width dimension — do NOT guess, infer, or use any default value
- panel_power_w: return null UNLESS the user explicitly states a wattage — do NOT guess, infer, or use any default value
- area_length_mm must always be a number — ask yourself if the user stated a carport length
- Always return valid JSON with all 6 fields"""

RESPONSE_SYSTEM = """You are a technical estimator for MetSolar solar carports.
Given the estimation results, write a 3-5 sentence factual summary:
1. State the key figures: system power (kW), total weight (kg), estimated cost (£)
2. If defaults were assumed for any unspecified inputs (panel dimensions, power, carport type, delivery location), state them explicitly — e.g. "Standard 600W panels (2333×1134mm) were assumed as panel size was not specified."
3. Add one brief technical note on installation or grid connection
Tone: direct and factual, like a written engineering estimate. Do NOT use any enthusiastic openers or filler phrases (forbidden: "I'm thrilled", "Great news", "Excited", "Happy to", "I'm pleased", "Certainly", "Of course"). Do not use bullet points. Do not use first person."""


class NLRequest(BaseModel):
    message: str


class PredictRequest(BaseModel):
    carport_type: int
    panel_length_mm: float
    panel_width_mm: float
    panel_power_w: float
    area_length_mm: float
    delivery_artic_cost_gbp: float


@app.get("/")
def root():
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.carport_type not in (1, 2, 3, 4):
        raise HTTPException(status_code=422, detail=f"Invalid carport_type {req.carport_type}. Must be 1, 2, 3, or 4.")
    params = req.model_dump()
    params["number_row"] = 6 if params["carport_type"] in (3, 4) else 3
    validate_model_inputs(params)
    results = predict_from_params(params)
    return {"params_used": params, **results}


@app.post("/estimate")
def estimate(req: NLRequest):
    client = get_groq_client()

    # Step 1: extract structured params from natural language
    try:
        extraction_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": req.message},
            ],
            temperature=0,
            max_tokens=300,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI extraction service error: {e}")
    raw = extraction_resp.choices[0].message.content.strip()

    # extract JSON object — robust to code fences and surrounding prose
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1:
        raise HTTPException(status_code=422, detail=f"No JSON object found in extraction response. Raw: {raw}")
    raw = raw[start:end + 1]

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"Could not parse parameters from message. Raw: {raw}")

    required_fields = {"carport_type", "panel_length_mm", "panel_width_mm", "panel_power_w", "area_length_mm", "delivery_area"}
    missing = required_fields - params.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f"Extraction missing fields: {', '.join(sorted(missing))}")

    # area_length_mm must always be provided — no sensible default exists
    if isinstance(params.get("area_length_mm"), str):
        try:
            params["area_length_mm"] = float(params["area_length_mm"])
        except ValueError:
            pass
    if not isinstance(params.get("area_length_mm"), (int, float)):
        raise HTTPException(status_code=422, detail="Could not determine carport length from your message. Please specify it (e.g. '50 metres long').")

    # apply server-side defaults for null panel fields and track what was assumed
    DEFAULTS = {"panel_length_mm": 2333.0, "panel_width_mm": 1134.0, "panel_power_w": 600.0, "carport_type": 1}
    DEFAULT_LABELS = {
        "panel_length_mm": "panel length 2333mm",
        "panel_width_mm": "panel width 1134mm",
        "panel_power_w": "panel power 600W",
        "carport_type": "carport type SingleMonoIncline",
    }
    assumed = []
    for k, default_val in DEFAULTS.items():
        if params.get(k) is None:
            params[k] = default_val
            assumed.append(DEFAULT_LABELS[k])

    try:
        params["carport_type"] = int(params["carport_type"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"Invalid carport_type {params['carport_type']!r}. Must be 1, 2, 3, or 4.")

    if params["carport_type"] not in (1, 2, 3, 4):
        raise HTTPException(status_code=422, detail=f"Invalid carport_type {params['carport_type']!r}. Must be 1, 2, 3, or 4.")

    # derive number_row from carport_type
    params["number_row"] = 6 if params["carport_type"] in (3, 4) else 3

    # server-side delivery cost lookup — never trust Groq with the numbers
    _NOT_SPECIFIED = {"", "unknown", "none", "null"}
    raw_area = params.pop("delivery_area", None)
    delivery_area_key = raw_area.strip().lower() if isinstance(raw_area, str) else ""
    if delivery_area_key in _NOT_SPECIFIED:
        delivery_area_key = ""

    if delivery_area_key and delivery_area_key in DELIVERY_COSTS:
        params["delivery_artic_cost_gbp"] = DELIVERY_COSTS[delivery_area_key]
        delivery_area = delivery_area_key.capitalize()
        location_defaulted = False
    else:
        params["delivery_artic_cost_gbp"] = DEFAULT_DELIVERY_COST
        delivery_area = DEFAULT_DELIVERY_AREA.capitalize()
        location_defaulted = True

    if location_defaulted:
        if delivery_area_key:
            location_note = f"delivery location ({delivery_area_key.capitalize()} is unsupported, London delivery cost used as default)"
        else:
            location_note = "delivery location (not specified, London delivery cost used as default)"
        assumed.append(location_note)

    # coerce any numeric strings Groq may have returned (e.g. "2333" instead of 2333)
    groq_numeric = {"carport_type", "panel_length_mm", "panel_width_mm", "panel_power_w", "area_length_mm"}
    for field in groq_numeric:
        if isinstance(params.get(field), str):
            try:
                params[field] = float(params[field])
            except ValueError:
                pass  # caught below as non-numeric

    # validate all model inputs are numeric before handing to XGBoost
    numeric_fields = {"carport_type", "panel_length_mm", "panel_width_mm", "panel_power_w",
                      "area_length_mm", "number_row", "delivery_artic_cost_gbp"}
    bad = [f for f in numeric_fields if not isinstance(params.get(f), (int, float))]
    if bad:
        raise HTTPException(status_code=422, detail=f"Non-numeric values for fields: {', '.join(sorted(bad))}")

    validate_model_inputs(params)

    # Step 2: run XGBoost prediction
    try:
        predictions = predict_from_params(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    try:
        power_kw = predictions["total_power_kw"]
        weight_kg = predictions["total_weight_kg"]
        cost_gbp = predictions["total_cost_gbp"]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Model output missing expected target: {e}")

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
        f"Predictions: power={power_kw:.1f} kW, "
        f"weight={weight_kg:.0f} kg, "
        f"cost=£{cost_gbp:.0f}"
    )
    try:
        response_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM},
                {"role": "user", "content": context},
            ],
            temperature=0.4,
            max_tokens=250,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI response service error: {e}")
    friendly = response_resp.choices[0].message.content.strip()

    carport_names = {1: "SingleMonoIncline", 2: "SingleMonoDecline", 3: "DuoMono", 4: "GullWing"}

    return {
        "response": friendly,
        "params_used": {**params, "carport_type_name": carport_names.get(params["carport_type"], "Unknown"), "delivery_area": delivery_area},
        "total_power_kw": round(power_kw, 2),
        "total_weight_kg": round(weight_kg, 1),
        "total_cost_gbp": round(cost_gbp, 2),
    }
