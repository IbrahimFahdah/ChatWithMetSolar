# MetSolar Carport Estimator — Build & Deploy Plan

## What We Are Building

A web app where users describe a solar carport installation in plain English and get
power, weight, and cost estimates back. Groq handles the natural language, XGBoost
handles the predictions, FastAPI serves it all, and Render hosts it for free.

```
User types natural language
        ↓
Groq API / Llama 3  →  extracts structured params as JSON
        ↓
XGBoost models      →  predicts total_power_kw, total_weight_kg, total_cost_gbp
        ↓
Groq API / Llama 3  →  turns numbers into a friendly response
        ↓
User sees result in chat UI
```

---

## Project Structure

```
solar-estimator/
├── app/
│   └── main.py                       FastAPI backend
├── static/
│   └── index.html                    Chat UI frontend
├── model/
│   ├── carport_training_data.csv     Real data from MetSolar calculator
│   ├── train.py                      Trains XGBoost → saves model.pkl
│   └── model.pkl                     Trained model (generated locally)
├── Dockerfile                        Render-ready container
├── render.yaml                       Render deployment config
└── requirements.txt                  Python dependencies
```

---

## Phase 1 — Generate Training Data from the MetSolar Calculator

Training data is produced by running `GenerateTrainingData` in
`CarportDesignerTests.cs` (xUnit Fact). It calls the real MetSolar
`CarportDesigner` + `CarportCostCalculator` across ~15,000 randomly sampled
input combinations and writes `carport_training_data.csv` to the test output
directory. Copy that file into `model/`.

**Input features (model inputs):**

| Column | Type | Range / Values |
|---|---|---|
| carport_type | int | 1=SingleMonoIncline, 2=SingleMonoDecline, 3=DuoMono, 4=GullWing |
| panel_length_mm | float | 1500 – 4000 |
| panel_width_mm | float | 750 – 2500 |
| panel_power_w | float | 200 – 1200 |
| area_length_mm | float | 9000 – 300000 |
| number_row | int | 3 (Single types) or 6 (DuoMono / GullWing) |
| delivery_artic_cost_gbp | float | varies by UK delivery area |

**Output targets (model outputs):**

| Column | Type | Description |
|---|---|---|
| total_power_kw | float | System power output |
| total_weight_kg | float | Total structural + panel weight |
| total_cost_gbp | float | Total installed cost |

**Notes:**
- `number_row` is determined by carport type — not a free input
- `panel_weight` and `panel_thickness_mm` are fixed at 0 and excluded
- `delivery_area` (string) is encoded via its `delivery_artic_cost_gbp` value

---

## Phase 2 — Train the XGBoost Model

Run locally (not on Render — free tier has no GPU and limited CPU).

```bash
pip install xgboost scikit-learn pandas numpy

# Train model — produces model.pkl
python model/train.py
```

`train.py` expects `model/carport_training_data.csv` with these feature columns:

```python
FEATURE_COLS = [
    "carport_type",
    "panel_length_mm",
    "panel_width_mm",
    "panel_power_w",
    "area_length_mm",
    "number_row",
    "delivery_artic_cost_gbp",
]
TARGET_COLS = ["total_power_kw", "total_weight_kg", "total_cost_gbp"]
```

The train script prints MAE and R² per target so you can see accuracy before deploying.
Target: R² above 0.97 and error under 5% of mean value for each output.

To regenerate training data: re-run `GenerateTrainingData` in
`CarportDesignerTests.cs`, copy the new CSV to `model/`, then retrain.

---

## Phase 3 — Build the App

### Backend — app/main.py (FastAPI)

Two endpoints:

**POST /estimate** — natural language input
1. Send user message to Groq (Llama 3), ask it to return JSON params
2. Feed params into XGBoost → get `total_power_kw`, `total_weight_kg`, `total_cost_gbp`
3. Send results back to Groq → get friendly response
4. Return all three estimates + params_used + response text

Groq must extract these fields from the user's message:
- `carport_type` (1–4)
- `panel_length_mm`, `panel_width_mm`, `panel_power_w`
- `area_length_mm`
- `delivery_artic_cost_gbp` (Groq looks up from delivery area name)

**POST /predict** — structured JSON input (for testing without Groq)

**GET /health** — returns {"status":"ok"} for Render health checks

Key environment variable needed: GROQ_API_KEY

### Frontend — static/index.html

Single HTML file, no framework, no build step.
- Chat-style interface
- User types or picks an example prompt
- Calls POST /estimate
- Displays power (kW), weight (kg), and cost (£) in a result card

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/       ./app/
COPY static/    ./static/
COPY model/model.pkl ./model/model.pkl
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
fastapi==0.111.0
uvicorn==0.30.1
xgboost==2.0.3
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
groq==0.9.0
python-multipart==0.0.9
```

---

## Phase 4 — Run Locally to Test

```bash
export GROQ_API_KEY=your_key_here   # free at console.groq.com

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 and test a few prompts.
Also test the structured endpoint directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "carport_type": 1,
    "panel_length_mm": 2333,
    "panel_width_mm": 1134,
    "panel_power_w": 600,
    "area_length_mm": 29000,
    "number_row": 3,
    "delivery_artic_cost_gbp": 850
  }'
```

---

## Phase 5 — Deploy to Render

1. Push the project folder to a GitHub repo
   - Make sure model.pkl is committed (remove it from .gitignore)
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render detects the Dockerfile automatically
5. Add environment variable: GROQ_API_KEY = your key
6. Click Deploy

render.yaml handles the rest:
```yaml
services:
  - type: web
    name: solar-estimator
    runtime: docker
    plan: free
    envVars:
      - key: GROQ_API_KEY
        sync: false
    healthCheckPath: /health
```

Render free tier note: the service spins down after 15 min of inactivity.
First request after sleep takes ~30 seconds. Upgrade to Starter ($7/mo) to
keep it always-on once you share it with users.

---

## Phase 6 — Regenerate Data and Retrain (when needed)

1. Run `GenerateTrainingData` test in `CarportDesignerTests.cs`
2. Copy the output `carport_training_data.csv` to `model/`
3. Retrain: `python model/train.py`
4. Commit new `model.pkl` → Render auto-redeploys

If the MetSolar calculator logic changes (new carport types, updated pricing),
re-run the generator to pick up the new outputs automatically.

---

## Accuracy Expectations

| Scenario | Expected error |
|---|---|
| Inputs within training range | 2 – 5% |
| Inputs near edge of training range | 5 – 10% |
| Inputs outside training range | unpredictable |

For a rough structural estimate this is more than sufficient.

---

## Cost

| Service | Cost |
|---|---|
| Render free tier | $0 |
| Groq API | Free tier: 14,400 requests/day — plenty for this use case |
| Total to start | $0 |
