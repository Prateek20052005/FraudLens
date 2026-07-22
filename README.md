# FraudLens

Real-time credit card fraud detection with **explainable** predictions.

FraudLens scores a transaction for fraud risk and returns the specific features that drove the score. A probability on its own is not actionable — an analyst reviewing a flagged transaction needs to know *why* it was flagged. Every prediction from this service ships with a SHAP-based breakdown of the top contributing features and the direction of their influence.

---

## Architecture

```
                  ┌──────────────────────┐
                  │  Streamlit Dashboard │
                  │  (live monitor, ad-  │
                  │  hoc analysis, stats)│
                  └──────────┬───────────┘
                             │ HTTP / JSON
                             ▼
                  ┌──────────────────────┐
                  │     FastAPI Service   │
                  │  /health /predict     │
                  │        /stats         │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌────────────────────┐        ┌────────────────────┐
   │  Feature pipeline  │        │  LightGBM model    │
   │  (time, amount,    │───────▶│  + SHAP explainer  │
   │  email, identity)  │        │  (model_v5.pkl)    │
   └────────────────────┘        └────────────────────┘
```

The model and its feature schema are loaded **once at process startup**, not per request. The serialized feature list (`feature_cols_v5.json`) is shipped alongside the model so that the column set and ordering used at inference exactly match training — guarding against training/serving skew.

---

## Features

- **Real-time scoring** — single-transaction inference over a REST API
- **Explainability built in** — SHAP `TreeExplainer` returns the ten highest-impact features per prediction, each labelled as increasing or decreasing risk
- **Risk bucketing** — raw probability mapped to `LOW` / `MEDIUM` / `HIGH` bands for triage
- **Schema validation** — Pydantic models reject malformed payloads at the API boundary before they reach the model
- **Health endpoint** — reports service status, loaded feature count, and predictions served, so a supervisor or orchestrator can distinguish "process running" from "model actually loaded"
- **Versioned artifacts** — model and feature schema are versioned together
- **Interactive dashboard** — simulated live transaction stream, ad-hoc transaction analysis, and served-prediction statistics

---

## Tech stack

| Layer | Choice |
|---|---|
| Model | LightGBM (gradient-boosted trees) |
| Explainability | SHAP (`TreeExplainer`) |
| Serving | FastAPI + Uvicorn |
| Validation | Pydantic |
| Dashboard | Streamlit + Plotly |
| Data | pandas, NumPy, Parquet |
| Training | Jupyter notebooks |

---

## Repository structure

```
FraudLens/
├── api/
│   └── app.py              # FastAPI service: feature pipeline + endpoints
├── dashboard/
│   └── app.py              # Streamlit dashboard (API client)
├── models/
│   ├── model_v5.pkl        # Serialized LightGBM model
│   └── feature_cols_v5.json# Feature schema (names + ordering)
├── notebooks/              # EDA, feature engineering, model training
├── data/                   # Sample transactions (Parquet)
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.9+
- Trained artifacts present in `models/`

### Installation

```bash
git clone https://github.com/Prateek20052005/FraudLens.git
cd FraudLens

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

Interactive API docs are then available at `http://127.0.0.1:8000/docs`.

### Run the dashboard

In a second terminal, with the API already running:

```bash
streamlit run dashboard/app.py
```

The dashboard reads the API base URL from the `API_URL` environment variable, defaulting to `http://127.0.0.1:8000`.

---

## API reference

### `GET /health`

Liveness and readiness check.

```json
{
  "status": "healthy",
  "model_features": 214,
  "predictions_served": 42
}
```

### `POST /predict`

Scores a single transaction.

**Request**

```json
{
  "TransactionAmt": 149.99,
  "TransactionDT": 86400,
  "card1": 13553,
  "C1": 1.0, "C2": 1.0, "C3": 0.0, "C4": 0.0, "C5": 0.0,
  "C6": 1.0, "C7": 0.0, "C8": 0.0, "C9": 1.0, "C10": 0.0,
  "C11": 1.0, "C12": 0.0, "C13": 1.0, "C14": 1.0,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "DeviceType": "mobile",
  "id_31": "chrome 65.0"
}
```

**Response**

```json
{
  "fraud_probability": 0.8123,
  "risk_level": "HIGH",
  "transaction_amount": 149.99,
  "reasons": [
    {
      "feature": "TransactionAmt_log",
      "value": 5.0173,
      "impact": 0.9142,
      "direction": "increases_risk"
    },
    {
      "feature": "is_night",
      "value": 1.0,
      "impact": 0.4417,
      "direction": "increases_risk"
    }
  ]
}
```

Risk bands: `HIGH` above 0.7, `MEDIUM` above 0.3, `LOW` otherwise. These thresholds are configurable and should be tuned to the relative business cost of a false positive (blocking a legitimate customer) versus a false negative (allowing fraud through).

### `GET /stats`

Aggregate statistics across predictions served by the current process.

```json
{
  "total_predictions": 42,
  "avg_fraud_probability": 0.1874,
  "max_fraud_probability": 0.9612,
  "avg_transaction_amount": 132.45,
  "risk_distribution": { "LOW": 35, "MEDIUM": 5, "HIGH": 2 }
}
```

---

## Model and features

Trained on the IEEE-CIS Fraud Detection transaction and identity tables.

Engineered features include:

- **Temporal** — hour of day, weekday, night-time flag derived from the transaction timestamp
- **Amount** — log transform, decimal component, round-amount flag
- **Email** — ISP-domain flags for purchaser and recipient, domain-match indicator, missingness flags
- **Identity** — device presence, mobile flag, normalized browser family
- **Interactions** — amount × night, amount × mobile, and ratios between counting features

LightGBM handles missing values natively, so absent optional fields are passed through as `NaN` rather than imputed.

> **Evaluation:** _fill in your held-out ROC-AUC and PR-AUC here._ Because the positive class is roughly 3.5% of transactions, accuracy is not a meaningful metric for this problem — ranking metrics and the precision-recall tradeoff are what matter.

---

## Known limitations and roadmap

This section is deliberate. These are understood tradeoffs and open items, not unknowns.

| Limitation | Impact | Planned fix |
|---|---|---|
| Categorical features are hash-encoded at inference rather than using persisted encoders | Python's string hashing is salted per process, so encodings can differ between training and serving | Fit and serialize label encoders during training; load them with the model |
| Card- and address-level aggregate features are stubbed at inference | The model was trained on real historical aggregates but receives placeholders at serving time | Introduce a feature store providing per-card historical statistics at request time |
| `prediction_log` is an unbounded in-process list | Grows without limit, is lost on restart, and is not shared across replicas | Bounded buffer plus an external store (Redis or a time-series database) |
| SHAP is computed on every request | Explanation dominates request latency relative to the prediction itself | Compute explanations only for MEDIUM/HIGH scores, or return them asynchronously |
| Single-process deployment | No horizontal scaling story yet | Containerize API and dashboard separately; scale the stateless API behind a load balancer |
| No automated tests or CI | Regressions are not caught | Unit tests for the feature pipeline, contract tests for the API, GitHub Actions pipeline |

---

## License

MIT
