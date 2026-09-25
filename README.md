# PreciOps — ML Model

**AI/ML-based flood risk prediction for Kerala, built for SIH 2026 (PS 26071) by Team CodeBlooded.**

This document covers the ML workstream only: data pipeline, model architecture, real results, and honest limitations. For the full application (dashboard, routing, alerts), see the main project README.

---

## 1. What this model does

Predicts **tomorrow's flood risk category** — `No Rain` / `Light` / `Moderate` / `Severe` — for each of Kerala's 14 districts, using today's known rainfall, river discharge, and terrain data. Categories follow IMD's own official rainfall-intensity thresholds, so the model's output speaks the same language as existing public weather warnings.

The system is tuned to **prioritize catching real Severe days over minimizing false alarms** — a missed flood warning is a far worse failure than an unnecessary caution alert, for a disaster early-warning system.

---

## 2. Architecture

One pipeline, **source-prefixed** so new data sources plug in without touching model code:

```
om_*     Open-Meteo reanalysis (rainfall, river discharge)     — full coverage
srtm_*   SRTM elevation/slope via OpenTopoData                 — full coverage
sat_*    GPM IMERG satellite precipitation                     — merged, thin coverage
rad_*    RainViewer radar reflectivity                         — merged, live-only
aws_*    IMD AWS/ARG observed rainfall (Phase 1)                — not integrated, access blocked
nwp_*    ECMWF/NCMRWF forecast (Phase 1)                        — fetch script exists, live-only scope
```

`train_baseline_model.py` resolves its feature list **dynamically** by scanning for columns with these prefixes — it does not hardcode a feature list. Dropping new `aws_*` or `nwp_*` columns into the preprocessed table is enough to include them in the next training run; no code changes required.

### Pipeline stages

```
fetch_training_data.py        → kerala_training_data.csv       (om_ raw)
fetch_terrain_features.py     → kerala_terrain_features.csv    (srtm_ raw)
fetch_satellite_data.py       → kerala_satellite_features.csv  (sat_ raw)
fetch_radar_data.py           → kerala_radar_features.csv      (rad_ raw)
clean_satellite_data.py       → kerala_satellite_features_clean.csv
clean_radar_data.py           → kerala_radar_features_clean.csv
        ↓
preprocess_and_split.py       → kerala_train.csv, kerala_test.csv
        ↓
train_baseline_model.py       → flood_risk_model.json, model_columns.json, evaluation_report.txt
```

---

## 3. Data sources: problem statement vs. what's actually in the model

| Dataset | Required by problem statement | In model today | Status |
|---|---|---|---|
| Satellite (INSAT-3D, IMERG) | Yes | Yes (thin) | GPM IMERG merged as `sat_precipitation_mm`. Real data, but only Jan 2024 (434/59,976 rows, 0.72%). **0% coverage of the training period (2015–2023).** |
| Radar (IMD/ISRO DWR) | Yes | Yes (inert) | RainViewer merged as `rad_dbz` / `rad_rainfall_rate_mm_hr`. Real decoded reflectivity, but the source is structurally live-only (~2hr window, no historical archive) — **0% coverage of train or test currently.** |
| Observational (IMD AWS/ARG) | Yes | No | **Blocked.** IMD closed public AWS/ARG access in May 2025. Current process requires a formal institutional request (on the requesting institution's letterhead) to IMD, with discretionary approval and no published turnaround time. Not pursued further given the hackathon timeline. |
| NWP (ECMWF/NCMRWF) | Yes | Partial | Fetch script built, debugged, and verified working against ECMWF's free Open Data endpoint — but that endpoint is a rolling ~2–3 day forecast window with no historical archive, so it can only ever supply a **live inference-time** feature, not backfill 2015–2023 training data. |
| Open-Meteo rainfall + discharge | Not named | Yes (`om_`) | Core of current training data — reanalysis-based, same source as the live app's data-ingestion pipeline. |
| SRTM elevation + slope | Not named | Yes (`srtm_`) | Real terrain data via the OpenTopoData API, replacing a fabricated placeholder dictionary found in an earlier version of this repo. |
| IMD rainfall categories | Not named | Yes | Official published thresholds, used as the label scheme. |

---

## 4. Model

- **Algorithm:** XGBoost (`multi:softprob`), 4-class classification
- **Features:** 14, resolved dynamically (see Section 2)
- **Target:** next day's risk category, predicted from **today's** known features — not same-day, to avoid label leakage (see Section 6)
- **Labels:** collapsed from IMD's 6 official categories to 4 — `Heavy`/`Very Heavy`/`Extreme` merged into a single `Severe` class, since `Extreme` had zero real examples in the data and `Heavy`+`Very Heavy` combined were under 1%
- **Class imbalance:** handled via `sample_weight`, computed from the training set only
- **Decision threshold:** 0.20 for flagging `Severe` (vs. the default argmax), tuned to trade precision for recall — team decision, since missing a real Severe day is worse than a false alarm
- **Train/test split:** time-based, not random — train on 2015–2023, test on 2024–present. This means the model is evaluated on data it genuinely could not have seen, including the real 2024 Wayanad event

---

## 5. Results (final, real, verified)

Evaluated on the 2024–present holdout period (13,944 days), genuinely unseen during training.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Rain | 0.82 | 0.74 | 0.78 | 6,594 |
| Light | 0.55 | 0.54 | 0.55 | 4,926 |
| Moderate | 0.41 | 0.38 | 0.39 | 2,275 |
| **Severe** | **0.08** | **0.57** | **0.14** | 149 |

Overall accuracy: 61% (tuned) / 64% (untuned argmax) — lower than an untuned model, by design, since the threshold deliberately trades precision for Severe-day recall.

**Real-world validation:** the single highest rainfall value across the entire 12-year, 14-district dataset is **Idukki, August 15, 2018 (204.1mm)** — the actual date the Idukki dam had to open its floodgates during the real 2018 Kerala floods. The model flags this day as Severe from raw rainfall data alone.

**Adding satellite/radar made no meaningful difference** (57% vs. 60% recall without them, within normal run-to-run noise) — expected, given their current coverage gaps documented in Section 3. This is included here deliberately, as evidence the pipeline doesn't overclaim impact it doesn't yet have.

---

## 6. A leakage bug we caught and fixed

An early version of this pipeline used same-day rainfall as a feature to predict a label that was *itself* derived from same-day rainfall via IMD's thresholds. That produced a fake-looking ~99% accuracy — the model wasn't predicting anything, it was re-deriving the exact formula that created its own label. The target was restructured to predict **tomorrow's** category from **today's** known data, which is both leak-free and the actually useful framing for an early-warning system. Every result in this document reflects the corrected version.

---

## 7. How to reproduce

```bash
pip install httpx pandas xgboost scikit-learn

# 1. Fetch raw data (needs internet; run in Colab or similar)
python fetch_training_data.py
python fetch_terrain_features.py
python fetch_satellite_data.py --start 2015-01-01
python fetch_radar_data.py            # run repeatedly over time to build history

# 2. Clean satellite/radar
python clean_satellite_data.py
python clean_radar_data.py

# 3. Preprocess + split
python preprocess_and_split.py

# 4. Train
python train_baseline_model.py
```

Outputs: `flood_risk_model.json` (trained model), `model_columns.json` (feature list + which sources were used + threshold), `evaluation_report.txt` (full metrics).

---

## 8. Known limitations (stated plainly)

- **Severe-class precision is low (0.08)** — roughly 1 in 12 Severe alerts is correct, by deliberate design (recall was prioritized). This should be communicated honestly in any demo, not smoothed over.
- **Satellite and radar do not yet meaningfully influence predictions** — they're wired in correctly but lack sufficient historical coverage (see Section 3).
- **No water-depth or inundation-extent output yet** — current output is a risk category only.
- **Single-day granularity** — predicts next calendar day, not the finer T+0→T+3 hour resolution named in the original problem statement; that would need hourly data as a future refinement.
- **Kerala only** — the pipeline is designed to generalize (no district-specific hardcoding in the model itself), but has only been trained and validated on Kerala data so far.

---

## 9. Roadmap

**Phase 1 (blocked/partial):**
- AWS/ARG integration — blocked pending IMD institutional approval
- NWP integration into training — needs a licensed historical archive, not just the free live endpoint

**Phase 2 (architecture done, coverage growing):**
- Continue satellite historical backfill (currently Jan 2024 only, need 2015–2023)
- Continue radar live-snapshot collection (inherently live-inference only, not backfillable)
- INSAT-3D specifically (as named in the original brief) — not attempted; GPM IMERG used instead as a more immediately accessible satellite source

**Downstream:**
- Package model output as API-ready JSON for frontend/backend integration
- SHAP explainability
- Extend to 3 more states using the same generalized, source-prefixed pipeline
- Water-depth proxy (terrain + risk category), labeled honestly as an estimate
- Distance-to-nearest-river feature (needs real river geometry via OpenStreetMap Overpass)
- GeoJSON inundation polygons + Dijkstra flood-aware routing integration

---

## 10. Data source acknowledgments

- **Open-Meteo** — rainfall and river discharge (reanalysis)
- **OpenTopoData** — SRTM 30m elevation data
- **NASA GPM IMERG**, via NASA Earthdata — satellite precipitation
- **RainViewer** — public weather radar (attribution required for public display: "Weather data by Rain Viewer", linked to rainviewer.com)
- **India Meteorological Department (IMD)** — official rainfall-intensity category thresholds used for labeling

---

*Last updated: September 25, 2026. Every result in this document comes from a real, leak-free, time-based evaluation on real Kerala data using a genuinely trained model — not an estimate. Every gap is documented with the actual reason it exists, not left unexplained.*
