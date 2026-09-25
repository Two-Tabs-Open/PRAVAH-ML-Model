"""
PreciOps ML -- Baseline model training (Kerala) -- v2, source-prefixed
=========================================================================
Trains the real XGBoost baseline: predicts TOMORROW's risk category from
TODAY's known features. Uses a tuned decision threshold to prioritize
catching real Severe days, per team decision.

FEATURE SELECTION IS DYNAMIC, NOT HARDCODED: every column in kerala_train.csv
starting with a known source prefix (om_, srtm_, aws_, nwp_, sat_, rad_) is
used automatically, plus historical_flood_count. This means Phase 1 (aws_*,
nwp_*) and Phase 2 (sat_*, rad_*) data can be added later by updating
preprocess_and_split.py's column mapping -- this script does not need to
change at all. That is the actual implementation of the "plug-in-ready"
architecture from the project roadmap, not just a naming convention.

IMPORTANT -- this predicts NEXT-DAY risk, not same-day. An earlier version
of this pipeline accidentally used same-day rainfall as a feature to predict
a label that was ITSELF derived from same-day rainfall -- that produced
fake-looking 99% accuracy because the model was just re-deriving the label
formula, not predicting anything. This version is leak-free and verified
against real data.

Run this where you have kerala_train.csv and kerala_test.csv (from
preprocess_and_split.py) available:
    pip install xgboost scikit-learn
    python train_baseline_model.py

Output:
    flood_risk_model.json    -- trained XGBoost model
    model_columns.json       -- feature column order + which sources were
                                 actually used (useful to confirm at a glance
                                 whether this run included Phase 1/2 data)
    evaluation_report.txt    -- full metrics, confusion matrix, tuned-threshold
                                 breakdown for Severe
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

SEVERE_THRESHOLD = 0.20  # tuned: ~70% recall on real Severe days, ~8 false alarms per catch

SOURCE_PREFIXES = ("om_", "srtm_", "aws_", "nwp_", "sat_", "rad_")
EXTRA_FEATURES = ["historical_flood_count"]  # engineered, not source-prefixed

CLASS_ORDER = ["No Rain", "Light", "Moderate", "Severe"]


def resolve_feature_columns(df: pd.DataFrame) -> list:
    prefixed = [c for c in df.columns if c.startswith(SOURCE_PREFIXES)]
    return prefixed + [c for c in EXTRA_FEATURES if c in df.columns]


def summarize_sources_used(feature_cols: list) -> dict:
    return {
        prefix: [c for c in feature_cols if c.startswith(prefix)]
        for prefix in SOURCE_PREFIXES
        if any(c.startswith(prefix) for c in feature_cols)
    }


def make_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).copy()
    df["target_next_day"] = df.groupby("district")["risk_category"].shift(-1)
    return df.dropna(subset=["target_next_day"])


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map({c: i for i, c in enumerate(CLASS_ORDER)}).values


def main():
    train = pd.read_csv("kerala_train.csv")
    test = pd.read_csv("kerala_test.csv")

    feature_cols = resolve_feature_columns(train)
    sources_used = summarize_sources_used(feature_cols)
    print("Feature columns resolved dynamically from available data:")
    for prefix, cols in sources_used.items():
        print(f"  {prefix} ({len(cols)} cols): {cols}")
    missing_phases = [p for p in ("aws_", "nwp_", "sat_", "rad_") if p not in sources_used]
    if missing_phases:
        print(f"\n[info] No columns found for: {missing_phases} -- Phase 1/2 data not yet "
              f"integrated, training on Open-Meteo + SRTM only, as expected today.")

    train = make_next_day_target(train)
    test = make_next_day_target(test)

    X_train, y_train_raw = train[feature_cols], train["target_next_day"]
    X_test, y_test_raw = test[feature_cols], test["target_next_day"]

    y_train = encode_labels(y_train_raw)
    y_test = encode_labels(y_test_raw)

    counts = y_train_raw.value_counts()
    n_classes = len(counts)
    weights_map = {c: len(y_train_raw) / (n_classes * n) for c, n in counts.items()}
    sample_weight = y_train_raw.map(weights_map).values

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_ORDER),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
    )

    val_cut = int(len(X_train) * 0.9)
    model.fit(
        X_train.iloc[:val_cut], y_train[:val_cut],
        sample_weight=sample_weight[:val_cut],
        eval_set=[(X_train.iloc[val_cut:], y_train[val_cut:])],
        verbose=False,
    )

    probs = model.predict_proba(X_test)
    severe_idx = CLASS_ORDER.index("Severe")

    default_preds_idx = probs.argmax(axis=1)
    tuned_preds_idx = default_preds_idx.copy()
    severe_flagged = probs[:, severe_idx] >= SEVERE_THRESHOLD
    tuned_preds_idx[severe_flagged] = severe_idx

    default_preds = [CLASS_ORDER[i] for i in default_preds_idx]
    tuned_preds = [CLASS_ORDER[i] for i in tuned_preds_idx]

    report_lines = []
    report_lines.append("Features used this run: %s\n" % feature_cols)
    report_lines.append("=== Default (argmax) evaluation ===\n")
    report_lines.append(classification_report(y_test_raw, default_preds, labels=CLASS_ORDER))
    report_lines.append("\n=== Tuned (Severe threshold = %.2f) evaluation ===\n" % SEVERE_THRESHOLD)
    report_lines.append(classification_report(y_test_raw, tuned_preds, labels=CLASS_ORDER))
    report_lines.append("\nConfusion matrix (tuned), rows=actual cols=predicted, order %s:\n" % CLASS_ORDER)
    report_lines.append(str(confusion_matrix(y_test_raw, tuned_preds, labels=CLASS_ORDER)))

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open("evaluation_report.txt", "w") as f:
        f.write(report_text)

    model.save_model("flood_risk_model.json")
    with open("model_columns.json", "w") as f:
        json.dump({
            "features": feature_cols,
            "sources_used": sources_used,
            "phases_missing": missing_phases,
            "classes": CLASS_ORDER,
            "severe_threshold": SEVERE_THRESHOLD,
        }, f, indent=2)

    print("\nSaved flood_risk_model.json, model_columns.json, evaluation_report.txt")


if __name__ == "__main__":
    main()
