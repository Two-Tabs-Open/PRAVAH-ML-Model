"""
PRAVAH ML -- Clean satellite data ONLY (Kerala)
=========================================================
Standalone cleaning script. Reads kerala_satellite_features.csv,
cleans it, writes kerala_satellite_features_clean.csv. Does not touch
any other file or script in the pipeline.

Cleaning steps:
1. Duplicate (district, date) rows dropped, keeping the first.
2. Outlier capping: precipitation can't be negative, and India's known
   24h rainfall record is ~1000mm -- values outside [0, 1000] are
   capped to that range rather than dropped (dropping would create
   date gaps in the time series).
3. Missing values: left as NaN, not imputed. A missing satellite
   reading means "we don't know", which is different from "zero
   rain" -- filling it with 0 or a mean would misrepresent that.
   (XGBoost handles NaN natively at training time.)
4. Scaling: NOT applied. XGBoost is a tree-based model and doesn't
   need standardized features -- see the main pipeline's notes if
   you add a non-tree model later.

Run:
    python clean_satellite_data.py
"""

import pandas as pd

INPUT_PATH = "kerala_satellite_features.csv"
OUTPUT_PATH = "kerala_satellite_features_clean.csv"

PRECIP_MIN, PRECIP_MAX = 0.0, 1000.0


def main():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded {len(df):,} rows from {INPUT_PATH}")

    # ---- Duplicates ------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["district", "date"], keep="first")
    if len(df) < before:
        print(f"[clean] dropped {before - len(df)} duplicate (district, date) rows")

    # ---- Outlier capping ---------------------------------------------------
    below = (df["sat_precipitation_mm"] < PRECIP_MIN).sum()
    above = (df["sat_precipitation_mm"] > PRECIP_MAX).sum()
    if below or above:
        print(f"[clean] capping {below} value(s) below {PRECIP_MIN}, "
              f"{above} value(s) above {PRECIP_MAX}")
    df["sat_precipitation_mm"] = df["sat_precipitation_mm"].clip(lower=PRECIP_MIN, upper=PRECIP_MAX)

    # ---- Missing values --------------------------------------------------
    missing = df["sat_precipitation_mm"].isna().sum()
    print(f"[info] {missing:,}/{len(df):,} rows have missing sat_precipitation_mm "
          f"-- left as NaN, not imputed")

    # ---- Sort + save -------------------------------------------------------
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df):,} cleaned rows to {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
