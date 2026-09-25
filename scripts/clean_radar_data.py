"""
PRAVAH ML -- Clean radar data ONLY (Kerala)
=========================================================
Standalone cleaning script. Reads kerala_radar_features.csv, cleans
it, writes kerala_radar_features_clean.csv. Does not touch any other
file or script in the pipeline.

Cleaning steps:
1. Duplicate (district, timestamp_utc) rows dropped, keeping the first
   -- can happen if fetch_radar_data.py was accidentally run twice for
   the same radar frame.
2. Outlier capping:
   - rad_dbz: RainViewer's Universal Blue scheme encodes -32 to 95 dBZ
     -- values outside that range are decoding errors, capped to
     [-32, 95].
   - rad_rainfall_rate_mm_hr: derived from dBZ via Marshall-Palmer, so
     it inherits the same cap; also floored at 0 (rate can't be
     negative).
3. Missing values: left as NaN, not imputed. A missing radar reading
   (pixel had no data, or tile fetch failed) means "we don't know",
   not "no rain" -- filling it in would misrepresent that.
4. Scaling: NOT applied -- same reasoning as the other sources; a
   tree-based model (XGBoost) doesn't need standardized features.

Reminder (not a cleaning issue, just worth repeating here): this file
only contains live snapshots from whenever fetch_radar_data.py has
been run -- there's no historical coverage. Cleaning it doesn't change
that; it just means whatever data does exist is now tidy.

Run:
    python clean_radar_data.py
"""

import pandas as pd

INPUT_PATH = "kerala_radar_features.csv"
OUTPUT_PATH = "kerala_radar_features_clean.csv"

DBZ_MIN, DBZ_MAX = -32.0, 95.0
RATE_MIN = 0.0


def main():
    df = pd.read_csv(INPUT_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    print(f"Loaded {len(df):,} rows from {INPUT_PATH}")

    # ---- Duplicates ------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["district", "timestamp_utc"], keep="first")
    if len(df) < before:
        print(f"[clean] dropped {before - len(df)} duplicate (district, timestamp_utc) rows")

    # ---- Outlier capping: dBZ ------------------------------------------
    below = (df["rad_dbz"] < DBZ_MIN).sum()
    above = (df["rad_dbz"] > DBZ_MAX).sum()
    if below or above:
        print(f"[clean] rad_dbz: capping {below} value(s) below {DBZ_MIN}, "
              f"{above} value(s) above {DBZ_MAX}")
    df["rad_dbz"] = df["rad_dbz"].clip(lower=DBZ_MIN, upper=DBZ_MAX)

    # ---- Outlier capping: rainfall rate (floor at 0, cap consistent with dBZ) --
    rate_cap = None
    if df["rad_rainfall_rate_mm_hr"].notna().any():
        # Recompute the theoretical max rate consistent with DBZ_MAX so the
        # two columns stay physically consistent with each other.
        rate_cap = (10 ** (DBZ_MAX / 10.0) / 200.0) ** (1.0 / 1.6)
    below_rate = (df["rad_rainfall_rate_mm_hr"] < RATE_MIN).sum()
    if below_rate:
        print(f"[clean] rad_rainfall_rate_mm_hr: capping {below_rate} value(s) below {RATE_MIN}")
    df["rad_rainfall_rate_mm_hr"] = df["rad_rainfall_rate_mm_hr"].clip(lower=RATE_MIN, upper=rate_cap)

    # ---- Missing values --------------------------------------------------
    missing_dbz = df["rad_dbz"].isna().sum()
    missing_rate = df["rad_rainfall_rate_mm_hr"].isna().sum()
    print(f"[info] {missing_dbz:,}/{len(df):,} rows missing rad_dbz, "
          f"{missing_rate:,}/{len(df):,} missing rad_rainfall_rate_mm_hr -- left as NaN, not imputed")

    # ---- Sort + save -------------------------------------------------------
    df = df.sort_values(["district", "timestamp_utc"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df):,} cleaned rows to {OUTPUT_PATH}")
    print(df.head())

    print(f"\n[reminder] This file only covers live snapshots collected so far -- "
          f"no historical (2015-2023) radar data exists to clean.")


if __name__ == "__main__":
    main()
