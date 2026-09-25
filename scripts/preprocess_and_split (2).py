import pandas as pd

TRAINING_DATA_PATH = "kerala_training_data.csv"
TERRAIN_DATA_PATH = "kerala_terrain_features.csv"
SATELLITE_DATA_PATH = "kerala_satellite_features_clean.csv"
RADAR_DATA_PATH = "kerala_radar_features_clean.csv"
SPLIT_DATE = "2024-01-01"

SEVERE_CATEGORIES = {"Heavy", "Very Heavy", "Extreme"}

OM_COLUMN_MAP = {
    "rainfall_mm": "om_rainfall_mm",
    "rainfall_mm_3d_sum": "om_rainfall_mm_3d_sum",
    "rainfall_mm_7d_sum": "om_rainfall_mm_7d_sum",
    "rainfall_mm_15d_sum": "om_rainfall_mm_15d_sum",
    "river_discharge": "om_river_discharge",
    "river_discharge_3d_sum": "om_river_discharge_3d_sum",
    "river_discharge_7d_sum": "om_river_discharge_7d_sum",
    "river_discharge_15d_sum": "om_river_discharge_15d_sum",
}
SRTM_COLUMN_MAP = {
    "elevation_m": "srtm_elevation_m",
    "slope_deg": "srtm_slope_deg",
}


def collapse_label(category: str) -> str:
    return "Severe" if category in SEVERE_CATEGORIES else category


def load_satellite(path: str) -> pd.DataFrame:
    """Already (district, date)-shaped and sat_-prefixed by fetch_satellite_data.py
    -- just needs the date parsed to match the main table's dtype."""
    sat = pd.read_csv(path)
    sat["date"] = pd.to_datetime(sat["date"])
    # Guard against duplicate (district, date) rows (e.g. a re-run that
    # appended instead of skipping) -- keep the first, same convention as
    # clean_satellite_data.py.
    sat = sat.drop_duplicates(subset=["district", "date"], keep="first")
    return sat


def load_radar(path: str) -> pd.DataFrame:
    """RAW SHAPE MISMATCH: radar data is (district, timestamp_utc) -- live
    snapshots, not one row per day. Collapse to (district, date) by taking
    the daily mean, so it can merge into the same daily table as everything
    else. Today this only ever produces one row (2026-09-25), which is
    outside the current train/test range -- but the aggregation is written
    to handle many intraday readings correctly once the script has been run
    repeatedly over time, per its own documented usage pattern."""
    rad = pd.read_csv(path)
    rad["timestamp_utc"] = pd.to_datetime(rad["timestamp_utc"])
    rad["date"] = rad["timestamp_utc"].dt.normalize().dt.tz_localize(None)
    daily = (
        rad.groupby(["district", "date"])[["rad_dbz", "rad_rainfall_rate_mm_hr"]]
        .mean()
        .reset_index()
    )
    return daily


def main():
    daily = pd.read_csv(TRAINING_DATA_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns=OM_COLUMN_MAP)

    terrain = pd.read_csv(TERRAIN_DATA_PATH)
    terrain = terrain.rename(columns=SRTM_COLUMN_MAP)

    missing_terrain = terrain[terrain["srtm_elevation_m"].isna() | terrain["srtm_slope_deg"].isna()]
    if len(missing_terrain):
        print(f"[warn] {len(missing_terrain)} districts have missing terrain data: "
              f"{missing_terrain['district'].tolist()} -- their rows will be dropped.")

    data = daily.merge(terrain, on="district", how="left")
    before = len(data)
    data = data.dropna(subset=["srtm_elevation_m", "srtm_slope_deg"])
    if len(data) < before:
        print(f"Dropped {before - len(data)} rows lacking terrain data.")

    # ---- Satellite merge --------------------------------------------------
    sat = load_satellite(SATELLITE_DATA_PATH)
    data = data.merge(sat, on=["district", "date"], how="left")
    sat_coverage = data["sat_precipitation_mm"].notna().sum()
    print(f"\n[sat_] merged -- {sat_coverage:,}/{len(data):,} rows ({sat_coverage/len(data)*100:.2f}%) "
          f"have a real sat_precipitation_mm value. The rest are NaN (no satellite data collected "
          f"for that date yet).")

    # Radar merge
    rad = load_radar(RADAR_DATA_PATH)
    data = data.merge(rad, on=["district", "date"], how="left")
    rad_coverage = data["rad_dbz"].notna().sum()
    print(f"[rad_] merged -- {rad_coverage:,}/{len(data):,} rows ({rad_coverage/len(data)*100:.2f}%) "
          f"have a real rad_dbz value. Radar has no historical archive (live-only source), so this "
          f"will be at or near 0% until this script is run again after collecting more live snapshots "
          f"over time.")

    data["risk_category_full"] = data["risk_category"]
    data["risk_category"] = data["risk_category_full"].apply(collapse_label)

    data = data.sort_values(["district", "date"]).reset_index(drop=True)

    data["is_severe"] = (data["risk_category"] == "Severe").astype(int)
    data["historical_flood_count"] = (
        data.groupby("district")["is_severe"]
        .transform(lambda s: s.shift(1).expanding().sum())
        .fillna(0)
    )
    data = data.drop(columns=["is_severe"])

    split_date = pd.Timestamp(SPLIT_DATE)
    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    print(f"\nTrain: {len(train):,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    print("\nTrain label distribution:")
    print(train["risk_category"].value_counts())
    print("\nTest label distribution:")
    print(test["risk_category"].value_counts())

    train_sat_cov = train["sat_precipitation_mm"].notna().sum()
    train_rad_cov = train["rad_dbz"].notna().sum()
    print(f"\n[IMPORTANT] Training-set sat_ coverage: {train_sat_cov:,}/{len(train):,} rows "
          f"({train_sat_cov/len(train)*100:.2f}%). Training-set rad_ coverage: {train_rad_cov:,}/{len(train):,} "
          f"rows ({train_rad_cov/len(train)*100:.2f}%). If either is at or near 0%, the model cannot learn "
          f"anything meaningful from that source yet -- this is expected given current data collection "
          f"status, not a bug.")

    class_counts = train["risk_category"].value_counts()
    total = len(train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
    print("\nSuggested class weights:")
    for cls, w in sorted(class_weights.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {w:.2f}")

    train["sample_weight"] = train["risk_category"].map(class_weights)

    train.to_csv("kerala_train.csv", index=False)
    test.to_csv("kerala_test.csv", index=False)
    print("\nSaved kerala_train.csv and kerala_test.csv")

    prefixes = ("om_", "srtm_", "aws_", "nwp_", "sat_", "rad_")
    feature_cols = [c for c in train.columns if c.startswith(prefixes)] + ["historical_flood_count"]
    print(f"\nFeature columns detected ({len(feature_cols)}): {feature_cols}")
    print("(Phase 1 columns -- aws_*, nwp_* -- will appear here automatically once added.)")


if __name__ == "__main__":
    main()
