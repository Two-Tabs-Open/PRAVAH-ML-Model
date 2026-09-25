import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client


# Config

# Surface / single-level parameters (no pressure level needed)
SURFACE_PARAMS = ["tp", "tprate", "2t", "2d", "msl", "tcc", "10u", "10v", "sp", "tcwv"]

# Pressure-level parameter(s) -- relative humidity at 850 hPa is a classic
# rain-prediction feature (moisture in the lower-mid atmosphere)
PRESSURE_PARAMS = ["r"]
PRESSURE_LEVELS = [850]

# Forecast lead times (hours) to pull. 0-144 by 3 covers the first 6 days
# at 3-hourly resolution, which is the densest step spacing ECMWF open
# data offers for the 00/12 UTC runs.
STEPS = list(range(0, 145, 3))


def download_forecast(target_grib: str, steps=None, date=None, time=None) -> None:
    """Download global GRIB2 file(s) with the requested parameters."""
    client = Client(source="ecmwf")  # switch to source="aws"/"azure"/"google" if this is slow/unavailable

    steps = steps or STEPS

    # IMPORTANT: only include date/time in the request if the caller
    # actually specified them. The ecmwf-opendata client's date/time
    # parsing calls int(time) unconditionally on whatever value is present
    # in the request dict -- passing date=None/time=None explicitly (as
    # opposed to omitting the keys) makes it crash trying to int(None).
    # Omitting the keys entirely is what makes the client fall back to
    # "latest available run", which is what date=None/time=None were
    # trying to mean in the first place.
    base_kwargs = dict(
        stream="oper",
        type="fc",
        step=steps,
    )
    if date is not None:
        base_kwargs["date"] = date
    if time is not None:
        base_kwargs["time"] = time

    # Surface-level fields
    client.retrieve(
        **base_kwargs,
        param=SURFACE_PARAMS,
        target=target_grib,
    )

    # Pressure-level fields go to a separate file, then get merged
    pl_target = target_grib.replace(".grib2", "_pl.grib2")
    client.retrieve(
        **base_kwargs,
        param=PRESSURE_PARAMS,
        levelist=PRESSURE_LEVELS,
        target=pl_target,
    )
    return pl_target


def extract_point_timeseries(sfc_grib: str, pl_grib: str, lat: float, lon: float) -> pd.DataFrame:
    """Open the GRIB files, pick the nearest grid point, return a tidy DataFrame.

    IMPORTANT: 2t/2d (2m fields) and 10u/10v (10m fields) are both
    typeOfLevel='heightAboveGround' but at DIFFERENT height values.
    cfgrib cannot build one dataset spanning two different heights under
    the same filter -- it raises DatasetBuildError, which the old version
    of this function silently swallowed, dropping 2t/2d entirely without
    any warning. Each (typeOfLevel, level) combination that actually
    differs needs its own filter group.
    """

    # (label, filter_by_keys) -- one group per genuinely distinct hypercube.
    # tcwv's typeOfLevel varies by product; we try a couple of known names.
    filter_groups = [
        ("surface (tp, tprate, sp, tcc)", {"typeOfLevel": "surface"}),
        ("2m fields (2t, 2d)", {"typeOfLevel": "heightAboveGround", "level": 2}),
        ("10m fields (10u, 10v)", {"typeOfLevel": "heightAboveGround", "level": 10}),
        ("mean sea level (msl)", {"typeOfLevel": "meanSea"}),
        ("total column water vapour (tcwv), attempt 1", {"typeOfLevel": "atmosphereSingleLayer"}),
        ("total column water vapour (tcwv), attempt 2", {"typeOfLevel": "entireAtmosphere"}),
    ]

    sfc_datasets = []
    loaded_vars = set()
    failed_groups = []
    for label, filter_kwargs in filter_groups:
        try:
            ds = xr.open_dataset(
                sfc_grib,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_kwargs, "indexpath": ""},
            )
            if len(ds.data_vars) == 0:
                continue  # this product just doesn't have this group, not an error
            sfc_datasets.append(ds)
            loaded_vars.update(ds.data_vars.keys())
        except Exception as e:
            failed_groups.append((label, str(e)))

    pl_ds = xr.open_dataset(
        pl_grib,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}, "indexpath": ""},
    )
    loaded_vars.update(pl_ds.data_vars.keys())

    # Only warn about tcwv if BOTH attempted typeOfLevel names failed --
    # trying one that doesn't match this product isn't a real error.
    tcwv_failures = [f for f in failed_groups if "tcwv" in f[0]]
    other_failures = [f for f in failed_groups if "tcwv" not in f[0]]
    real_failures = other_failures + (tcwv_failures if len(tcwv_failures) == 2 else [])

    if real_failures:
        print("\n[WARNING] Some variable groups failed to load and were skipped:")
        for label, err in real_failures:
            print(f"  - {label}: {err.splitlines()[0] if err else err}")

    expected_vars = {"tp", "tprate", "sp", "tcc", "2t", "2d", "10u", "10v", "msl", "tcwv", "r"}
    missing_vars = expected_vars - loaded_vars
    if missing_vars:
        print(f"\n[WARNING] These requested variables did not make it into the output: {sorted(missing_vars)}")
        print("  Check the failure messages above, or that this ECMWF product actually includes them.")
    else:
        print("\nAll requested variables loaded successfully.")

    frames = []
    for ds in sfc_datasets + [pl_ds]:
        point = ds.sel(latitude=lat, longitude=lon % 360, method="nearest")
        df = point.to_dataframe().reset_index()
        frames.append(df)

    merged = frames[0]
    for f in frames[1:]:
        merge_cols = [c for c in ("time", "step", "valid_time") if c in merged.columns and c in f.columns]
        merged = pd.merge(merged, f, on=merge_cols, how="outer", suffixes=("", "_dup"))

    # Drop duplicate/helper columns cfgrib adds
    drop_cols = [c for c in merged.columns if c.endswith("_dup") or c in ("latitude", "longitude", "number", "surface", "heightAboveGround", "meanSea", "isobaricInhPa")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    if "valid_time" in merged.columns:
        merged = merged.sort_values("valid_time").reset_index(drop=True)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch ECMWF open data for rain prediction")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of your location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of your location (-180 to 180)")
    parser.add_argument("--out", type=str, default="ecmwf_rain_features.csv", help="Output CSV path")
    parser.add_argument("--date", type=str, default=None, help="Forecast run date YYYYMMDD (default: latest)")
    parser.add_argument("--time", type=int, default=None, choices=[0, 6, 12, 18], help="Forecast run hour UTC (default: latest)")
    args = parser.parse_args()

    sfc_grib = "ecmwf_surface.grib2"
    print("Downloading ECMWF open data (this can take a minute)...")
    pl_grib = download_forecast(sfc_grib, date=args.date, time=args.time)

    print("Extracting nearest grid point and building feature table...")
    df = extract_point_timeseries(sfc_grib, pl_grib, args.lat, args.lon)

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
