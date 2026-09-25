"""
PRAVAH ML -- Satellite precipitation features (Kerala)
=========================================================
Fetches NASA GPM IMERG (Final Run, daily) satellite precipitation data
and extracts a value per Kerala district per day, in the same
(district, date) shape as kerala_training_data.csv, so it merges
straight into the existing pipeline as sat_* columns.

Data source: NASA GPM IMERG (satellite-based, NOT radar)
Access: NASA Earthdata Cloud, via the earthaccess library

--------------------------------------------------------------------
ONE-TIME SETUP (before running this script)
--------------------------------------------------------------------
1. Free Earthdata account: https://urs.earthdata.nasa.gov/users/new
2. Log in there, go to Applications -> Authorized Apps, and approve
   "NASA GESDISC DATA ARCHIVE" (required, or downloads will fail).

--------------------------------------------------------------------
INSTALL
--------------------------------------------------------------------
pip install earthaccess xarray netCDF4 pandas

--------------------------------------------------------------------
WHY THIS VERSION IS DIFFERENT (v2 -- Drive-safe)
--------------------------------------------------------------------
The first version downloaded every raw NetCDF file for the whole date
range before extracting anything, and someone (correctly) pointed
--raw_dir at a Google Drive folder -- so it filled the Drive.

This version fixes that in two ways:
1. Raw files are downloaded to LOCAL Colab disk by default (/content/...
   not a Drive path), which is temporary and does NOT count against
   your Drive storage quota.
2. Processing happens in monthly CHUNKS: download a month's worth of
   files, extract the district values, append to the running output
   CSV, then DELETE that month's raw files before moving to the next
   month. So raw data never piles up -- at most ~1 month of NetCDF
   files exist on disk at any moment (a few hundred MB, not gigabytes).

Only the small output CSV (a few hundred KB at most) needs to live on
Drive -- point --out at a Drive path for that, and leave --raw_dir on
local disk (the default).

--------------------------------------------------------------------
WHAT THIS SCRIPT DOES
--------------------------------------------------------------------
For each month in your date range:
  1. Searches GPM_3IMERGDF granules for that month over Kerala's bbox.
  2. Downloads them to local disk.
  3. Samples the "precipitation" variable at each of the 14 district
     centroids (V07 files store this at the file root, no sub-groups --
     confirmed against an actual downloaded file; V06 docs describe a
     different structure/"precipitationCal" name that does not apply here).
  4. Appends those rows to the output CSV.
  5. Deletes the month's raw files.

Honest limitations:
- IMERG Final Run has ~3.5 month latency -- fine for historical
  training data, not for live/current-day inference.
- Satellite rainfall estimates are less accurate than ground station
  or radar data, especially over complex terrain like the Western
  Ghats -- treat this as a supplementary feature, not a replacement
  for om_ rainfall.
- If Colab disconnects mid-run, the CSV already has every month
  completed so far (it's appended incrementally) -- just re-run the
  same command and it will skip months already saved.
"""

import argparse
import os
import re
import shutil

import pandas as pd
import xarray as xr

try:
    import earthaccess
except ImportError:
    raise SystemExit("Run: pip install earthaccess xarray netCDF4 pandas")

KERALA_DISTRICTS = {
    "Thiruvananthapuram": {"lat": 8.52, "lon": 76.93},
    "Kollam": {"lat": 8.89, "lon": 76.61},
    "Pathanamthitta": {"lat": 9.26, "lon": 76.78},
    "Alappuzha": {"lat": 9.49, "lon": 76.33},
    "Kottayam": {"lat": 9.59, "lon": 76.52},
    "Idukki": {"lat": 9.85, "lon": 76.94},
    "Ernakulam": {"lat": 9.98, "lon": 76.28},
    "Thrissur": {"lat": 10.52, "lon": 76.21},
    "Palakkad": {"lat": 10.78, "lon": 76.65},
    "Malappuram": {"lat": 11.07, "lon": 76.07},
    "Kozhikode": {"lat": 11.25, "lon": 75.78},
    "Wayanad": {"lat": 11.68, "lon": 76.13},
    "Kannur": {"lat": 11.87, "lon": 75.37},
    "Kasaragod": {"lat": 12.49, "lon": 74.98},
}

# (west, south, east, north)
KERALA_BBOX = (74.8, 8.2, 77.4, 12.8)

SHORT_NAME = "GPM_3IMERGDF"  # daily, Final Run
VERSION = "07"


def month_ranges(start: str, end: str):
    """Yield (month_start, month_end) date strings covering [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cur = start_ts.replace(day=1)
    while cur <= end_ts:
        month_end = cur + pd.offsets.MonthEnd(0)
        chunk_start = max(cur, start_ts)
        chunk_end = min(month_end, end_ts)
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = cur + pd.offsets.MonthBegin(1)


def extract_date_from_filename(path: str):
    """IMERG filenames embed the date, e.g. ...3IMERG.20230615-S000000-E235959.V07.nc4"""
    m = re.search(r"3IMERG\.(\d{8})", os.path.basename(path))
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return None


def extract_district_values(files: list) -> pd.DataFrame:
    rows = []
    for f in sorted(files):
        date = extract_date_from_filename(f)
        try:
            # V07 files have no sub-groups (everything at root level), and
            # the precipitation variable is named "precipitation" here,
            # not "precipitationCal" as in older V06 documentation.
            ds = xr.open_dataset(f)
        except Exception as e:
            print(f"    [warn] could not open {f}: {e}")
            continue

        for district, coords in KERALA_DISTRICTS.items():
            try:
                point = ds["precipitation"].sel(
                    lon=coords["lon"], lat=coords["lat"], method="nearest"
                )
                value = float(point.values.squeeze())
            except Exception:
                value = None

            rows.append({
                "district": district,
                "date": date,
                "sat_precipitation_mm": value,
            })
        ds.close()

    return pd.DataFrame(rows)


def already_done_months(out_path: str):
    """If out_path exists, return the set of year-month strings already saved."""
    if not os.path.exists(out_path):
        return set()
    existing = pd.read_csv(out_path, parse_dates=["date"])
    return set(existing["date"].dt.to_period("M").astype(str))


def main():
    parser = argparse.ArgumentParser(description="Fetch GPM IMERG satellite precipitation for Kerala districts (Drive-safe, chunked)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--raw_dir", type=str, default="/content/imerg_raw_tmp",
                         help="LOCAL (non-Drive) scratch folder for raw files -- deleted after each month")
    parser.add_argument("--out", type=str, default="kerala_satellite_features.csv",
                         help="Output CSV path -- put this on Drive, e.g. the PRAVAH_data folder")
    args = parser.parse_args()

    if "drive" in args.raw_dir.lower() or "mydrive" in args.raw_dir.lower():
        print("[warn] --raw_dir looks like it's on Google Drive. This defeats the point of "
              "chunked downloading -- raw files should go on local Colab disk (e.g. /content/...). "
              "Continuing anyway, but consider changing --raw_dir.")

    earthaccess.login()  # prompts once, then cached for the session

    done_months = already_done_months(args.out)
    if done_months:
        print(f"Resuming: {len(done_months)} month(s) already in {args.out}, will skip those.")

    header_written = os.path.exists(args.out)

    for m_start, m_end in month_ranges(args.start, args.end):
        month_key = pd.Timestamp(m_start).strftime("%Y-%m")
        if month_key in done_months:
            print(f"=== {month_key} -- already done, skipping ===")
            continue

        print(f"=== {month_key} ({m_start} to {m_end}) ===")
        results = earthaccess.search_data(
            short_name=SHORT_NAME,
            version=VERSION,
            temporal=(m_start, m_end),
            bounding_box=KERALA_BBOX,
        )
        print(f"  Found {len(results)} granules.")
        if not results:
            continue

        os.makedirs(args.raw_dir, exist_ok=True)
        files = earthaccess.download(results, local_path=args.raw_dir)

        print("  Extracting per-district values...")
        month_df = extract_district_values(files)
        month_df = month_df.sort_values(["district", "date"]).reset_index(drop=True)

        month_df.to_csv(args.out, mode="a", header=not header_written, index=False)
        header_written = True
        print(f"  Appended {len(month_df)} rows to {args.out}")

        # Free the local disk space before moving to the next month
        shutil.rmtree(args.raw_dir, ignore_errors=True)

    print(f"\nDone. Final file: {args.out}")
    if os.path.exists(args.out):
        final = pd.read_csv(args.out)
        print(f"Total rows so far: {len(final):,}")
        print(final.head())


if __name__ == "__main__":
    main()
