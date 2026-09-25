"""
PRAVAH ML -- Live radar features (Kerala) -- RainViewer
=========================================================
Fetches the CURRENT weather radar snapshot from RainViewer (free, no
signup) and extracts a rainfall-intensity value per Kerala district,
appending one row per run to a running CSV -- ready to merge in as
rad_* columns once you have enough live history collected.

Data source: RainViewer public Weather Maps API
https://api.rainviewer.com/public/weather-maps.json
License/terms: free for personal/educational/small-scale use.
Required attribution if you show this publicly: "Weather data by
Rain Viewer" with a link to rainviewer.com.

--------------------------------------------------------------------
IMPORTANT LIMITATION -- READ THIS FIRST
--------------------------------------------------------------------
RainViewer only keeps the past ~2 hours of radar frames. There is NO
historical archive. This means:
  - This script CANNOT backfill 2015-2023 radar data to match your
    existing om_/srtm_/sat_ training history.
  - It only produces data FROM THE MOMENT YOU START RUNNING IT ONWARD.
  - To build up a useful radar history, run this script repeatedly
    (e.g. every 10-15 minutes) over time and let the output CSV grow.
    A single run gives you one data point, which is not enough to
    train on -- treat this as starting a live collection process, not
    a one-shot historical fetch like the other scripts.
  - Realistically, this is better suited to a future LIVE INFERENCE
    feature (radar right now, for today's prediction) than to
    retraining the existing historical model immediately.

--------------------------------------------------------------------
INSTALL
--------------------------------------------------------------------
pip install requests Pillow pandas

--------------------------------------------------------------------
HOW THE DECODING WORKS
--------------------------------------------------------------------
RainViewer serves radar as colored map tiles (images), not raw numbers.
Since Jan 2026 they only offer the "Universal Blue" color scheme
(ID 2), which encodes dBZ (radar reflectivity) directly in the pixel's
red channel:
    dBZ = (red_value & 127) - 32
    (bit 7 set means snow instead of rain; we ignore snow here)
    red_value 0 means "no data" for that pixel

dBZ is then converted to an estimated rainfall rate (mm/hr) using the
standard Marshall-Palmer Z-R relationship:
    R = (10^(dBZ/10) / 200) ^ (1/1.6)

This is a well-established meteorological approximation, not an exact
measurement -- treat rad_rainfall_rate_mm_hr as an estimate.
"""

import math
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

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

WEATHER_MAPS_URL = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM = 8       # tile zoom level -- 8 gives ~1.5km/pixel resolution, plenty for district-level
TILE_SIZE = 256
COLOR_SCHEME = 2  # "Universal Blue" -- the only scheme available since Jan 2026


def latlon_to_tile(lat: float, lon: float, zoom: int):
    """Standard slippy-map (Web Mercator) lat/lon -> tile x/y, plus the
    fractional pixel position within that tile."""
    n = 2 ** zoom
    lat_rad = math.radians(lat)

    x_float = (lon + 180.0) / 360.0 * n
    y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    xtile = int(x_float)
    ytile = int(y_float)

    px = int((x_float - xtile) * TILE_SIZE)
    py = int((y_float - ytile) * TILE_SIZE)

    return xtile, ytile, px, py


def dbz_to_rainfall_rate(dbz: float) -> float:
    """Marshall-Palmer Z-R relationship: Z = 200 * R^1.6, dBZ = 10*log10(Z)."""
    if dbz is None:
        return None
    z = 10 ** (dbz / 10.0)
    r = (z / 200.0) ** (1.0 / 1.6)
    return round(r, 3)


def get_latest_frame():
    resp = requests.get(WEATHER_MAPS_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    host = data["host"]
    past_frames = data.get("radar", {}).get("past", [])
    if not past_frames:
        raise RuntimeError("No radar frames currently available from RainViewer.")

    latest = past_frames[-1]  # most recent frame
    return host, latest


def fetch_tile(host: str, frame_path: str, xtile: int, ytile: int):
    url = f"{host}{frame_path}/{TILE_SIZE}/{ZOOM}/{xtile}/{ytile}/{COLOR_SCHEME}/1_1.png"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")


def decode_pixel_to_dbz(pixel) -> float:
    r, g, b, a = pixel
    if r == 0 or a == 0:
        return None  # no data at this point
    value = r & 127  # mask off the snow bit (bit 7)
    return float(value - 32)


def main():
    out_path = "kerala_radar_features.csv"

    host, frame = get_latest_frame()
    frame_time = datetime.fromtimestamp(frame["time"], tz=timezone.utc)
    print(f"Latest radar frame: {frame_time.isoformat()}")

    # Cache tiles we've already downloaded this run (districts can share a tile at low zoom)
    tile_cache = {}
    rows = []

    for district, coords in KERALA_DISTRICTS.items():
        xtile, ytile, px, py = latlon_to_tile(coords["lat"], coords["lon"], ZOOM)

        key = (xtile, ytile)
        if key not in tile_cache:
            try:
                tile_cache[key] = fetch_tile(host, frame["path"], xtile, ytile)
            except Exception as e:
                print(f"  [warn] could not fetch tile for {district}: {e}")
                tile_cache[key] = None

        img = tile_cache[key]
        if img is None:
            dbz = None
        else:
            pixel = img.getpixel((px, py))
            dbz = decode_pixel_to_dbz(pixel)

        rate = dbz_to_rainfall_rate(dbz)

        rows.append({
            "district": district,
            "timestamp_utc": frame_time.isoformat(),
            "rad_dbz": dbz,
            "rad_rainfall_rate_mm_hr": rate,
        })

    df = pd.DataFrame(rows)

    header_needed = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", header=header_needed, index=False)

    print(f"\nAppended {len(df)} rows to {out_path} for frame {frame_time.isoformat()}")
    print(df)
    print("\nAttribution reminder: RainViewer requires 'Weather data by Rain Viewer' "
          "credit with a link to rainviewer.com wherever this is shown publicly.")


if __name__ == "__main__":
    main()
