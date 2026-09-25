import asyncio
import math
import httpx
import pandas as pd

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

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
OFFSET_KM = 5.0  # distance for the 4 sample points used to estimate slope


def offset_point(lat, lon, bearing_deg, distance_km):
    """Move a lat/lon point a given distance (km) along a compass bearing."""
    R = 6371.0  # Earth radius km
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / R)
        + math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
        math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


async def query_elevations(client: httpx.AsyncClient, points: list):
    """points: list of (lat, lon). Returns list of elevations in meters, same order."""
    locations_param = "|".join(f"{lat},{lon}" for lat, lon in points)
    for attempt in range(2):
        try:
            res = await client.get(
                OPENTOPODATA_URL, params={"locations": locations_param}, timeout=20.0
            )
            res.raise_for_status()
            results = res.json().get("results", [])
            return [r.get("elevation") for r in results]
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(3.0)
                continue
            print(f"  [error] elevation query failed after retry: {e}")
            return [None] * len(points)


async def fetch_district_terrain(client: httpx.AsyncClient, district: str, lat: float, lon: float):
    # Centroid + 4 points ~5km away (N/E/S/W) to estimate slope
    bearings = {"N": 0, "E": 90, "S": 180, "W": 270}
    sample_points = [(lat, lon)] + [
        offset_point(lat, lon, b, OFFSET_KM) for b in bearings.values()
    ]

    elevations = await query_elevations(client, sample_points)
    center_elev = elevations[0]
    neighbor_elevs = elevations[1:]

    if center_elev is None or any(e is None for e in neighbor_elevs):
        return {"district": district, "elevation_m": center_elev, "slope_deg": None}

    # Average |elevation difference| over the 5km offset, converted to a slope angle
    diffs = [abs(center_elev - e) for e in neighbor_elevs]
    avg_diff_m = sum(diffs) / len(diffs)
    slope_deg = math.degrees(math.atan(avg_diff_m / (OFFSET_KM * 1000)))

    return {"district": district, "elevation_m": center_elev, "slope_deg": round(slope_deg, 3)}


async def main():
    rows = []
    async with httpx.AsyncClient() as client:
        for district, coords in KERALA_DISTRICTS.items():
            print(f"Fetching terrain for {district}...")
            row = await fetch_district_terrain(client, district, coords["lat"], coords["lon"])
            rows.append(row)
            await asyncio.sleep(1.1)  # respect OpenTopoData's ~1 req/sec public rate limit

    df = pd.DataFrame(rows)
    df.to_csv("kerala_terrain_features.csv", index=False)

    print("\nSaved kerala_terrain_features.csv")
    print(df.to_string(index=False))
    missing = df["elevation_m"].isna().sum() + df["slope_deg"].isna().sum()
    if missing:
        print(f"\n[warn] {missing} missing values -- Open-Elevation may be rate-limited. "
              f"Re-run for just the missing districts, or wait a few minutes and retry.")


if __name__ == "__main__":
    asyncio.run(main())
