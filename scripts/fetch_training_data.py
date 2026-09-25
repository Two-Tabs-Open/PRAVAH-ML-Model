import asyncio
import httpx
import pandas as pd
from datetime import date

# Config

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

# Open-Meteo's flood (river discharge) archive is the shorter-coverage
# endpoint of the two - if it errors for dates before ~2015, narrow this.
START_DATE = "2015-01-01"
END_DATE = str(date.today())

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_ARCHIVE_URL = "https://flood-api.open-meteo.com/v1/flood"

# IMD daily rainfall categories (mm/day) - standard, publicly documented
def imd_category(rainfall_mm: float) -> str:
    if rainfall_mm < 2.5:
        return "No Rain"
    elif rainfall_mm < 15.6:
        return "Light"
    elif rainfall_mm < 64.5:
        return "Moderate"
    elif rainfall_mm < 115.6:
        return "Heavy"
    elif rainfall_mm < 204.5:
        return "Very Heavy"
    else:
        return "Extreme"


async def fetch_district_history(client: httpx.AsyncClient, district: str, lat: float, lon: float):
    weather_res = await client.get(
        WEATHER_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": "precipitation_sum",
            "timezone": "auto",
        },
    )
    flood_res = await client.get(
        FLOOD_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": "river_discharge",
        },
    )

    weather_res.raise_for_status()
    daily_weather = weather_res.json().get("daily", {})
    dates = daily_weather.get("time", [])
    precip = daily_weather.get("precipitation_sum", [])

    # Flood archive has shorter historical coverage than weather archive in
    # some regions - handle a failure gracefully rather than losing the
    # whole district's rainfall data.
    if flood_res.status_code == 200:
        daily_flood = flood_res.json().get("daily", {})
        discharge = daily_flood.get("river_discharge", [])
    else:
        print(f"  [warn] flood archive failed for {district} ({flood_res.status_code}) - discharge will be NaN")
        discharge = [None] * len(dates)

    # Pad/truncate discharge to match weather length if they differ
    if len(discharge) < len(dates):
        discharge = discharge + [None] * (len(dates) - len(discharge))
    elif len(discharge) > len(dates):
        discharge = discharge[: len(dates)]

    df = pd.DataFrame(
        {
            "district": district,
            "date": dates,
            "rainfall_mm": [float(x) if x is not None else 0.0 for x in precip],
            "river_discharge": [float(x) if x is not None else None for x in discharge],
        }
    )
    return df


async def main():
    all_frames = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for district, coords in KERALA_DISTRICTS.items():
            print(f"Fetching {district}...")
            try:
                df = await fetch_district_history(client, district, coords["lat"], coords["lon"])
                all_frames.append(df)
            except Exception as e:
                print(f"  [error] {district} failed entirely: {e}")
            await asyncio.sleep(0.3)  # stay polite to the free API

    if not all_frames:
        print("No data fetched - check your internet connection and try again.")
        return

    data = pd.concat(all_frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["district", "date"]).reset_index(drop=True)

    # Rolling-window features, matching the live pipeline's feature set
    for col, prefix in [("rainfall_mm", "rainfall_mm"), ("river_discharge", "river_discharge")]:
        data[f"{prefix}_3d_sum"] = data.groupby("district")[col].transform(
            lambda s: s.rolling(3, min_periods=1).sum()
        )
        data[f"{prefix}_7d_sum"] = data.groupby("district")[col].transform(
            lambda s: s.rolling(7, min_periods=1).sum()
        )
        data[f"{prefix}_15d_sum"] = data.groupby("district")[col].transform(
            lambda s: s.rolling(15, min_periods=1).sum()
        )

    # Label: IMD category from same-day rainfall
    data["risk_category"] = data["rainfall_mm"].apply(imd_category)

    data.to_csv("kerala_training_data.csv", index=False)
    print(f"\nSaved kerala_training_data.csv - {len(data):,} rows across {data['district'].nunique()} districts")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print("\nLabel distribution:")
    print(data["risk_category"].value_counts())
    print(f"\nMissing river_discharge rows: {data['river_discharge'].isna().sum():,} "
          f"({data['river_discharge'].isna().mean()*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
