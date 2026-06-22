"""
06_download_weather.py
Descarga datos históricos y pronóstico de clima para las sedes del WC2026.
Fuente: Open-Meteo API (gratuita, sin API key)
Sedes: 16 ciudades en USA, México y Canadá
Salida: data/raw/weather_venues.parquet
"""

import pandas as pd
import requests
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, ensure_dirs

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Las 16 sedes del Mundial 2026
WC2026_VENUES = [
    # USA
    {"city": "New York",         "stadium": "MetLife Stadium",          "lat": 40.8135,  "lon": -74.0745, "alt_m": 8,    "country": "USA"},
    {"city": "Los Angeles",      "stadium": "SoFi Stadium",             "lat": 33.9535,  "lon": -118.3392,"alt_m": 88,   "country": "USA"},
    {"city": "Dallas",           "stadium": "AT&T Stadium",             "lat": 32.7480,  "lon": -97.0928, "alt_m": 186,  "country": "USA"},
    {"city": "San Francisco",    "stadium": "Levi's Stadium",           "lat": 37.4033,  "lon": -121.9694,"alt_m": 10,   "country": "USA"},
    {"city": "Miami",            "stadium": "Hard Rock Stadium",        "lat": 25.9580,  "lon": -80.2389, "alt_m": 2,    "country": "USA"},
    {"city": "Seattle",          "stadium": "Lumen Field",              "lat": 47.5952,  "lon": -122.3316,"alt_m": 18,   "country": "USA"},
    {"city": "Boston",           "stadium": "Gillette Stadium",         "lat": 42.0909,  "lon": -71.2643, "alt_m": 30,   "country": "USA"},
    {"city": "Philadelphia",     "stadium": "Lincoln Financial Field",  "lat": 39.9008,  "lon": -75.1675, "alt_m": 10,   "country": "USA"},
    {"city": "Kansas City",      "stadium": "Arrowhead Stadium",        "lat": 39.0489,  "lon": -94.4839, "alt_m": 311,  "country": "USA"},
    {"city": "Atlanta",          "stadium": "Mercedes-Benz Stadium",    "lat": 33.7554,  "lon": -84.4009, "alt_m": 306,  "country": "USA"},
    {"city": "Houston",          "stadium": "NRG Stadium",              "lat": 29.6847,  "lon": -95.4107, "alt_m": 16,   "country": "USA"},
    # México
    {"city": "Mexico City",      "stadium": "Estadio Azteca",           "lat": 19.3029,  "lon": -99.1505, "alt_m": 2240, "country": "MEX"},
    {"city": "Guadalajara",      "stadium": "Estadio Akron",            "lat": 20.6688,  "lon": -103.4732,"alt_m": 1650, "country": "MEX"},
    {"city": "Monterrey",        "stadium": "Estadio BBVA",             "lat": 25.6694,  "lon": -100.2431,"alt_m": 537,  "country": "MEX"},
    # Canadá
    {"city": "Toronto",          "stadium": "BMO Field",                "lat": 43.6333,  "lon": -79.4187, "alt_m": 76,   "country": "CAN"},
    {"city": "Vancouver",        "stadium": "BC Place",                 "lat": 49.2768,  "lon": -123.1118,"alt_m": 5,    "country": "CAN"},
]


def fetch_historical_weather(lat, lon, city, start="2026-06-11", end="2026-07-19"):
    """
    Obtiene temperatura y viento para las fechas del torneo.
    - Pasado: archive API (datos reales)
    - Futuro <= 16 dias: forecast API
    - Futuro > 16 dias: archive 2025 como proxy climatico
    """
    from datetime import datetime, timedelta

    today = datetime.now().date()
    forecast_limit = today + timedelta(days=16)

    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end,   "%Y-%m-%d").date()

    segments = []  # (range_start, range_end, url, year_offset)

    # Segmento 1: fechas pasadas -> archivo real
    if start_d < today:
        seg_end = min(end_d, today - timedelta(days=1))
        segments.append((start_d, seg_end, OPEN_METEO_URL, 0))

    # Segmento 2: fechas futuras dentro de 16 dias -> forecast
    if start_d <= forecast_limit and end_d > today:
        seg_start = max(start_d, today)
        seg_end   = min(end_d, forecast_limit)
        segments.append((seg_start, seg_end, OPEN_METEO_FORECAST, 0))

    # Segmento 3: fechas futuras mas alla de 16 dias -> proxy con datos 2025
    if end_d > forecast_limit:
        seg_start = max(start_d, forecast_limit + timedelta(days=1))
        # Convertir fechas a 2025 para el archivo
        proxy_start = seg_start.replace(year=2025)
        proxy_end   = end_d.replace(year=2025)
        segments.append((proxy_start, proxy_end, OPEN_METEO_URL, 1))  # offset=1 -> volver a 2026

    all_rows = []
    for seg_start, seg_end, url, year_offset in segments:
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": seg_start.strftime("%Y-%m-%d"),
            "end_date":   seg_end.strftime("%Y-%m-%d"),
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "windspeed_10m_max",
            ]),
            "timezone": "auto",
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data  = r.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            for i, date in enumerate(dates):
                # Si es proxy de 2025, ajustar la fecha a 2026
                if year_offset:
                    d = datetime.strptime(date, "%Y-%m-%d").replace(year=2026)
                    date = d.strftime("%Y-%m-%d")
                all_rows.append({
                    "city":      city,
                    "date":      date,
                    "temp_max":  daily.get("temperature_2m_max",  [None]*len(dates))[i],
                    "temp_min":  daily.get("temperature_2m_min",  [None]*len(dates))[i],
                    "temp_mean": daily.get("temperature_2m_mean", [None]*len(dates))[i],
                    "precip_mm": daily.get("precipitation_sum",   [None]*len(dates))[i],
                    "humidity":  None,
                    "wind_kmh":  daily.get("windspeed_10m_max",   [None]*len(dates))[i],
                    "is_proxy":  bool(year_offset),
                })
        except Exception as e:
            print(f"    Aviso {city} ({seg_start}->{seg_end}): {e}")

    return all_rows


def download():
    ensure_dirs()
    print("Descargando datos de clima para las 16 sedes del WC2026...")
    print("Fuente: Open-Meteo API (gratuita)\n")

    all_rows    = []
    venues_meta = []

    for venue in WC2026_VENUES:
        city = venue["city"]
        print(f"  {city} ({venue['country']}) alt={venue['alt_m']}m...", end=" ", flush=True)

        weather_rows = fetch_historical_weather(
            venue["lat"], venue["lon"], city,
            start="2026-06-11",
            end="2026-07-19"
        )

        for row in weather_rows:
            row.update({
                "stadium":  venue["stadium"],
                "country":  venue["country"],
                "lat":      venue["lat"],
                "lon":      venue["lon"],
                "alt_m":    venue["alt_m"],
            })
        all_rows.extend(weather_rows)

        # Metadata de la sede (para features de altitud)
        venues_meta.append({
            "city":         venue["city"],
            "stadium":      venue["stadium"],
            "country":      venue["country"],
            "lat":          venue["lat"],
            "lon":          venue["lon"],
            "alt_m":        venue["alt_m"],
            "alt_flag":     venue["alt_m"] > 1500,  # flag de altitud significativa
        })

        if weather_rows:
            avg_t = sum(r["temp_mean"] for r in weather_rows if r["temp_mean"]) / max(len(weather_rows), 1)
            print(f"[OK] {len(weather_rows)} días, temp_media={avg_t:.1f}°C")
        else:
            print("[AVISO] sin datos")

        time.sleep(0.5)  # respetar rate limit de Open-Meteo

    # Guardar datos de clima
    df_weather = pd.DataFrame(all_rows)
    out_weather = DATA_RAW / "weather_venues.parquet"
    df_weather.to_parquet(out_weather, index=False)
    print(f"\nClima guardado: {out_weather} ({len(df_weather)} filas)")

    # Guardar metadata de sedes
    df_meta = pd.DataFrame(venues_meta)
    out_meta = DATA_RAW / "venues_metadata.parquet"
    df_meta.to_parquet(out_meta, index=False)
    print(f"Metadata sedes: {out_meta}")

    # Resumen de sedes con altitud alta (relevante para el modelo)
    print("\n[AVISO]  Sedes con altitud > 1500m (ajuste necesario en el modelo):")
    high_alt = [v for v in venues_meta if v["alt_flag"]]
    for v in high_alt:
        print(f"   {v['city']:15} {v['alt_m']}m — impacto estimado en xG: +{v['alt_m']/1000*8:.1f}%")


if __name__ == "__main__":
    download()
