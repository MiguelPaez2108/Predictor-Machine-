"""
01_download_martj42.py
Descarga los 49,000+ partidos internacionales desde el repo de martj42.
Fuente: https://github.com/martj42/international_results
Salida: data/raw/international_results.parquet
"""

import pandas as pd
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INTL_RESULTS, ensure_dirs

URL_RESULTS  = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
URL_SHOOTOUT = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
URL_GOALSC   = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"

def download():
    ensure_dirs()
    print("Descargando resultados históricos...")

    df = pd.read_csv(URL_RESULTS, parse_dates=["date"])

    # columnas: date, home_team, away_team, home_score, away_score,
    #           tournament, city, country, neutral
    print(f"  Partidos descargados: {len(df):,}")
    print(f"  Rango: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Torneos únicos: {df['tournament'].nunique()}")

    # Agregar resultado como columna explícita
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["home_score"] < df["away_score"], "result"] = "A"

    # Agregar diferencia de goles
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    # Flag de partido de alta competencia
    competitive = [
        "FIFA World Cup", "FIFA World Cup qualification",
        "UEFA Euro", "UEFA Euro qualification",
        "Copa América", "CONMEBOL",
        "Africa Cup of Nations", "AFC Asian Cup",
        "CONCACAF Gold Cup", "Nations League"
    ]
    df["is_competitive"] = df["tournament"].str.contains(
        "|".join(competitive), case=False, na=False
    )

    # Guardar como Parquet
    df.to_parquet(INTL_RESULTS, index=False)
    print(f"  Guardado en: {INTL_RESULTS}")
    print(f"  Partidos competitivos: {df['is_competitive'].sum():,}")

    # Descargar penaltis también
    print("\nDescargando historial de penaltis...")
    df_sh = pd.read_csv(URL_SHOOTOUT, parse_dates=["date"])
    out_sh = INTL_RESULTS.parent / "shootouts.parquet"
    df_sh.to_parquet(out_sh, index=False)
    print(f"  Penaltis: {len(df_sh):,} → {out_sh}")

    return df

if __name__ == "__main__":
    df = download()
    print("\n── Preview ──────────────────────────────────")
    print(df.tail(5).to_string())
    print(f"\nDistribución de resultados:")
    print(df["result"].value_counts())
