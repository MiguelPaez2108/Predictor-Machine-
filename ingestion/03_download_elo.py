"""
03_download_elo.py
Descarga Elo histórico de selecciones desde el dataset de Kaggle
(eloratings.net — 48 equipos clasificados WC2026, 1901-2026).

Fuente alternativa directa: eloratings.net por scraping.
Salida: data/raw/elo_historical.parquet
"""

import pandas as pd
import requests
import sys
from pathlib import Path
from io import StringIO

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ELO_HISTORICAL, ensure_dirs

# Dataset público Kaggle (descarga directa sin login vía URL raw)
# https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
KAGGLE_URL = (
    "https://raw.githubusercontent.com/openfootball/world-cup/master/"
    "2026/worldcup.json"  # fallback para fixtures
)

# Scraping directo de eloratings.net para los ratings actuales
ELORATINGS_BASE = "https://eloratings.net"

def scrape_current_elo():
    """Scraping de eloratings.net para top 200 selecciones."""
    from bs4 import BeautifulSoup

    print("Scraping eloratings.net...")
    headers = {"User-Agent": "Mozilla/5.0 (research project)"}

    try:
        r = requests.get(ELORATINGS_BASE, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        rows = []
        table = soup.find("table")
        if table:
            for tr in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(cols) >= 4:
                    rows.append({
                        "rank":   cols[0],
                        "team":   cols[1],
                        "rating": cols[2],
                        "change": cols[3] if len(cols) > 3 else None,
                    })

        df = pd.DataFrame(rows)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["rank"]   = pd.to_numeric(df["rank"],   errors="coerce")
        df["source"] = "eloratings.net_current"
        print(f"  Selecciones scrapeadas: {len(df)}")
        return df

    except Exception as e:
        print(f"  Scraping falló: {e}")
        return None


def load_kaggle_elo():
    """
    Carga el CSV del dataset de Kaggle con Elo histórico WC2026.
    El usuario debe descargar manualmente desde:
    https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
    y colocar el CSV en data/raw/elo_ratings_wc2026.csv
    """
    kaggle_csv = ELO_HISTORICAL.parent / "elo_ratings_wc2026.csv"

    if kaggle_csv.exists():
        print(f"Cargando Kaggle dataset desde {kaggle_csv}...")
        df = pd.read_csv(kaggle_csv)
        print(f"  Filas: {len(df):,}")
        print(f"  Columnas: {list(df.columns)}")
        return df
    else:
        print(f"⚠  Archivo no encontrado: {kaggle_csv}")
        print("   Descárgalo desde:")
        print("   https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings")
        print("   y colócalo en data/raw/elo_ratings_wc2026.csv")
        return None


def build_elo_from_results():
    """
    Calcula Elo dinámico desde cero usando los resultados históricos.
    Usa el dataset de martj42 ya descargado.
    Implementación de la fórmula estándar World Football Elo.
    """
    from config import INTL_RESULTS, ELO_K, ELO_HOME_ADVANTAGE

    if not INTL_RESULTS.exists():
        print("⚠  Ejecuta primero 01_download_martj42.py")
        return None

    print("Calculando Elo dinámico desde resultados históricos...")
    df = pd.read_parquet(INTL_RESULTS)
    df = df.sort_values("date").reset_index(drop=True)

    # Rating inicial para todas las selecciones
    elo = {}
    DEFAULT_ELO = 1500

    def get_k(tournament):
        t = str(tournament).lower()
        if "world cup" in t and "qualif" not in t:
            return ELO_K["world_cup"]
        elif "qualif" in t or "qualifier" in t:
            return ELO_K["qualifier"]
        elif "friendly" in t or "amistoso" in t:
            return ELO_K["friendly"]
        return ELO_K["default"]

    def expected_score(ra, rb, home_advantage=0):
        dr = ra - rb + home_advantage
        return 1 / (1 + 10 ** (-dr / 400))

    def goal_weight(gd):
        """Factor G por diferencia de goles (World Football Elo estándar)."""
        if gd <= 1:   return 1.0
        elif gd == 2: return 1.5
        elif gd == 3: return 1.75
        else:         return (11 + gd) / 8

    records = []
    for _, row in df.iterrows():
        hs, as_ = row["home_score"], row["away_score"]
        # Saltar partidos sin resultado (partidos futuros)
        if pd.isna(hs) or pd.isna(as_):
            continue

        h = row["home_team"]
        a = row["away_team"]
        rh = elo.get(h, DEFAULT_ELO)
        ra = elo.get(a, DEFAULT_ELO)

        home_adv = 0 if row.get("neutral", False) else ELO_HOME_ADVANTAGE
        eh = expected_score(rh, ra, home_adv)
        ea = 1 - eh

        if hs > as_:   wh, wa = 1.0, 0.0
        elif hs < as_: wh, wa = 0.0, 1.0
        else:          wh, wa = 0.5, 0.5

        K  = get_k(row["tournament"])
        G  = goal_weight(abs(int(hs) - int(as_)))

        new_rh = rh + K * G * (wh - eh)
        new_ra = ra + K * G * (wa - ea)

        records.append({
            "date":        row["date"],
            "home_team":   h,
            "away_team":   a,
            "elo_home_pre":  round(rh, 1),
            "elo_away_pre":  round(ra, 1),
            "elo_home_post": round(new_rh, 1),
            "elo_away_post": round(new_ra, 1),
            "expected_home": round(eh, 4),
            "tournament":  row["tournament"],
        })

        elo[h] = new_rh
        elo[a] = new_ra

    df_elo = pd.DataFrame(records)
    print(f"  Snapshots Elo calculados: {len(df_elo):,}")
    print(f"  Selecciones en el sistema: {len(elo)}")

    # Top 20 actuales
    top20 = sorted(elo.items(), key=lambda x: -x[1])[:20]
    print("\n  Top 20 Elo actual:")
    for i, (team, rating) in enumerate(top20, 1):
        print(f"    {i:2}. {team:<25} {rating:.0f}")

    return df_elo, elo


def download():
    ensure_dirs()

    # Intento 1: Kaggle dataset (más rico, con datos históricos completos)
    df_kaggle = load_kaggle_elo()

    # Intento 2: Calcular Elo dinámico desde resultados
    result = build_elo_from_results()

    if result is not None:
        df_elo, current_elo = result
        df_elo.to_parquet(ELO_HISTORICAL, index=False)
        print(f"\nGuardado: {ELO_HISTORICAL}")

        # Guardar snapshot actual también
        df_current = pd.DataFrame(
            [(t, r) for t, r in current_elo.items()],
            columns=["team", "elo_current"]
        ).sort_values("elo_current", ascending=False).reset_index(drop=True)
        df_current["rank"] = df_current.index + 1
        df_current.to_parquet(ELO_HISTORICAL.parent / "elo_current.parquet", index=False)
        print(f"Snapshot actual: {ELO_HISTORICAL.parent / 'elo_current.parquet'}")

    if df_kaggle is not None:
        kaggle_out = ELO_HISTORICAL.parent / "elo_kaggle_wc2026.parquet"
        df_kaggle.to_parquet(kaggle_out, index=False)
        print(f"Kaggle dataset: {kaggle_out}")


if __name__ == "__main__":
    download()
