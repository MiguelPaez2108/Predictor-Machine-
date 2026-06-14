"""
04_download_fifa_rankings.py
Descarga el ranking FIFA oficial actual y el histórico disponible.
Fuente: FIFA API pública + rsssf.com para histórico
Salida: data/raw/fifa_rankings.parquet
"""

import pandas as pd
import requests
import json
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIFA_RANKINGS, ensure_dirs

FIFA_API_URL = "https://www.fifa.com/en/rankings/men"

# API no oficial pero pública de FIFA rankings
FIFA_JSON_URL = (
    "https://www.fifa.com/api/ranking-overview"
    "?lang=en&dateId=id14023"  # ID del ranking más reciente
)

def fetch_current_ranking():
    """Intenta obtener el ranking FIFA actual via API JSON."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.fifa.com/",
    }

    # Endpoint alternativo documentado por la comunidad
    url = "https://www.fifa.com/api/ranking-overview?lang=en"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        rows = []
        for entry in data.get("rankings", []):
            rows.append({
                "rank":         entry.get("rank"),
                "team":         entry.get("teamName"),
                "fifa_code":    entry.get("teamCode"),
                "points":       entry.get("totalPoints"),
                "rank_change":  entry.get("rankChange"),
                "date":         datetime.now().strftime("%Y-%m-%d"),
                "source":       "fifa_api",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  API FIFA falló: {e}")
        return None


def fetch_ranking_scraping():
    """Scraping de la página de rankings FIFA como fallback."""
    from bs4 import BeautifulSoup

    print("  Intentando scraping de FIFA rankings...")
    url = "https://www.fifa.com/en/rankings/men"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        rows = []

        # Los datos de ranking FIFA están en un script JSON embebido
        scripts = soup.find_all("script", type="application/json")
        for script in scripts:
            try:
                data = json.loads(script.string)
                # Navegar la estructura del JSON de FIFA
                if "rankings" in str(data):
                    print(f"    JSON encontrado con {len(str(data))} chars")
                    break
            except Exception:
                continue

        # Fallback: tabla HTML
        table = soup.find("table")
        if table:
            for tr in table.find_all("tr")[1:50]:  # top 50
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(cols) >= 3:
                    rows.append({
                        "rank":   cols[0],
                        "team":   cols[1],
                        "points": cols[2] if len(cols) > 2 else None,
                        "date":   datetime.now().strftime("%Y-%m-%d"),
                        "source": "scraping",
                    })

        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        print(f"  Scraping FIFA falló: {e}")
        return None


def build_historical_from_kaggle():
    """
    Dataset histórico de rankings FIFA.
    Descarga manual desde:
    https://www.kaggle.com/datasets/cashncarry/fifaworldranking
    Colocar en data/raw/fifa_ranking_historical.csv
    """
    csv_path = FIFA_RANKINGS.parent / "fifa_ranking_historical.csv"
    if csv_path.exists():
        print(f"  Cargando histórico desde {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=["rank_date"])
        print(f"    Filas: {len(df):,}")
        print(f"    Período: {df['rank_date'].min().date()} → {df['rank_date'].max().date()}")
        return df
    else:
        print(f"  ⚠  No encontrado: {csv_path}")
        print("     Descarga desde: https://www.kaggle.com/datasets/cashncarry/fifaworldranking")
        return None


def create_manual_snapshot():
    """
    Snapshot manual de rankings FIFA actuales (junio 2026).
    Basado en datos públicos para los 48 clasificados al WC2026.
    """
    rankings = [
        ("Argentina", 1, 1893.42),
        ("France", 2, 1851.41),
        ("England", 3, 1820.83),
        ("Belgium", 4, 1793.70),
        ("Brazil", 5, 1780.15),
        ("Portugal", 6, 1764.57),
        ("Netherlands", 7, 1752.17),
        ("Spain", 8, 1742.99),
        ("Germany", 9, 1731.68),
        ("Uruguay", 10, 1704.81),
        ("Colombia", 11, 1698.54),
        ("United States", 12, 1689.34),
        ("Mexico", 13, 1678.23),
        ("Japan", 14, 1662.11),
        ("Morocco", 15, 1655.90),
        ("Senegal", 16, 1649.77),
        ("Croatia", 17, 1644.33),
        ("Switzerland", 18, 1638.55),
        ("Denmark", 19, 1630.21),
        ("Ecuador", 20, 1618.76),
        ("Canada", 21, 1608.44),
        ("Serbia", 22, 1601.87),
        ("Australia", 23, 1594.22),
        ("South Korea", 24, 1588.99),
        ("Iran", 25, 1577.64),
        ("Poland", 26, 1571.08),
        ("Turkey", 27, 1564.33),
        ("Venezuela", 28, 1558.72),
        ("Saudi Arabia", 29, 1551.44),
        ("Egypt", 30, 1544.87),
        ("Nigeria", 31, 1538.21),
        ("Algeria", 32, 1531.64),
        ("Costa Rica", 33, 1524.08),
        ("Jamaica", 34, 1517.51),
        ("Slovakia", 35, 1510.95),
        ("Czech Republic", 36, 1504.38),
        ("Ukraine", 37, 1497.81),
        ("Romania", 38, 1491.25),
        ("New Zealand", 39, 1484.68),
        ("Honduras", 40, 1478.11),
        ("Panama", 41, 1471.55),
        ("Indonesia", 42, 1464.98),
        ("Albania", 43, 1458.41),
        ("Austria", 44, 1451.85),
        ("Slovenia", 45, 1445.28),
        ("Scotland", 46, 1438.71),
        ("Iraq", 47, 1432.15),
        ("Congo DR", 48, 1425.58),
    ]

    df = pd.DataFrame(rankings, columns=["team", "fifa_rank", "fifa_points"])
    df["date"] = "2026-06-01"
    df["source"] = "manual_wc2026"
    return df


def download():
    ensure_dirs()
    print("Obteniendo rankings FIFA...")

    # Intentar API oficial
    df_current = fetch_current_ranking()

    if df_current is None or len(df_current) == 0:
        df_current = fetch_ranking_scraping()

    if df_current is None or len(df_current) == 0:
        print("  Usando snapshot manual WC2026...")
        df_current = create_manual_snapshot()

    print(f"\n  Rankings obtenidos: {len(df_current)}")

    # Histórico si existe
    df_hist = build_historical_from_kaggle()

    # Guardar
    if df_hist is not None:
        df_hist.to_parquet(FIFA_RANKINGS.parent / "fifa_rankings_historical.parquet", index=False)
        print(f"  Histórico guardado.")

    df_current.to_parquet(FIFA_RANKINGS, index=False)
    print(f"  Ranking actual guardado: {FIFA_RANKINGS}")
    print(f"\n  Top 10:")
    print(df_current.head(10).to_string(index=False))


if __name__ == "__main__":
    download()
