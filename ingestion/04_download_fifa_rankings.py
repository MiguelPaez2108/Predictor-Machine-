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
        print(f"  [AVISO]  No encontrado: {csv_path}")
        print("     Descarga desde: https://www.kaggle.com/datasets/cashncarry/fifaworldranking")
        return None


def create_manual_snapshot():
    """
    Snapshot manual de rankings FIFA actuales (junio 2026).
    Lee desde ranking.txt si existe y está en el escritorio/workspace.
    """
    import re
    from pathlib import Path
    
    # Intentar leer del ranking.txt
    txt_path = Path(r"c:\Users\Asus\OneDrive\Escritorio\wc_predictor\ranking.txt")
    if txt_path.exists():
        print(f"  Cargando rankings FIFA actuales desde {txt_path}...")
        content = txt_path.read_text(encoding="utf-8")
        
        translation_map = {
            "España": "Spain",
            "Países Bajos": "Netherlands",
            "Alemania": "Germany",
            "Bélgica": "Belgium",
            "México": "Mexico",
            "Estados Unidos": "United States",
            "Japón": "Japan",
            "Turquía": "Turkey",
            "Canadá": "Canada",
            "Costa de Marfil": "Ivory Coast",
            "Argelia": "Algeria",
            "Suecia": "Sweden",
            "Escocia": "Scotland",
            "Panamá": "Panama",
            "RD Congo": "DR Congo",
            "Chequia": "Czech Republic",
            "Túnez": "Tunisia",
            "Uzbekistán": "Uzbekistan",
            "Catar": "Qatar",
            "Arabia Saudí": "Saudi Arabia",
            "Arabia Saudita": "Saudi Arabia",
            "Sudáfrica": "South Africa",
            "Cabo Verde": "Cape Verde",
            "Bosnia y Herzegovina": "Bosnia and Herzegovina",
            "Jordania": "Jordan",
            "Nueva Zelanda": "New Zealand",
            "Curazao": "Curaçao",
            "Haití": "Haiti",
            "Francia": "France",
            "Inglaterra": "England",
            "Brasil": "Brazil",
            "Marruecos": "Morocco",
            "Croacia": "Croatia",
            "Suiza": "Switzerland",
            "Irán": "Iran",
            "Corea del Sur": "South Korea",
            "Egipto": "Egypt",
            "Austria": "Austria",
            "Colombia": "Colombia",
            "Uruguay": "Uruguay",
            "Australia": "Australia",
            "Ecuador": "Ecuador",
            "Senegal": "Senegal",
            "Irak": "Iraq",
            "Noruega": "Norway"
        }
        
        pattern = re.compile(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|")
        rows = []
        for line in content.split("\n"):
            match = pattern.search(line)
            if match:
                rank_str = match.group(1).strip()
                team_sp = match.group(2).strip()
                pts_str = match.group(3).strip()
                if team_sp == "Selección" or "---" in team_sp:
                    continue
                # Translate name
                team_en = translation_map.get(team_sp, team_sp)
                rows.append({
                    "team": team_en,
                    "fifa_rank": int(rank_str),
                    "fifa_points": float(pts_str),
                    "date": "2026-06-11",
                    "source": "ranking_txt_file"
                })
        if rows:
            print(f"  Rankings cargados y mapeados exitosamente: {len(rows)}")
            return pd.DataFrame(rows)
            
    # Fallback si no existe o falla parsing
    print("  Usando fallback manual estático...")
    rankings = [
        ("Argentina", 1, 1893.42),
        ("France", 2, 1851.41),
        ("Spain", 3, 1829.00),
        ("England", 4, 1807.00),
        ("Brazil", 5, 1780.15),
        ("Portugal", 6, 1764.57),
        ("Netherlands", 7, 1752.17),
        ("Belgium", 8, 1793.70),
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
        ("Ecuador", 20, 1618.76),
        ("Canada", 21, 1608.44),
        ("Australia", 23, 1594.22),
        ("South Korea", 24, 1588.99),
        ("Iran", 25, 1577.64),
        ("Turkey", 27, 1564.33),
        ("Saudi Arabia", 29, 1551.44),
        ("Egypt", 30, 1544.87),
        ("Algeria", 32, 1531.64),
        ("Iraq", 47, 1432.15),
        ("DR Congo", 48, 1425.58),
        ("New Zealand", 39, 1484.68),
        ("Scotland", 46, 1438.71),
        ("Panama", 41, 1471.55),
        ("Czech Republic", 36, 1504.38),
        ("Norway", 31, 1530.00),
        ("Ivory Coast", 33, 1540.00),
        ("Sweden", 38, 1509.00),
        ("Paraguay", 41, 1505.00),
        ("Tunisia", 45, 1476.00),
        ("Uzbekistan", 50, 1458.00),
        ("Qatar", 56, 1450.00),
        ("South Africa", 60, 1428.00),
        ("Jordan", 63, 1387.00),
        ("Bosnia and Herzegovina", 64, 1387.00),
        ("Cape Verde", 67, 1371.00),
        ("Ghana", 73, 1346.00),
        ("Curacao", 82, 1294.00),
        ("Haiti", 83, 1293.00),
        ("Austria", 44, 1451.85)
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
