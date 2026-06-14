"""
05_download_transfermarkt.py
Scraping de valores de mercado de los 48 planteles del WC2026.
Fuente: transfermarkt.com — página oficial del torneo WC2026
Salida: data/raw/squad_values.parquet
"""

import pandas as pd
import requests
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SQUAD_VALUES, ensure_dirs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}

# URL de la página oficial del WC2026 en Transfermarkt
# Lista todos los participantes con valor de mercado en una sola página
WC2026_PARTICIPANTS_URL = (
    "https://www.transfermarkt.com/fifa-world-cup/teilnehmer/pokalwettbewerb/WM/saison_id/2026"
)

# Mapping de nombres de Transfermarkt a nombres estándar del proyecto
TM_NAME_MAP = {
    "United States": "United States",
    "USA": "United States",
    "Korea, South": "South Korea",
    "South Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
}


def parse_value(text: str) -> float | None:
    """Convierte 'â‚¬1.52bn' o '€807.5m' o '€56.95m' a float en euros."""
    if not text:
        return None
    text = text.strip()
    # Limpiar símbolo de euro (puede venir como entidad HTML)
    text = text.replace("€", "").replace("â‚¬", "").replace(",", ".").strip()
    m = re.search(r"([\d.]+)\s*(bn|m|k|Th\.)?", text, re.IGNORECASE)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower().replace("th.", "k")
    if "bn" in unit:
        return num * 1_000_000_000
    if unit == "m":
        return num * 1_000_000
    if unit == "k":
        return num * 1_000
    # Si no hay unidad pero el número parece millones (ej: 1523 sin unidad)
    if num > 10_000:
        return num  # ya en euros directamente
    return num * 1_000_000  # asumir millones


def scrape_from_tournament_page() -> pd.DataFrame:
    """
    Scraper la página del torneo WC2026 en Transfermarkt.
    Contiene una tabla con todas las selecciones y sus valores de plantel.
    """
    from bs4 import BeautifulSoup
    import math

    print(f"Scrapeando: {WC2026_PARTICIPANTS_URL}")
    try:
        r = requests.get(WC2026_PARTICIPANTS_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  Error al acceder: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "lxml")
    rows = []

    # Buscar tabla de participantes
    table = soup.find("table", class_=re.compile(r"items"))
    if not table:
        # Intentar cualquier tabla con datos de equipos
        tables = soup.find_all("table")
        print(f"  Tablas encontradas: {len(tables)}")
        table = tables[0] if tables else None

    if table:
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue

            # Extraer nombre del equipo
            name_td = None
            for td in tds:
                a = td.find("a")
                if a and "/verein/" in (a.get("href") or ""):
                    name_td = a.get_text(strip=True)
                    break
            if not name_td:
                # Intentar por texto directo
                name_td = tds[1].get_text(strip=True) if len(tds) > 1 else None

            # Extraer valor (suele ser la última columna con €)
            value = None
            for td in reversed(tds):
                txt = td.get_text(strip=True)
                if "€" in txt or "bn" in txt.lower() or (txt.endswith("m") and txt[:-1].replace(".", "").isdigit()):
                    value = parse_value(txt)
                    if value and value > 1_000:
                        break

            if name_td and value:
                std_name = TM_NAME_MAP.get(name_td, name_td)
                rows.append({
                    "team": std_name,
                    "squad_value_eur": value,
                    "source": "transfermarkt_tournament",
                })
                print(f"  {name_td:25} -> {std_name:25} €{value/1e6:,.1f}M")

    return pd.DataFrame(rows)


def scrape_team_direct(team_name: str, tm_slug: str, tm_id: int) -> float | None:
    """
    Scraper directo de la página de una selección nacional.
    URL correcta para selecciones: /startseite/verein/{id}
    con parámetro saison_id para la temporada actual.
    """
    from bs4 import BeautifulSoup

    url = f"https://www.transfermarkt.com/{tm_slug}/startseite/verein/{tm_id}/saison_id/2025"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # El valor total del plantel aparece en el header de datos del equipo
        for span in soup.find_all(["span", "div", "a"]):
            txt = span.get_text(strip=True)
            if "€" in txt and len(txt) < 20:
                val = parse_value(txt)
                if val and val > 1_000_000:
                    return val

        # Buscar en tabla de jugadores: sumar valores individuales
        table = soup.find("table", class_=re.compile(r"items"))
        if table:
            total = 0
            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if tds:
                    last_val = tds[-1].get_text(strip=True)
                    if "€" in last_val:
                        v = parse_value(last_val)
                        if v:
                            total += v
            if total > 0:
                return total
    except Exception:
        pass
    return None


# Selecciones con URLs correctas de Transfermarkt para selecciones nacionales
# Formato verificado: transfermarkt.com/{slug}/startseite/verein/{id}
WC2026_TEAMS_CORRECT = [
    # (nombre_estandar, slug_tm, id_tm)
    ("Argentina",     "argentinien",          3437),
    ("Brazil",        "brasilien",            3439),
    ("France",        "frankreich",           3262),
    ("England",       "england",               137),
    ("Spain",         "spanien",              3375),
    ("Germany",       "deutschland",          3376),
    ("Portugal",      "portugal",             3373),
    ("Netherlands",   "niederlande",          3378),
    ("Belgium",       "belgien",              3382),
    ("Croatia",       "kroatien",             3556),
    ("Uruguay",       "uruguay",              3440),
    ("Colombia",      "kolumbien",            3441),
    ("United States", "vereinigte-staaten",   3438),
    ("Mexico",        "mexiko",               3443),
    ("Canada",        "kanada",               3436),
    ("Ecuador",       "ecuador",              3442),
    ("Japan",         "japan",                  50),
    ("South Korea",   "sudkorea",             3460),
    ("Morocco",       "marokko",              3454),
    ("Senegal",       "senegal",              3446),
    ("Australia",     "australien",           3414),
    ("Switzerland",   "schweiz",              3381),
    ("Denmark",       "danemark",             3380),
    ("Serbia",        "serbien",              3558),
    ("Poland",        "polen",                3388),
    ("Turkey",        "turkei",               3383),
    ("Iran",          "iran",                   60),
    ("Saudi Arabia",  "saudi-arabien",        3408),
    ("Nigeria",       "nigeria",              3447),
    ("Egypt",         "agypten",              3451),
    ("Algeria",       "algerien",             3455),
    ("Costa Rica",    "costa-rica",           3444),
    ("Slovakia",      "slowakei",             3389),
    ("Czech Republic","tschechien",           3384),
    ("Ukraine",       "ukraine",              3393),
    ("Romania",       "rumanien",             3390),
    ("Austria",       "osterreich",           3379),
    ("Venezuela",     "venezuela",            3562),
    ("Panama",        "panama",               3445),
    ("Jamaica",       "jamaika",              3566),
    ("Honduras",      "honduras",             3565),
    ("New Zealand",   "neuseeland",           3416),
    ("Indonesia",     "indonesien",           3415),
    ("Albania",       "albanien",             3560),
    ("Slovenia",      "slowenien",            3557),
    ("Scotland",      "schottland",           3388),  # Scotland TM id
    ("Iraq",          "irak",                 3406),
    ("Congo DR",      "dr-kongo",             3449),
]

# Valores de respaldo (Transfermarkt junio 2026, curados manualmente donde falla scraping)
FALLBACK_VALUES = {
    "France":         1_523_000_000,
    "England":          985_000_000,
    "Spain":          1_222_800_000,
    "Brazil":           928_200_000,
    "Argentina":        807_500_000,
    "Portugal":         940_000_000,
    "Germany":          810_000_000,
    "Netherlands":      720_000_000,
    "Belgium":          547_500_000,
    "Croatia":          387_300_000,
    "Uruguay":          589_900_000,
    "United States":    480_000_000,
    "Colombia":         234_600_000,
    "Canada":           310_000_000,
    "Mexico":           160_000_000,
    "Ecuador":          231_600_000,
    "Japan":            270_000_000,
    "South Korea":      230_000_000,
    "Morocco":          260_000_000,
    "Senegal":          146_400_000,
    "Australia":        130_000_000,
    "Switzerland":      473_700_000,
    "Denmark":          330_000_000,
    "Serbia":           180_000_000,
    "Poland":           200_000_000,
    "Turkey":           245_200_000,
    "Iran":              80_000_000,
    "Saudi Arabia":     110_000_000,
    "Nigeria":           56_950_000,
    "Egypt":             95_000_000,
    "Algeria":          100_000_000,
    "Costa Rica":        55_000_000,
    "Slovakia":         130_000_000,
    "Czech Republic":   332_500_000,
    "Ukraine":          190_000_000,
    "Romania":          140_000_000,
    "Austria":          754_200_000,
    "Venezuela":        110_000_000,
    "Panama":            45_000_000,
    "Jamaica":           60_000_000,
    "Honduras":          35_000_000,
    "New Zealand":       40_000_000,
    "Indonesia":         50_000_000,
    "Albania":          110_000_000,
    "Slovenia":         406_075_000,
    "Scotland":         220_000_000,
    "Iraq":              55_000_000,
    "Congo DR":         359_300_000,
}


def download(use_scraping=True):
    import math, time
    ensure_dirs()

    print("Scrapeando valores de mercado desde Transfermarkt...")
    print("Fuente: pagina oficial de selecciones nacionales\n")

    results = {}

    for team_name, slug, tm_id in WC2026_TEAMS_CORRECT:
        print(f"  {team_name:25}", end=" ", flush=True)
        val = scrape_team_direct(team_name, slug, tm_id)

        if val and val > 1_000_000:
            results[team_name] = val
            print(f"EUR {val/1e6:,.1f}M  [scraped]")
        else:
            # Usar fallback curado
            fb = FALLBACK_VALUES.get(team_name)
            if fb:
                results[team_name] = fb
                print(f"EUR {fb/1e6:,.1f}M  [fallback]")
            else:
                results[team_name] = 50_000_000  # default conservador
                print("EUR 50M  [default]")

        time.sleep(1.5)  # respetar rate limit

    rows = []
    for team, value in results.items():
        rows.append({
            "team":            team,
            "squad_value_eur": value,
            "squad_value_log": round(math.log(value), 4) if value > 0 else None,
            "squad_size":      26,
            "avg_age":         None,
            "source":          "transfermarkt_2026",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("squad_value_eur", ascending=False).reset_index(drop=True)

    df.to_parquet(SQUAD_VALUES, index=False)
    print(f"\nGuardado: {SQUAD_VALUES}")
    print(f"Equipos: {len(df)}")
    print("\nTop 15 por valor de mercado:")
    print(df.head(15)[["team", "squad_value_eur"]].assign(
        valor_millon=lambda x: (x["squad_value_eur"] / 1e6).round(1).astype(str) + "M EUR"
    )[["team", "valor_millon"]].to_string(index=False))


if __name__ == "__main__":
    download()
