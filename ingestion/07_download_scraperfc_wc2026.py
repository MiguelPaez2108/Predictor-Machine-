"""
07_download_scraperfc_wc2026.py — Datos WC2026 en tiempo real via ScraperFC
=============================================================================
Cuando StatsBomb aún no tiene los datos del Mundial 2026 (se añaden
partido a partido tras disputarse), ScraperFC nos permite raspar FBref
para obtener estadísticas avanzadas actualizadas.

Extrae:
  - Estadísticas de resumen por equipo (goles, asistencias, xG, xGA)
  - Estadísticas de tiro (shots, shots on target, xG por partido)
  - Estadísticas defensivas (presiones, intercepciones, duelos)
  - Resultados de partidos en tiempo real

Salida:
  data/raw/wc2026_fbref_teams.parquet    ← stats por equipo del torneo
  data/raw/wc2026_fbref_matches.parquet  ← resultados partido a partido

Uso:
  python ingestion/07_download_scraperfc_wc2026.py
  python ingestion/07_download_scraperfc_wc2026.py --category shooting
  python ingestion/07_download_scraperfc_wc2026.py --all

NOTA: ScraperFC hace scraping real de FBref. No abusar de la velocidad
(añadir delays entre peticiones). FBref permite scraping moderado.
"""

import sys
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, ensure_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

WC2026_COMPETITION = "FIFA World Cup"
WC2026_SEASON      = "2025-2026"
SCRAPER_DELAY      = 3.0   # segundos entre peticiones (respetar FBref)

# Categorías disponibles en ScraperFC / FBref
FBREF_CATEGORIES = [
    "summary",    # goles, asistencias, xG, xA, shots
    "shooting",   # shots, shots on target, xG, distancia promedio
    "passing",    # pases completados, progresivos, xA
    "defense",    # intercepciones, tackles, presiones
    "possession", # touches, carries progresivos, dribbles
    "misc",       # tarjetas, penaltis, fouled
]


# ─────────────────────────────────────────────────────────────────────────────
# Descarga con ScraperFC
# ─────────────────────name────────────────────────────────────────────────────

def download_fbref_category(category: str = "summary") -> pd.DataFrame:
    """
    Descarga estadísticas de una categoría de FBref para el WC2026.
    Devuelve DataFrame con estadísticas de jugadores/equipos.
    """
    try:
        from scraperfc import FBref
    except ImportError:
        print("  ERROR: scraperfc no instalado. pip install scraperfc")
        return pd.DataFrame()

    print(f"  Descargando FBref [{category}] — {WC2026_COMPETITION} {WC2026_SEASON}...")

    try:
        fbref = FBref()
        df = fbref.scrape_player_season_stats(
            competition   = WC2026_COMPETITION,
            season        = WC2026_SEASON,
            stat_category = category
        )
        print(f"    ✓ {len(df):,} filas descargadas")
        time.sleep(SCRAPER_DELAY)
        return df

    except Exception as e:
        print(f"    ✗ Error en [{category}]: {e}")
        return pd.DataFrame()


def download_fbref_matches() -> pd.DataFrame:
    """
    Descarga resultados de partidos del WC2026 desde FBref.
    """
    try:
        from scraperfc import FBref
    except ImportError:
        print("  ERROR: scraperfc no instalado.")
        return pd.DataFrame()

    print(f"  Descargando partidos WC2026 desde FBref...")

    try:
        fbref   = FBref()
        matches = fbref.scrape_match_urls(
            competition = WC2026_COMPETITION,
            season      = WC2026_SEASON
        )
        print(f"    ✓ {len(matches)} partidos encontrados")
        time.sleep(SCRAPER_DELAY)

        # Descargar stats de cada partido
        all_match_data = []
        for i, url in enumerate(matches[:20]):   # Limitar a 20 por sesión
            try:
                match_df = fbref.scrape_match(url)
                if isinstance(match_df, pd.DataFrame) and not match_df.empty:
                    all_match_data.append(match_df)
                time.sleep(SCRAPER_DELAY)
                if (i + 1) % 5 == 0:
                    print(f"    Partido {i+1}/{min(20, len(matches))} procesado")
            except Exception as e:
                print(f"    ⚠ Error en partido {i+1}: {e}")
                continue

        if all_match_data:
            return pd.concat(all_match_data, ignore_index=True)

    except Exception as e:
        print(f"    ✗ Error general en partidos: {e}")

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Agregación por equipo (xG, xGA, forma) para usar como features
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_team_stats(summary_df: pd.DataFrame,
                         shooting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega estadísticas de jugadores a nivel de equipo para el WC2026.
    Produce un snapshot por selección usado como features del modelo.
    """
    if summary_df.empty:
        return pd.DataFrame()

    records = []
    team_col = None
    for col in ["squad", "team", "Squad", "Team"]:
        if col in summary_df.columns:
            team_col = col
            break

    if team_col is None:
        print("  ⚠ No se encontró columna de equipo en el DataFrame")
        return pd.DataFrame()

    for team_name, grp in summary_df.groupby(team_col):

        # xG del torneo (suma de todos los jugadores)
        xg_total  = 0.0
        xga_total = 0.0
        goals_for = 0
        goals_against_total = 0

        for col, val_storage in [
            (["xg", "xG", "expected_goals"], "xg"),
            (["npxg", "npxG"], "xg"),  # non-penalty xG
        ]:
            for c in col:
                if c in grp.columns:
                    xg_total = pd.to_numeric(grp[c], errors="coerce").sum()
                    break

        for col in ["goals", "Gls", "gls"]:
            if col in grp.columns:
                goals_for = int(pd.to_numeric(grp[col], errors="coerce").sum())
                break

        # xG de tiros (más fiable que summary)
        xg_shots = xg_total
        if not shooting_df.empty and team_col in shooting_df.columns:
            sh_grp = shooting_df[shooting_df[team_col] == team_name]
            for col in ["xg", "xG", "expected_goals"]:
                if col in sh_grp.columns:
                    xg_shots = pd.to_numeric(sh_grp[col], errors="coerce").sum()
                    break

        records.append({
            "team"          : str(team_name),
            "source"        : "fbref_wc2026",
            "timestamp"     : datetime.utcnow().isoformat(),
            "xg_total_wc"   : round(xg_total, 3),
            "xg_shots_wc"   : round(xg_shots, 3),
            "goals_for_wc"  : goals_for,
            "xg_conversion_wc": round(goals_for / xg_total, 3) if xg_total > 0 else np.nan,
            "n_players"     : len(grp),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: xG estimado desde partidos históricos disponibles
# ─────────────────────────────────────────────────────────────────────────────

def build_wc2026_manual_snapshot() -> pd.DataFrame:
    """
    Cuando FBref aún no tiene datos completos del WC2026,
    construye un snapshot manual con los valores más recientes
    disponibles en nuestro sistema (StatsBomb + martj42).

    Lee los parquets existentes y genera un resumen por selección
    con sus últimas 5-10 actuaciones antes del torneo.
    """
    ensure_dirs()
    from config import DATA_FEATURES, DATA_RAW
    from simulation.wc2026_fixtures import WC2026_GROUPS, normalize_team_name

    print("  Construyendo snapshot manual desde datos existentes...")

    # Cargar xG derivado de StatsBomb
    xg_path = DATA_FEATURES / "xg_derived.parquet"
    form_path = DATA_FEATURES / "form_rolling.parquet"

    xg_df   = pd.read_parquet(xg_path)   if xg_path.exists()   else pd.DataFrame()
    form_df = pd.read_parquet(form_path) if form_path.exists() else pd.DataFrame()

    # Todos los equipos del WC2026
    all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]

    records = []
    for team in all_teams:
        team_norm = normalize_team_name(team)

        # xG de los últimos 5 partidos disponibles en StatsBomb
        xg_recent = pd.DataFrame()
        if not xg_df.empty and "team" in xg_df.columns:
            mask = xg_df["team"].isin([team, team_norm])
            xg_recent = xg_df[mask].sort_values("date").tail(5)

        xg_mean  = float(xg_recent["xg"].mean())  if not xg_recent.empty else np.nan
        xga_mean = float(xg_recent["xga"].mean()) if not xg_recent.empty else np.nan

        # Forma reciente (desde rolling)
        form_recent = pd.DataFrame()
        if not form_df.empty and "team" in form_df.columns:
            mask = form_df["team"].isin([team, team_norm])
            form_recent = form_df[mask].sort_values("date").tail(1)

        form_weighted = float(form_recent["form_weighted"].iloc[0]) \
            if not form_recent.empty and "form_weighted" in form_recent.columns else np.nan

        records.append({
            "team"             : team,
            "group"            : next(
                (g for g, ts in WC2026_GROUPS.items() if team in ts), "?"
            ),
            "xg_avg_recent"    : round(xg_mean,  3) if pd.notna(xg_mean)  else None,
            "xga_avg_recent"   : round(xga_mean, 3) if pd.notna(xga_mean) else None,
            "form_weighted"    : round(form_weighted, 3) if pd.notna(form_weighted) else None,
            "n_matches_sb"     : len(xg_recent),
            "source"           : "statsbomb_historical",
            "timestamp"        : datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(records)
    out = DATA_RAW / "wc2026_team_snapshot.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Snapshot guardado: {out} ({len(df)} equipos)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(categories: list = None, download_matches: bool = False,
         manual_only: bool = False):
    ensure_dirs()
    print("═" * 60)
    print("SCRAPERFC — Datos WC2026 en tiempo real")
    print("═" * 60)

    if manual_only:
        build_wc2026_manual_snapshot()
        return

    # Intentar con ScraperFC primero
    all_stats = {}
    cats = categories or ["summary", "shooting"]

    for cat in cats:
        df = download_fbref_category(cat)
        if not df.empty:
            all_stats[cat] = df
            out = DATA_RAW / f"wc2026_fbref_{cat}.parquet"
            df.to_parquet(out, index=False)
            print(f"  Guardado: {out}")

    # Agregar a nivel de equipo
    if "summary" in all_stats or "shooting" in all_stats:
        team_stats = aggregate_team_stats(
            summary_df  = all_stats.get("summary",  pd.DataFrame()),
            shooting_df = all_stats.get("shooting", pd.DataFrame())
        )
        if not team_stats.empty:
            out = DATA_RAW / "wc2026_fbref_teams.parquet"
            team_stats.to_parquet(out, index=False)
            print(f"\n  ✓ Stats por equipo: {out} ({len(team_stats)} equipos)")

    # Partidos si se solicita
    if download_matches:
        matches = download_fbref_matches()
        if not matches.empty:
            out = DATA_RAW / "wc2026_fbref_matches.parquet"
            matches.to_parquet(out, index=False)
            print(f"  ✓ Partidos: {out} ({len(matches)} filas)")

    # Snapshot manual como fallback siempre
    print("\n  Construyendo snapshot manual como fallback...")
    build_wc2026_manual_snapshot()

    print("\n✓ Descarga WC2026 completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga datos del Mundial 2026 via ScraperFC/FBref"
    )
    parser.add_argument(
        "--category", nargs="+",
        choices=FBREF_CATEGORIES,
        default=["summary", "shooting"],
        help="Categorías FBref a descargar (default: summary shooting)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Descargar todas las categorías disponibles"
    )
    parser.add_argument(
        "--matches", action="store_true",
        help="Descargar también datos de partidos individuales"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Solo construir snapshot manual (sin scraping)"
    )
    args = parser.parse_args()

    cats = FBREF_CATEGORIES if args.all else args.category
    main(categories=cats, download_matches=args.matches, manual_only=args.manual)
