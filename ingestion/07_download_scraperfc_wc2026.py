"""
07_download_scraperfc_wc2026.py — Datos WC2026 en tiempo real via ScraperFC 4.x
================================================================================
Compatible con ScraperFC >= 4.0 (API totalmente renovada en v4.0).

CAMBIOS PRINCIPALES vs versión anterior:
  - Import: from ScraperFC import FBref  (S mayúscula, sin .scraperfc)
  - scrape_player_season_stats() → scrape_stats() con stat_type=
  - scrape_match_urls() → ya NO existe → se itera desde scrape_matches()
  - scrape_all_stats() disponible como atajo para todas las categorías

ESTRATEGIA para el WC2026:
  1. Intentar FBref via ScraperFC 4.x (stats de jugadores del torneo)
  2. Fallback: snapshot manual desde nuestros parquets de StatsBomb + form

El Mundial está en curso → FBref va poblando datos partido a partido.
Ejecutar este script cada 1-2 días para mantener el snapshot actualizado.

Uso:
  python ingestion/07_download_scraperfc_wc2026.py            # auto (fbref + fallback)
  python ingestion/07_download_scraperfc_wc2026.py --manual   # solo fallback, sin scraping
  python ingestion/07_download_scraperfc_wc2026.py --category shooting
  python ingestion/07_download_scraperfc_wc2026.py --list-leagues  # ver nombres válidos
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

# Nombre exacto del Mundial en comps.yaml de ScraperFC 4.x
# Si falla, ejecutar con --list-leagues para ver los nombres válidos de FBref
WC2026_LEAGUE = "FIFA World Cup"
WC2026_YEAR   = "2026"
SCRAPER_DELAY = 4.0   # segundos entre llamadas (FBref rate-limit es estricto)

# Categorías válidas en ScraperFC 4.x FBref
# scrape_stats(year, league, stat_type=...)
STAT_TYPES = [
    "standard",   # goals, assists, xG, xA → lo más útil para nuestro modelo
    "shooting",   # shots, shots_on_target, xG por tiro
    "passing",    # pases, pases progresivos, xA
    "defense",    # intercepciones, tackles, presiones
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: importar ScraperFC con mensaje claro si falta
# ─────────────────────────────────────────────────────────────────────────────

def _import_scraperfc():
    """Importa ScraperFC 4.x. Devuelve el módulo FBref o None."""
    try:
        # En ScraperFC >=4.0 el import es con S mayúscula
        from ScraperFC import FBref
        return FBref
    except ImportError:
        # Intentar import alternativo (algunas instalaciones difieren)
        try:
            from scraperfc import FBref
            return FBref
        except ImportError:
            print("  [ERROR] ScraperFC no encontrado.")
            print("    Instalar: pip install ScraperFC")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Listar competiciones válidas en FBref (útil para debug)
# ─────────────────────────────────────────────────────────────────────────────

def list_valid_leagues():
    """Imprime las ligas válidas para FBref en ScraperFC 4.x."""
    FBref = _import_scraperfc()
    if FBref is None:
        return
    try:
        fbref = FBref()
        try:
            fbref.scrape_stats("2023-2024", "__INVALID__", "standard")
        except Exception as e:
            print(str(e))
    except Exception as e:
        print(f"  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Descarga de stats de jugadores via FBref (ScraperFC 4.x)
# ─────────────────────────────────────────────────────────────────────────────

def download_fbref_stats(stat_type: str = "standard") -> pd.DataFrame:
    FBref = _import_scraperfc()
    if FBref is None:
        return pd.DataFrame()

    print(f"  Descargando FBref [{stat_type}] — {WC2026_LEAGUE} {WC2026_YEAR}...")
    try:
        fbref = FBref()
        # Firma correcta en 4.x: posicionales, devuelve tupla (squad, opp, players)
        squad_df, opp_df, player_df = fbref.scrape_stats(
            WC2026_YEAR, WC2026_LEAGUE, stat_type
        )
        if player_df is None or (isinstance(player_df, pd.DataFrame) and player_df.empty):
            print(f"    [AVISO] FBref devolvió datos vacíos para [{stat_type}].")
            return pd.DataFrame()

        print(f"    [OK] {len(player_df):,} filas descargadas")
        time.sleep(SCRAPER_DELAY)
        return player_df

    except Exception as e:
        err = str(e)
        if "InvalidLeague" in err or "not valid" in err.lower():
            print(f"    [ERROR] '{WC2026_LEAGUE}' no es un nombre válido.")
            print(f"    Ejecuta con --list-leagues para ver opciones.")
        elif "NoMatchLinks" in err:
            print(f"    [AVISO] Sin partidos disponibles aún en FBref para {WC2026_YEAR}.")
        else:
            print(f"    [ERROR] {err[:200]}")
        return pd.DataFrame()


def download_all_stats(categories: list = None) -> dict:
    """Descarga múltiples categorías de stats. Devuelve {cat: DataFrame}."""
    cats = categories or ["standard", "shooting"]
    results = {}
    for cat in cats:
        df = download_fbref_stats(cat)
        if not df.empty:
            results[cat] = df
            out = DATA_RAW / f"wc2026_fbref_{cat}.parquet"
            df.to_parquet(out, index=False)
            print(f"    Guardado: {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Agregar stats de jugadores a nivel de equipo
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_teams(stats_dict: dict) -> pd.DataFrame:
    """
    Agrega stats de jugadores a nivel de equipo para usar como features.

    ScraperFC 4.x nombra las columnas de forma diferente a versiones anteriores.
    Esta función es robusta a variaciones de nombres de columna.
    """
    standard = stats_dict.get("standard", pd.DataFrame())
    shooting  = stats_dict.get("shooting",  pd.DataFrame())

    if standard.empty and shooting.empty:
        return pd.DataFrame()

    df = standard if not standard.empty else shooting

    # Detectar columna de equipo (varía entre versiones)
    team_col = None
    for candidate in ["squad", "Squad", "team", "Team", "club", "Club"]:
        if candidate in df.columns:
            team_col = candidate
            break

    if team_col is None:
        print("  [AVISO] No se encontró columna de equipo en stats de FBref.")
        return pd.DataFrame()

    records = []
    for team_name, grp in df.groupby(team_col):
        # xG (varios nombres posibles en ScraperFC 4.x)
        xg = 0.0
        for col in ["xg", "xG", "expected_goals", "xg_x", "npxg"]:
            if col in grp.columns:
                xg = pd.to_numeric(grp[col], errors="coerce").sum()
                break

        # Goles
        goals = 0
        for col in ["goals", "gls", "Gls", "g", "G"]:
            if col in grp.columns:
                goals = int(pd.to_numeric(grp[col], errors="coerce").sum())
                break

        # Shots (desde shooting si está disponible)
        xg_shots = xg
        if not shooting.empty and team_col in shooting.columns:
            sh_grp = shooting[shooting[team_col] == team_name]
            for col in ["xg", "xG", "expected_goals"]:
                if col in sh_grp.columns:
                    xg_shots = pd.to_numeric(sh_grp[col], errors="coerce").sum()
                    break

        records.append({
            "team"               : str(team_name),
            "source"             : "fbref_wc2026_scraperfc4",
            "timestamp"          : datetime.utcnow().isoformat(),
            "xg_total_wc"        : round(float(xg), 3),
            "xg_shots_wc"        : round(float(xg_shots), 3),
            "goals_for_wc"       : goals,
            "xg_conversion_wc"   : round(goals / xg, 3) if xg > 0 else np.nan,
            "n_players"          : len(grp),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: snapshot manual desde nuestros parquets ya generados
# ─────────────────────────────────────────────────────────────────────────────

def build_manual_snapshot() -> pd.DataFrame:
    """
    Construye snapshot de features por selección desde los parquets existentes.
    No requiere internet. Usa StatsBomb (xG histórico) + form_rolling.

    Esta es la fuente más fiable durante el torneo porque:
    - StatsBomb tiene los datos de WC2018, WC2022, Euro2024, etc. → parámetros
      de xG estructurales de cada selección
    - form_rolling tiene la forma pre-torneo basada en 49k partidos históricos
    - FBref tarda días en subir datos del WC2026 partido a partido
    """
    ensure_dirs()
    from simulation.wc2026_fixtures import WC2026_GROUPS, normalize_team_name

    print("  Construyendo snapshot manual desde parquets existentes...")

    xg_path   = Path(__file__).parent.parent / "data" / "features" / "xg_derived.parquet"
    form_path = Path(__file__).parent.parent / "data" / "features" / "form_rolling.parquet"

    xg_df   = pd.read_parquet(xg_path)   if xg_path.exists()   else pd.DataFrame()
    form_df = pd.read_parquet(form_path) if form_path.exists() else pd.DataFrame()

    if not xg_df.empty:
        xg_df["date"] = pd.to_datetime(xg_df["date"])
    if not form_df.empty:
        form_df["date"] = pd.to_datetime(form_df["date"])

    all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
    records = []

    for team in all_teams:
        team_norm = normalize_team_name(team)

        # xG reciente (últimos 5 partidos en StatsBomb)
        xg_recent = pd.DataFrame()
        if not xg_df.empty and "team" in xg_df.columns:
            mask = xg_df["team"].isin([team, team_norm])
            xg_recent = xg_df[mask].sort_values("date").tail(5)

        xg_mean  = float(xg_recent["xg"].mean())  if not xg_recent.empty and "xg"  in xg_recent.columns else np.nan
        xga_mean = float(xg_recent["xga"].mean()) if not xg_recent.empty and "xga" in xg_recent.columns else np.nan
        ppda_mean= float(xg_recent["ppda"].mean()) if not xg_recent.empty and "ppda" in xg_recent.columns else np.nan

        # Forma reciente (último snapshot en form_rolling)
        form_weighted = np.nan
        form_pts6     = np.nan
        momentum      = np.nan
        if not form_df.empty and "team" in form_df.columns:
            mask = form_df["team"].isin([team, team_norm])
            last = form_df[mask].sort_values("date").tail(1)
            if not last.empty:
                form_weighted = float(last["form_weighted"].iloc[0])   if "form_weighted"  in last.columns else np.nan
                form_pts6     = float(last["form_pts_last6"].iloc[0])  if "form_pts_last6" in last.columns else np.nan
                momentum      = float(last["momentum_trend"].iloc[0])  if "momentum_trend" in last.columns else np.nan

        records.append({
            "team"             : team,
            "group"            : next((g for g, ts in WC2026_GROUPS.items() if team in ts), "?"),
            "xg_avg_recent"    : round(xg_mean,  3) if pd.notna(xg_mean)   else None,
            "xga_avg_recent"   : round(xga_mean, 3) if pd.notna(xga_mean)  else None,
            "ppda_avg_recent"  : round(ppda_mean, 3) if pd.notna(ppda_mean) else None,
            "form_weighted"    : round(form_weighted, 3) if pd.notna(form_weighted) else None,
            "form_pts_last6"   : round(form_pts6, 3)    if pd.notna(form_pts6)     else None,
            "momentum_trend"   : round(momentum, 3)     if pd.notna(momentum)      else None,
            "n_matches_sb"     : len(xg_recent),
            "source"           : "statsbomb_historical",
            "timestamp"        : datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(records)
    out = DATA_RAW / "wc2026_team_snapshot.parquet"
    df.to_parquet(out, index=False)
    print(f"  [OK] Snapshot manual guardado: {out} ({len(df)} equipos)")

    # Resumen rápido
    n_con_xg   = df["xg_avg_recent"].notna().sum()
    n_con_form = df["form_weighted"].notna().sum()
    print(f"    Equipos con xG StatsBomb:   {n_con_xg}/{len(df)}")
    print(f"    Equipos con forma rolling:  {n_con_form}/{len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(categories: list = None, manual_only: bool = False,
         list_leagues: bool = False):
    ensure_dirs()

    print("═" * 60)
    print("WC2026 — Actualización de datos en tiempo real")
    print("═" * 60)

    # ── Listar ligas válidas ──────────────────────────────────────────────────
    if list_leagues:
        print("\nListando ligas válidas para FBref en ScraperFC 4.x...")
        list_valid_leagues()
        return

    # ── Modo solo fallback manual ─────────────────────────────────────────────
    if manual_only:
        print("\n[Modo manual] Usando solo datos locales (sin scraping).")
        build_manual_snapshot()
        return

    # ── Intentar FBref via ScraperFC 4.x ─────────────────────────────────────
    print(f"\n[1/2] Intentando FBref ({WC2026_LEAGUE} {WC2026_YEAR})...")
    cats = categories or ["standard", "shooting"]
    stats = download_all_stats(cats)

    if stats:
        print(f"\n  Agregando a nivel de equipo...")
        team_stats = aggregate_to_teams(stats)
        if not team_stats.empty:
            out = DATA_RAW / "wc2026_fbref_teams.parquet"
            team_stats.to_parquet(out, index=False)
            print(f"  [OK] Stats por equipo guardadas: {out} ({len(team_stats)} equipos)")
    else:
        print("  [AVISO] FBref no devolvió datos. Normal si el torneo recién empezó.")

    # ── Siempre generar snapshot manual como base sólida ─────────────────────
    print(f"\n[2/2] Generando snapshot manual (siempre como base del modelo)...")
    build_manual_snapshot()

    print("\n[OK] Proceso completado.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Datos WC2026 en tiempo real (ScraperFC 4.x compatible)"
    )
    parser.add_argument(
        "--category", nargs="+",
        choices=STAT_TYPES, default=["standard", "shooting"],
        help="Categorías FBref (default: standard shooting)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Descargar todas las categorías disponibles"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Solo usar datos locales, sin scraping (más rápido y fiable)"
    )
    parser.add_argument(
        "--list-leagues", action="store_true",
        help="Listar nombres válidos de ligas para FBref en ScraperFC 4.x"
    )
    args = parser.parse_args()

    cats = STAT_TYPES if args.all else args.category
    main(
        categories  = cats,
        manual_only = args.manual,
        list_leagues= args.list_leagues,
    )