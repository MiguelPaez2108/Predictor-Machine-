"""
08_scrape_sofascore_wc2026.py — Stats del WC2026 desde Sofascore via ScraperFC
================================================================================
Usa ScraperFC.Sofascore(), que internamente resuelve el challenge anti-bot de
Cloudflare con un navegador real (Botasaurus). El intento anterior con
`requests` puro fallaba con 403 en todos los endpoints — confirmado en
diagnóstico previo. Esta versión no llama a la API cruda directamente.

Datos que extrae por partido (limitado a lo que expone esta versión de
ScraperFC — ver LIMITACIONES abajo):
  - match_team_stats.parquet    : stats de equipo (xG, posesión, tiros,
                                   corners, faltas, etc.) — una fila por
                                   (match_id, team), formato ancho.
  - match_player_stats.parquet  : stats individuales por jugador (rating,
                                   goles, asistencias, xG, xA, duelos,
                                   pases, etc.). Columnas VARÍAN entre
                                   partidos (ScraperFC solo agrega columnas
                                   como yellowCard/redCard cuando ocurrieron
                                   en ese partido) — se concatenan con unión
                                   de columnas (NaN donde no aplica).
  - match_shots.parquet         : un tiro por fila, con xG y xGOT, ubicación,
                                   minuto, jugador, tipo de tiro/situación.
  - match_goals.parquet         : subconjunto de match_shots donde el tiro
                                   terminó en gol (goalType / incidentType),
                                   con minuto y jugador — sustituye un
                                   endpoint de incidentes que no existe en
                                   esta versión de ScraperFC.
  - matches_index.parquet       : metadata de cada partido (equipos, marcador,
                                   estado, fecha, grupo/ronda) — se construye
                                   directamente desde get_match_dicts, sin
                                   llamadas extra.

LIMITACIONES (confirmadas en diagnóstico, no son un bug de este script):
  - No hay método de incidentes en esta versión de ScraperFC → NO se pueden
    obtener tarjetas amarillas/rojas ni sustituciones con minuto exacto.
    Los goles SÍ se recuperan, derivados de match_shots.
  - Los nombres de columnas de stats de equipo y jugador dependen de lo que
    la API de Sofascore decidió incluir para ESE partido en particular.
    Por eso se usa pd.concat (une columnas, rellena NaN) en vez de un
    esquema fijo.

Salida: data/raw/sofascore_wc2026/

Uso:
  python ingestion/08_scrape_sofascore_wc2026.py              # scrapea todos los finalizados
  python ingestion/08_scrape_sofascore_wc2026.py --new-only   # solo partidos nuevos
  python ingestion/08_scrape_sofascore_wc2026.py --dry-run    # lista sin descargar
  python ingestion/08_scrape_sofascore_wc2026.py --delay 4    # más conservador con rate limit

NOTA: usa navegador (Botasaurus) por debajo. Es normal que abra Chrome.
La primera corrida completa (~66 partidos × 3 llamadas) puede tardar 15-20 min.
"""

import sys
import re
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, ensure_dirs

import ScraperFC as sfc


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

YEAR     = "2026"
LEAGUE   = "FIFA World Cup"
OUT_DIR  = DATA_RAW / "sofascore_wc2026"

DEFAULT_DELAY = 2.5  # segundos entre llamadas, aunque use navegador


# ─────────────────────────────────────────────────────────────────────────────
# Mapeo de nombres Sofascore → nombres canónicos del proyecto
# (mismo criterio que API_TO_CANONICAL en ingestion/sync_results.py)
# ─────────────────────────────────────────────────────────────────────────────

SOFASCORE_NAME_MAP = {
    "Türkiye"               : "Turkey",
    "Czechia"               : "Czech Republic",
    "Côte d'Ivoire"         : "Ivory Coast",
    "Bosnia & Herzegovina"  : "Bosnia and Herzegovina",
    "Bosnia-Herzegovina"    : "Bosnia and Herzegovina",
    "Korea Republic"        : "South Korea",
    "USA"                   : "United States",
    "IR Iran"               : "Iran",
    "Cabo Verde"            : "Cape Verde",
}


def canonical(name: str) -> str:
    return SOFASCORE_NAME_MAP.get(name, name)


def check_team_names(matches: list) -> None:
    """
    Compara los nombres de equipo que devuelve Sofascore contra los 48
    nombres canónicos de simulation/wc2026_fixtures.py. Avisa de cualquier
    nombre sin mapear, en vez de descubrirlo recién al hacer el merge con
    el resto del pipeline.
    """
    try:
        from simulation.wc2026_fixtures import all_teams
        canon_set = set(all_teams())
    except Exception as e:
        print(f"  [AVISO] No se pudo importar wc2026_fixtures para validar nombres: {e}")
        return

    sofa_names = set()
    for m in matches:
        sofa_names.add((m.get("homeTeam") or {}).get("name", "?"))
        sofa_names.add((m.get("awayTeam") or {}).get("name", "?"))

    unmapped = sorted(n for n in sofa_names if canonical(n) not in canon_set)
    if unmapped:
        print("  [AVISO] Nombres de Sofascore que no matchean ningún equipo canónico:")
        for n in unmapped:
            print(f"    '{n}'  ->  canonical = '{canonical(n)}'  (sin match en WC2026_GROUPS)")
        print("    Agregalos a SOFASCORE_NAME_MAP arriba si corresponde a un equipo real.")
    else:
        print("  [OK] Los 48 nombres de equipo de Sofascore matchean con WC2026_GROUPS.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de limpieza
# ─────────────────────────────────────────────────────────────────────────────

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    scrape_player_match_stats devuelve columnas DUPLICADAS: 'position' y
    'jerseyNumber' aparecen 2 veces (una del perfil general del jugador,
    otra del contexto específico del partido — confirmado por el orden de
    columnas: la 2da ocurrencia aparece justo al lado de 'substitute' y
    el resto de las stats del partido).

    pd.concat falla con columnas no-únicas (InvalidIndexError: "Reindexing
    only valid with uniquely valued Index objects") al intentar alinear
    el índice de columnas entre varios DataFrames. Nos quedamos con la
    ÚLTIMA ocurrencia de cada nombre duplicado (la específica del partido,
    más relevante para el análisis que el dato de perfil genérico).
    """
    if df.columns.duplicated().any():
        dup_names = sorted(set(df.columns[df.columns.duplicated()].tolist()))
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        print(f"      [info] columnas duplicadas deduplicadas (keep=last): {dup_names}")
    return df


def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas con valores anidados (dict/list) que rompen la
    inferencia de esquema de parquet (ej. 'country', 'fieldTranslations',
    'ratingVersions' en scrape_player_match_stats), y deduplica columnas
    repetidas (ver _dedupe_columns).
    """
    if df is None or df.empty:
        return df
    df = _dedupe_columns(df.copy())
    drop_cols = []
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        if isinstance(sample.iloc[0], (dict, list)):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def _to_number(val):
    """Convierte '45%', '1,23', 53.0, etc. a float. Deja el valor tal cual si no es numérico."""
    if val is None:
        return None
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    s = str(val).replace("%", "").replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return val


def _camel_to_snake(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", str(s))
    return s.lower()


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip().lower())
    return s.strip("_")


def _stat_colname(key, name) -> str:
    if key:
        return _camel_to_snake(str(key))
    return _slugify(name)


def _save_accumulated(df: pd.DataFrame, path: Path, subset: list = None) -> pd.DataFrame:
    """Guarda df acumulando con lo que ya existía en disco (para --new-only)."""
    if path.exists():
        try:
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception as e:
            print(f"    [AVISO] No se pudo leer {path.name} existente, se sobreescribe: {e}")
    if subset:
        subset = [c for c in subset if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="last")
    df.to_parquet(path, index=False)
    return df


def _get_already_done_matches() -> set:
    """
    Un partido se considera 'completo' solo si está presente en LOS TRES
    outputs (team_stats, player_stats, shots) — no alcanza con que esté
    en uno solo. Esto evita el problema de que un run anterior se haya
    caído a mitad de camino (como pasó con el bug de columnas duplicadas:
    team_stats se guardó pero player_stats/shots nunca llegaron a disco)
    y --new-only los considere "ya hechos" para siempre por error.
    """
    required = [
        ("match_team_stats.parquet",   "match_id"),
        ("match_player_stats.parquet", "match_id"),
        ("match_shots.parquet",        "match_id"),
    ]
    done_sets = []
    for fname, col in required:
        path = OUT_DIR / fname
        if not path.exists():
            return set()  # falta un archivo entero -> ningún partido cuenta como completo
        df = pd.read_parquet(path)
        if col not in df.columns:
            return set()
        done_sets.append(set(df[col].dropna().astype(int).tolist()))
    return set.intersection(*done_sets) if done_sets else set()


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del índice de partidos (sin llamadas extra a la API)
# ─────────────────────────────────────────────────────────────────────────────

def is_finished(match: dict) -> bool:
    status = match.get("status") or {}
    return status.get("type") == "finished" or status.get("description") == "Ended"


def build_matches_index(matches: list) -> pd.DataFrame:
    rows = []
    for m in matches:
        status     = m.get("status") or {}
        round_info = m.get("roundInfo") or {}
        tourn      = m.get("tournament") or {}
        ts         = m.get("startTimestamp")
        date_str   = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if ts else None
        )
        home = canonical((m.get("homeTeam") or {}).get("name", "?"))
        away = canonical((m.get("awayTeam") or {}).get("name", "?"))

        rows.append({
            "sofascore_match_id": m.get("id"),
            "match_date"        : date_str,
            "stage"             : tourn.get("name"),      # ej. "FIFA World Cup, Group I"
            "round"             : round_info.get("round") or round_info.get("name"),
            "home_team"         : home,
            "away_team"         : away,
            "home_goals"        : (m.get("homeScore") or {}).get("current"),
            "away_goals"        : (m.get("awayScore") or {}).get("current"),
            "status"            : status.get("description"),
            "has_xg"            : m.get("hasXg"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Stats de equipo (pivot de formato largo a ancho)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_team_stats_wide(ss, match_id: int, home: str, away: str) -> list:
    """
    scrape_team_match_stats devuelve formato largo: una fila por stat,
    con columnas homeValue/awayValue. Lo pivoteamos a 2 filas (home/away)
    con una columna por stat.
    """
    try:
        df = ss.scrape_team_match_stats(match_id)
    except Exception as e:
        print(f"    [FALLO] team stats: {e}")
        return []

    if df is None or df.empty:
        return []

    if "period" in df.columns:
        df = df[df["period"] == "ALL"]

    home_row = {"match_id": match_id, "team": home, "is_home": True}
    away_row = {"match_id": match_id, "team": away, "is_home": False}

    for _, r in df.iterrows():
        key  = r.get("key")
        name = r.get("name")
        col  = _stat_colname(key, name)
        if not col:
            continue
        home_row[col] = _to_number(r.get("homeValue"))
        away_row[col] = _to_number(r.get("awayValue"))

    return [home_row, away_row]


# ─────────────────────────────────────────────────────────────────────────────
# Stats de jugadores
# ─────────────────────────────────────────────────────────────────────────────

def scrape_player_stats_clean(ss, match_id: int) -> pd.DataFrame:
    try:
        df = ss.scrape_player_match_stats(match_id)
    except Exception as e:
        print(f"    [FALLO] player stats: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _clean_for_parquet(df)
    df["match_id"] = match_id
    if "teamName" in df.columns:
        df["team"] = df["teamName"].map(canonical)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tiros y goles (sustituye al endpoint de incidentes, que no existe acá)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_shots_clean(ss, match_id: int) -> pd.DataFrame:
    try:
        df = ss.scrape_match_shots(match_id)
    except Exception as e:
        print(f"    [FALLO] shots: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _clean_for_parquet(df)
    df["match_id"] = match_id
    return df


def extract_goals(shots_df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Filtra los tiros que terminaron en gol, usando goalType o incidentType."""
    if shots_df is None or shots_df.empty:
        return pd.DataFrame()

    mask = pd.Series(False, index=shots_df.index)
    if "goalType" in shots_df.columns:
        mask = mask | shots_df["goalType"].notna()
    if "incidentType" in shots_df.columns:
        mask = mask | (shots_df["incidentType"].astype(str).str.lower() == "goal")

    goals = shots_df[mask].copy()
    if goals.empty:
        return goals

    if "isHome" in goals.columns:
        goals["team"] = goals["isHome"].map({True: home, False: away})
    return goals


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(new_only: bool = False, dry_run: bool = False, delay: float = DEFAULT_DELAY):
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 62)
    print("  SOFASCORE WC2026 — Scraping via ScraperFC")
    print("═" * 62)

    ss = sfc.Sofascore()

    print("\n[1/4] Obteniendo partidos del WC2026...")
    matches = ss.get_match_dicts(year=YEAR, league=LEAGUE)
    print(f"  Total partidos devueltos: {len(matches)}")

    check_team_names(matches)

    idx_df = build_matches_index(matches)
    idx_path = OUT_DIR / "matches_index.parquet"
    idx_df.to_parquet(idx_path, index=False)
    print(f"  [OK] matches_index.parquet: {len(idx_df)} partidos")

    finished_df = idx_df[
        idx_df["sofascore_match_id"].isin(
            [m["id"] for m in matches if is_finished(m)]
        )
    ].reset_index(drop=True)
    print(f"  Finalizados: {len(finished_df)}/{len(idx_df)}")

    to_scrape = finished_df
    if new_only:
        already_done = _get_already_done_matches()
        to_scrape = finished_df[
            ~finished_df["sofascore_match_id"].astype(int).isin(already_done)
        ].reset_index(drop=True)
        print(f"  Completos en los 3 parquets: {len(already_done)}  |  Pendientes: {len(to_scrape)}")

    if dry_run:
        print("\n[DRY RUN] Partidos que se scrapearían:")
        for _, r in to_scrape.iterrows():
            print(f"  [{r['sofascore_match_id']}] {r['home_team']} "
                  f"{r['home_goals']}-{r['away_goals']} {r['away_team']}  ({r['stage']})")
        return

    print(f"\n[2/4] Scrapeando {len(to_scrape)} partidos "
          f"(delay={delay}s entre llamadas)...")

    all_team_stats   = []
    all_player_stats = []
    all_shots        = []

    for i, row in to_scrape.iterrows():
        mid  = int(row["sofascore_match_id"])
        home = row["home_team"]
        away = row["away_team"]
        print(f"\n  [{i+1}/{len(to_scrape)}] {home} {row['home_goals']}-{row['away_goals']} "
              f"{away}  (id={mid})")

        ts_rows = scrape_team_stats_wide(ss, mid, home, away)
        if ts_rows:
            all_team_stats.extend(ts_rows)
            print(f"    [OK] Team stats: {len(ts_rows[0]) - 3} métricas")
        time.sleep(delay)

        pdf = scrape_player_stats_clean(ss, mid)
        if not pdf.empty:
            all_player_stats.append(pdf)
            print(f"    [OK] Player stats: {len(pdf)} jugadores")
        time.sleep(delay)

        sdf = scrape_shots_clean(ss, mid)
        if not sdf.empty:
            all_shots.append(sdf)
            print(f"    [OK] Shots: {len(sdf)} tiros")
        time.sleep(delay)

    # ── Guardar ──────────────────────────────────────────────────────────────
    print("\n[3/4] Guardando resultados...")

    if all_team_stats:
        try:
            df_ts = pd.DataFrame(all_team_stats)
            df_ts = _save_accumulated(
                df_ts, OUT_DIR / "match_team_stats.parquet",
                subset=["match_id", "team"]
            )
            print(f"  [OK] match_team_stats.parquet: {len(df_ts)} filas")
        except Exception as e:
            print(f"  [FALLO] No se pudo guardar match_team_stats.parquet: {e}")

    if all_player_stats:
        try:
            df_ps = pd.concat(all_player_stats, ignore_index=True)
            id_col = "id" if "id" in df_ps.columns else None
            subset = ["match_id", id_col] if id_col else ["match_id", "name"]
            df_ps = _save_accumulated(
                df_ps, OUT_DIR / "match_player_stats.parquet", subset=subset
            )
            print(f"  [OK] match_player_stats.parquet: {len(df_ps)} filas")
        except Exception as e:
            print(f"  [FALLO] No se pudo guardar match_player_stats.parquet: {e}")

    if all_shots:
        try:
            df_sh = pd.concat(all_shots, ignore_index=True)
            shot_id_col = "id" if "id" in df_sh.columns else None
            subset = ["match_id", shot_id_col] if shot_id_col else None
            df_sh = _save_accumulated(
                df_sh, OUT_DIR / "match_shots.parquet", subset=subset
            )
            print(f"  [OK] match_shots.parquet: {len(df_sh)} filas")

            # Goles derivados de los tiros (no hay endpoint de incidentes)
            try:
                match_teams = idx_df.set_index("sofascore_match_id")[["home_team", "away_team"]]
                goal_rows = []
                for mid, grp in df_sh.groupby("match_id"):
                    if mid not in match_teams.index:
                        continue
                    h, a = match_teams.loc[mid, "home_team"], match_teams.loc[mid, "away_team"]
                    g = extract_goals(grp, h, a)
                    if not g.empty:
                        goal_rows.append(g)
                if goal_rows:
                    df_goals = pd.concat(goal_rows, ignore_index=True)
                    subset = ["match_id", shot_id_col] if shot_id_col else None
                    df_goals = _save_accumulated(
                        df_goals, OUT_DIR / "match_goals.parquet", subset=subset
                    )
                    print(f"  [OK] match_goals.parquet: {len(df_goals)} goles")
            except Exception as e:
                print(f"  [FALLO] No se pudo derivar/guardar match_goals.parquet: {e}")
        except Exception as e:
            print(f"  [FALLO] No se pudo guardar match_shots.parquet: {e}")

    # ── Resumen xG ───────────────────────────────────────────────────────────
    ts_path = OUT_DIR / "match_team_stats.parquet"
    if ts_path.exists():
        df_ts = pd.read_parquet(ts_path)
        xg_col = next((c for c in df_ts.columns if "expected_goal" in c.lower()), None)
        if xg_col:
            print(f"\n[4/4] xG promedio por equipo/partido: {df_ts[xg_col].mean():.3f}")
            top_xg = (
                df_ts.groupby("team")[xg_col].sum()
                .sort_values(ascending=False).head(10)
            )
            print("  Top 10 equipos por xG acumulado:")
            for team, xg in top_xg.items():
                print(f"    {team:25s}: {xg:.2f}")

    print(f"\n[OK] Scraping completado. Datos en: {OUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraper de stats WC2026 desde Sofascore via ScraperFC"
    )
    parser.add_argument("--new-only", action="store_true",
                        help="Solo scrapear partidos no descargados aún (incremental)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Listar partidos disponibles sin descargar")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Segundos de espera entre llamadas (default: {DEFAULT_DELAY})")
    args = parser.parse_args()
    main(new_only=args.new_only, dry_run=args.dry_run, delay=args.delay)