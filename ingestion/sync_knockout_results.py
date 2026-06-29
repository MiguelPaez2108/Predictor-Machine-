"""
sync_knockout_results.py — Sincroniza resultados reales de la fase eliminatoria
=================================================================================
Agrega resultados del R32 (y fases siguientes) a resultados_reales.json
y dispara la actualización de stats de Sofascore para esos partidos.

Uso:
  python ingestion/sync_knockout_results.py                    # sync vía API
  python ingestion/sync_knockout_results.py --add "Brasil 2-1 Japón"
  python ingestion/sync_knockout_results.py --add "South Africa 0-1 Canada"
  python ingestion/sync_knockout_results.py --list             # ver todos los registrados
"""

import sys
import re
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_WC_CODE,
    RESULTS_FILE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Mapeo nombre natural → nombre canónico del proyecto
# ─────────────────────────────────────────────────────────────────────────────

NAME_ALIASES = {
    # Español → canónico
    "brasil": "Brazil", "brazil": "Brazil",
    "brasil": "Brazil",
    "japón": "Japan", "japon": "Japan", "japan": "Japan",
    "sudáfrica": "South Africa", "sudafrica": "South Africa",
    "south africa": "South Africa",
    "canadá": "Canada", "canada": "Canada",
    "alemania": "Germany", "germany": "Germany",
    "paraguay": "Paraguay",
    "francia": "France", "france": "France",
    "suecia": "Sweden", "sweden": "Sweden",
    "países bajos": "Netherlands", "paises bajos": "Netherlands",
    "netherlands": "Netherlands",
    "marruecos": "Morocco", "morocco": "Morocco",
    "portugal": "Portugal",
    "croacia": "Croatia", "croatia": "Croatia",
    "españa": "Spain", "espana": "Spain", "spain": "Spain",
    "austria": "Austria",
    "estados unidos": "United States", "usa": "United States",
    "united states": "United States",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia y herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bélgica": "Belgium", "belgica": "Belgium", "belgium": "Belgium",
    "senegal": "Senegal",
    "costa de marfil": "Ivory Coast", "ivory coast": "Ivory Coast",
    "noruega": "Norway", "norway": "Norway",
    "méxico": "Mexico", "mexico": "Mexico",
    "ecuador": "Ecuador",
    "inglaterra": "England", "england": "England",
    "república democrática del congo": "DR Congo",
    "rep. democrática del congo": "DR Congo",
    "dr congo": "DR Congo", "dr. congo": "DR Congo",
    "argentina": "Argentina",
    "cabo verde": "Cape Verde", "cape verde": "Cape Verde",
    "australia": "Australia",
    "egipto": "Egypt", "egypt": "Egypt",
    "suiza": "Switzerland", "switzerland": "Switzerland",
    "argelia": "Algeria", "algeria": "Algeria",
    "colombia": "Colombia",
    "ghana": "Ghana",
}

# Etapa del torneo
STAGE_MAP = {
    "LAST_32": "R32",
    "ROUND_OF_32": "R32",
    "LAST_16": "R16",
    "ROUND_OF_16": "R16",
    "QUARTER_FINALS": "QF",
    "SEMI_FINALS": "SF",
    "FINAL": "F",
    "THIRD_PLACE": "3P",
    "GROUP_STAGE": "GROUP",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def c(text, code):
    codes = {
        "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
        "cyan": "\033[96m", "bold": "\033[1m", "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return codes.get(code, "") + text + codes["reset"]


def normalize_team(name: str) -> Optional[str]:
    """Convierte nombre libre al nombre canónico del proyecto."""
    key = name.strip().lower()
    if key in NAME_ALIASES:
        return NAME_ALIASES[key]
    # Búsqueda parcial
    for alias, canonical in NAME_ALIASES.items():
        if alias in key or key in alias:
            return canonical
    # Devolver con capitalización correcta si no hay alias
    return name.strip().title()


def parse_result_string(raw: str) -> Optional[Tuple[str, int, str, int]]:
    """
    Parsea strings como:
      'Brasil 2-1 Japón'
      'South Africa 0 Canada 1'
      'Argentina 3 - 0 Cape Verde'
    """
    raw = raw.strip()

    # Formato: Team1 X-Y Team2
    m = re.match(r'^(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+)$', raw)
    if m:
        return (normalize_team(m.group(1)), int(m.group(2)),
                normalize_team(m.group(4)), int(m.group(3)))

    # Formato: Team1 X Team2 Y
    parts = raw.split()
    nums = [(i, int(p)) for i, p in enumerate(parts) if p.isdigit()]
    if len(nums) >= 2:
        i1, n1 = nums[0]
        i2, n2 = nums[1]
        team1 = " ".join(parts[:i1])
        team2 = " ".join(parts[i1+1:i2])
        if team1 and team2:
            return normalize_team(team1), n1, normalize_team(team2), n2

    return None


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def match_key(home, away):
    return "|".join(sorted([home.lower(), away.lower()]))


# ─────────────────────────────────────────────────────────────────────────────
# Agregar resultado manual
# ─────────────────────────────────────────────────────────────────────────────

def add_knockout_result(home: str, away: str, home_goals: int, away_goals: int,
                         stage: str = "R32", date: str = None,
                         trigger_sofascore: bool = True) -> dict:
    """Agrega un resultado de fase eliminatoria a resultados_reales.json."""
    results = load_results()
    key = match_key(home, away)
    existing_keys = {match_key(r["home_team"], r["away_team"]): i
                     for i, r in enumerate(results)}

    entry = {
        "home_team":  home,
        "away_team":  away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "stage":      stage,
        "group":      stage,
        "date":       date or datetime.now().strftime("%Y-%m-%d"),
        "synced_at":  datetime.now().isoformat(),
        "source":     "manual",
    }

    if key in existing_keys:
        results[existing_keys[key]] = entry
        status = "updated"
    else:
        results.append(entry)
        status = "created"

    save_results(results)

    winner = home if home_goals > away_goals else (away if away_goals > home_goals else "Empate")
    print(c(f"\n  [{status.upper()}] {home} {home_goals}-{away_goals} {away}", "green"))
    print(c(f"  Ganador: {winner}  |  Etapa: {stage}", "cyan"))

    # Disparar scraping de Sofascore para este partido si corresponde
    if trigger_sofascore:
        _try_sofascore_update(home, away)

    return {"status": status, "entry": entry}


# ─────────────────────────────────────────────────────────────────────────────
# Actualización automática vía Sofascore
# ─────────────────────────────────────────────────────────────────────────────

def _try_sofascore_update(home: str, away: str):
    """Intenta actualizar stats de Sofascore para el partido recién registrado."""
    try:
        import ScraperFC as sfc
        from config import DATA_RAW

        sofascore_dir = DATA_RAW / "sofascore_wc2026"
        idx_path = sofascore_dir / "matches_index.parquet"

        if not idx_path.exists():
            print(c("  [INFO] matches_index no encontrado, omitiendo update Sofascore", "dim"))
            return

        import pandas as pd
        idx = pd.read_parquet(idx_path)

        # Buscar el partido en el índice
        mask = (
            ((idx["home_team"] == home) & (idx["away_team"] == away)) |
            ((idx["home_team"] == away) & (idx["away_team"] == home))
        )
        match_rows = idx[mask]

        if match_rows.empty:
            print(c(f"  [INFO] Partido {home} vs {away} no encontrado en matches_index.parquet", "dim"))
            print(c("  Puede que Sofascore aún no lo tenga o sea un partido nuevo de R32", "dim"))
            return

        mid = int(match_rows.iloc[0]["sofascore_match_id"])
        print(c(f"\n  [SOFASCORE] Actualizando stats del partido id={mid}...", "cyan"))

        ss = sfc.Sofascore()

        # Importar helpers del scraper principal
        from ingestion.sync_knockout_results import _scrape_one_match
        _scrape_one_match(ss, mid, home, away, sofascore_dir)

    except ImportError:
        print(c("  [INFO] ScraperFC no disponible, omitiendo update Sofascore", "dim"))
    except Exception as e:
        print(c(f"  [WARN] No se pudo actualizar Sofascore: {e}", "yellow"))


def _scrape_one_match(ss, match_id: int, home: str, away: str, out_dir: Path):
    """Scrapea un partido específico y lo agrega a los parquets existentes."""
    import pandas as pd
    import math
    from ingestion._scraper_helpers import (
        scrape_team_stats_wide, scrape_player_stats_clean,
        scrape_shots_clean, extract_goals, _save_accumulated,
        _camel_to_snake, _to_number, _clean_for_parquet,
    )

    print(f"    Scrapeando {home} vs {away} (id={match_id})...")

    # Team stats
    ts_rows = scrape_team_stats_wide(ss, match_id, home, away)
    if ts_rows:
        df_ts = pd.DataFrame(ts_rows)
        _save_accumulated(df_ts, out_dir / "match_team_stats.parquet",
                         subset=["match_id", "team"])
        print(c(f"    [OK] Team stats actualizadas", "green"))
    time.sleep(2.5)

    # Player stats
    pdf = scrape_player_stats_clean(ss, match_id)
    if not pdf.empty:
        _save_accumulated(pdf, out_dir / "match_player_stats.parquet",
                         subset=["match_id"])
        print(c(f"    [OK] Player stats: {len(pdf)} jugadores", "green"))
    time.sleep(2.5)

    # Shots + goals
    sdf = scrape_shots_clean(ss, match_id)
    if not sdf.empty:
        _save_accumulated(sdf, out_dir / "match_shots.parquet",
                         subset=["match_id"])
        goals = extract_goals(sdf, home, away)
        if not goals.empty:
            _save_accumulated(goals, out_dir / "match_goals.parquet",
                             subset=["match_id"])
        print(c(f"    [OK] Shots: {len(sdf)} tiros / Goles: {len(goals)}", "green"))


# ─────────────────────────────────────────────────────────────────────────────
# Sync automático vía football-data.org (incluye R32)
# ─────────────────────────────────────────────────────────────────────────────

def sync_from_api(dry_run: bool = False) -> list:
    """Sincroniza TODOS los partidos terminados (grupos + eliminatoria) desde la API."""
    if not FOOTBALL_DATA_API_KEY:
        print(c("  [ERROR] FOOTBALL_DATA_API_KEY no configurada en .env", "red"))
        return []

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{FOOTBALL_DATA_WC_CODE}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

    # Obtener todos los estados
    all_new = []
    for status in ["FINISHED"]:
        try:
            resp = requests.get(url, params={"status": status},
                               headers=headers, timeout=15)
            if resp.status_code != 200:
                print(c(f"  [ERROR] API HTTP {resp.status_code}", "red"))
                continue

            data = resp.json()
            matches = data.get("matches", [])
            print(c(f"  [API] {len(matches)} partidos terminados encontrados", "cyan"))

            results = load_results()
            existing_keys = {match_key(r["home_team"], r["away_team"])
                            for r in results}

            for m in matches:
                home_api = m["homeTeam"]["name"]
                away_api = m["awayTeam"]["name"]
                home = normalize_team(home_api)
                away = normalize_team(away_api)

                ft = m["score"].get("fullTime", {})
                hg = ft.get("home")
                ag = ft.get("away")

                if hg is None or ag is None:
                    continue

                stage_api = m.get("stage", "GROUP_STAGE")
                stage = STAGE_MAP.get(stage_api, stage_api)
                date = m.get("utcDate", "")[:10]

                key = match_key(home, away)
                if key not in existing_keys:
                    new_entry = {
                        "home_team":  home,
                        "away_team":  away,
                        "home_goals": int(hg),
                        "away_goals": int(ag),
                        "stage":      stage,
                        "group":      m.get("group", stage),
                        "date":       date,
                        "synced_at":  datetime.now().isoformat(),
                        "source":     "football-data.org",
                    }
                    all_new.append(new_entry)
                    print(c(f"  [NUEVO] [{stage}] {home} {hg}-{ag} {away}", "green"))
                else:
                    # Verificar si el marcador cambió
                    for r in results:
                        if match_key(r["home_team"], r["away_team"]) == key:
                            if r.get("home_goals") != int(hg) or r.get("away_goals") != int(ag):
                                r["home_goals"] = int(hg)
                                r["away_goals"] = int(ag)
                                r["updated_at"] = datetime.now().isoformat()
                                print(c(f"  [CORREGIDO] {home} {hg}-{ag} {away}", "yellow"))
                            break

        except Exception as e:
            print(c(f"  [ERROR] {e}", "red"))

    if all_new and not dry_run:
        results = load_results()
        results.extend(all_new)
        save_results(results)
        print(c(f"\n  [OK] {len(all_new)} resultados nuevos guardados", "green"))
    elif dry_run:
        print(c(f"\n  [DRY RUN] {len(all_new)} partidos nuevos detectados (no guardados)", "yellow"))

    return all_new


# ─────────────────────────────────────────────────────────────────────────────
# Listar resultados actuales
# ─────────────────────────────────────────────────────────────────────────────

def list_results(stage_filter: str = None):
    results = load_results()
    if not results:
        print(c("  No hay resultados registrados.", "yellow"))
        return

    # Agrupar por etapa
    by_stage = {}
    for r in results:
        stage = r.get("stage", r.get("group", "?"))
        by_stage.setdefault(stage, []).append(r)

    stage_order = ["GROUP", "GROUP_STAGE", "R32", "R16", "QF", "SF", "3P", "F"]
    stages_sorted = sorted(by_stage.keys(),
                          key=lambda s: stage_order.index(s) if s in stage_order else 99)

    print(c(f"\n  Total: {len(results)} resultados registrados\n", "bold"))

    for stage in stages_sorted:
        if stage_filter and stage != stage_filter.upper():
            continue
        matches = by_stage[stage]
        label = {"GROUP": "Fase de Grupos", "GROUP_STAGE": "Fase de Grupos",
                 "R32": "Ronda de 32", "R16": "Octavos de Final",
                 "QF": "Cuartos de Final", "SF": "Semifinales",
                 "F": "Final", "3P": "Tercer Puesto"}.get(stage, stage)

        print(c(f"  ── {label} ({len(matches)} partidos) ──", "cyan"))
        for r in sorted(matches, key=lambda x: x.get("date", "")):
            hg = r["home_goals"]
            ag = r["away_goals"]
            if hg > ag:
                color_code = "green"
            elif ag > hg:
                color_code = "red"
            else:
                color_code = "yellow"
            date = r.get("date", "?")
            src = r.get("source", "?")[:3].upper()
            print(f"    {date}  {r['home_team']:22s} "
                  f"{c(f'{hg}-{ag}', color_code):>14s} "
                  f"  {r['away_team']:22s}  {c(f'[{src}]', 'dim')}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Update automático de live features post-R32
# ─────────────────────────────────────────────────────────────────────────────

def update_live_features():
    """Re-genera wc2026_live_snapshot.parquet con todos los partidos disponibles."""
    try:
        from features.build_wc2026_live_features import build_wc2026_live_features
        from config import DATA_FEATURES

        print(c("\n  Re-generando features en vivo...", "cyan"))
        snapshot = build_wc2026_live_features()
        out = DATA_FEATURES / "wc2026_live_snapshot.parquet"
        snapshot.to_parquet(out, index=False)
        print(c(f"  [OK] {len(snapshot)} equipos actualizados → {out.name}", "green"))
    except Exception as e:
        print(c(f"  [WARN] No se pudo actualizar live features: {e}", "yellow"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sync de resultados de fase eliminatoria WC2026"
    )
    parser.add_argument("--add", type=str, default=None,
                        help="Agregar resultado: 'Brasil 2-1 Japón'")
    parser.add_argument("--stage", type=str, default="R32",
                        help="Etapa del partido (R32, R16, QF, SF, F)")
    parser.add_argument("--date", type=str, default=None,
                        help="Fecha del partido YYYY-MM-DD (default: hoy)")
    parser.add_argument("--sync-api", action="store_true",
                        help="Sincronizar desde football-data.org API")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar sin guardar")
    parser.add_argument("--list", action="store_true",
                        help="Listar todos los resultados")
    parser.add_argument("--update-features", action="store_true",
                        help="Re-generar live features después del sync")
    args = parser.parse_args()

    print(c("\n  WC2026 — SYNC DE RESULTADOS ELIMINATORIA", "bold"))
    print(c("  " + "─" * 45, "cyan"))

    if args.list:
        list_results()
        return

    if args.sync_api:
        new = sync_from_api(dry_run=args.dry_run)
        if new and args.update_features:
            update_live_features()
        return

    if args.add:
        parsed = parse_result_string(args.add)
        if not parsed:
            print(c(f"  [ERROR] No se pudo parsear: '{args.add}'", "red"))
            print(c("  Formato: 'Brasil 2-1 Japón' o 'Brazil 2 Japan 1'", "dim"))
            return
        home, hg, away, ag = parsed
        add_knockout_result(home, away, hg, ag,
                           stage=args.stage.upper(),
                           date=args.date,
                           trigger_sofascore=True)
        if args.update_features:
            update_live_features()
        return

    # Sin argumentos: mostrar ayuda
    parser.print_help()
    print()
    list_results()


if __name__ == "__main__":
    main()
