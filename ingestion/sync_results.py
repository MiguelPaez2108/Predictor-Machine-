"""
sync_results.py — Sincronización automática de resultados del Mundial 2026
============================================================================
Consulta la API de football-data.org para obtener resultados de partidos
terminados y actualiza automáticamente resultados_reales.json.

Uso:
  python ingestion/sync_results.py                   # Sync una vez
  python ingestion/sync_results.py --watch            # Re-sync cada 5 min
  python ingestion/sync_results.py --watch -i 2       # Re-sync cada 2 min
  python ingestion/sync_results.py --dry-run           # Solo mostrar sin guardar
  python ingestion/sync_results.py --tabla J           # Sync + tabla grupo J
  python ingestion/sync_results.py --all-tables        # Sync + todas las tablas
"""

import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_WC_CODE,
    RESULTS_FILE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MAPEO DE NOMBRES: football-data.org → nombres canónicos del proyecto
# ═══════════════════════════════════════════════════════════════════════════════

API_TO_CANONICAL = {
    # Nombres que difieren entre la API y nuestro dataset
    "Bosnia-Herzegovina"    : "Bosnia and Herzegovina",
    "Cape Verde Islands"    : "Cape Verde",
    "Congo DR"              : "DR Congo",
    "Czechia"               : "Czech Republic",
    "Curaçao"               : "Curaçao",       # mismo pero por si acaso
    "Korea Republic"        : "South Korea",
    "IR Iran"               : "Iran",
    "Türkiye"               : "Turkey",
    "Côte d'Ivoire"         : "Ivory Coast",
    "Cabo Verde"            : "Cape Verde",
    # Nombres que ya coinciden (no necesitan mapeo, pero los listamos
    # para documentar que fueron verificados):
    # Algeria, Argentina, Australia, Austria, Belgium, Brazil, Canada,
    # Colombia, Croatia, Ecuador, Egypt, England, France, Germany, Ghana,
    # Haiti, Iran, Iraq, Ivory Coast, Japan, Jordan, Mexico, Morocco,
    # Netherlands, New Zealand, Norway, Panama, Paraguay, Portugal, Qatar,
    # Saudi Arabia, Scotland, Senegal, South Africa, South Korea, Spain,
    # Sweden, Switzerland, Tunisia, Turkey, United States, Uruguay, Uzbekistan
}

# Mapeo de grupo API → letra de grupo del proyecto
API_GROUP_MAP = {
    "GROUP_A": "A", "GROUP_B": "B", "GROUP_C": "C", "GROUP_D": "D",
    "GROUP_E": "E", "GROUP_F": "F", "GROUP_G": "G", "GROUP_H": "H",
    "GROUP_I": "I", "GROUP_J": "J", "GROUP_K": "K", "GROUP_L": "L",
}


# ═══════════════════════════════════════════════════════════════════════════════
# COLORES TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def c(text: str, code: str) -> str:
    codes = {
        "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
        "blue": "\033[94m", "cyan": "\033[96m", "magenta": "\033[95m",
        "white": "\033[97m", "bold": "\033[1m", "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return codes.get(code, "") + text + codes["reset"]


# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_finished_matches() -> List[Dict]:
    """
    Consulta la API de football-data.org para obtener partidos terminados
    del Mundial 2026.
    """
    if not FOOTBALL_DATA_API_KEY:
        print(c("  [ERROR] Error: FOOTBALL_DATA_API_KEY no configurada en .env", "red"))
        print(c("    Regístrate gratis en https://www.football-data.org/client/register", "dim"))
        return []

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{FOOTBALL_DATA_WC_CODE}/matches"
    params = {"status": "FINISHED"}
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
    except requests.RequestException as e:
        print(c(f"  [ERROR] Error de conexión: {e}", "red"))
        return []

    if resp.status_code == 403:
        print(c("  [ERROR] API key inválida o sin permisos.", "red"))
        return []
    elif resp.status_code == 429:
        print(c("  [ERROR] Rate limit alcanzado. Espera un minuto.", "yellow"))
        return []
    elif resp.status_code != 200:
        print(c(f"  [ERROR] Error HTTP {resp.status_code}: {resp.text[:200]}", "red"))
        return []

    data = resp.json()
    api_matches = data.get("matches", [])

    # Convertir al formato del proyecto
    converted = []
    for m in api_matches:
        home_api = m["homeTeam"]["name"]
        away_api = m["awayTeam"]["name"]
        home = API_TO_CANONICAL.get(home_api, home_api)
        away = API_TO_CANONICAL.get(away_api, away_api)

        ft = m["score"].get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")

        if hg is None or ag is None:
            continue  # Score no disponible

        group_api = m.get("group", "")
        group = API_GROUP_MAP.get(group_api, "?")
        stage = m.get("stage", "GROUP_STAGE")
        date = m.get("utcDate", "")[:10]

        converted.append({
            "home_team": home,
            "away_team": away,
            "home_goals": int(hg),
            "away_goals": int(ag),
            "group": group,
            "stage": stage,
            "date": date,
            "matchday": m.get("matchday"),
            "api_id": m.get("id"),
        })

    return converted


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHIVOS LOCALES
# ═══════════════════════════════════════════════════════════════════════════════

def load_local_results() -> List[Dict]:
    """Carga los resultados reales desde disco."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_local_results(results: List[Dict]):
    """Guarda los resultados reales a disco."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _match_key(home: str, away: str) -> str:
    """Genera una clave única para un partido (orden-insensible)."""
    return "|".join(sorted([home, away]))


# ═══════════════════════════════════════════════════════════════════════════════
# SINCRONIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def sync(dry_run: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Sincroniza resultados de la API con el archivo local.

    Returns:
        (new_matches, updated_matches, all_results)
    """
    print()
    print(c("  ┌───────────────────────────────────────────────────────────┐", "cyan"))
    print(c("  │             SINCRONIZACIÓN DE RESULTADOS                  │", "cyan"))
    print(c("  └───────────────────────────────────────────────────────────┘", "cyan"))
    print()

    # 1. Obtener datos de la API
    print(c("  Consultando football-data.org...", "dim"))
    api_matches = fetch_finished_matches()
    if not api_matches:
        print(c("  No se obtuvieron partidos de la API.", "yellow"))
        return [], [], load_local_results()

    print(c(f"  [OK] {len(api_matches)} partidos terminados en la API", "green"))

    # 2. Cargar datos locales
    local = load_local_results()
    local_keys = {}
    for r in local:
        key = _match_key(r["home_team"], r["away_team"])
        local_keys[key] = r

    print(c(f"  [OK] {len(local)} partidos en archivo local", "dim"))

    # 3. Detectar nuevos y actualizados
    new_matches = []
    updated_matches = []

    for am in api_matches:
        key = _match_key(am["home_team"], am["away_team"])
        existing = local_keys.get(key)

        if existing is None:
            # Partido nuevo
            match_entry = {
                "home_team": am["home_team"],
                "away_team": am["away_team"],
                "home_goals": am["home_goals"],
                "away_goals": am["away_goals"],
                "group": am["group"],
                "date": am["date"],
                "stage": am.get("stage", "GROUP_STAGE"),
                "matchday": am.get("matchday"),
                "synced_at": datetime.now().isoformat(),
                "source": "football-data.org",
            }
            new_matches.append(match_entry)
        else:
            # Verificar si el resultado cambió (corrección de la API)
            if (existing["home_goals"] != am["home_goals"] or
                    existing["away_goals"] != am["away_goals"]):
                existing["home_goals"] = am["home_goals"]
                existing["away_goals"] = am["away_goals"]
                existing["updated_at"] = datetime.now().isoformat()
                updated_matches.append(existing)

    # 4. Guardar
    if not dry_run:
        all_results = local + new_matches
        save_local_results(all_results)
    else:
        all_results = local + new_matches

    # 5. Imprimir resumen
    _print_sync_summary(new_matches, updated_matches, len(api_matches), dry_run)

    return new_matches, updated_matches, all_results


def _print_sync_summary(new: List[Dict], updated: List[Dict],
                        total_api: int, dry_run: bool):
    """Imprime un resumen colorido de la sincronización."""
    print()

    if dry_run:
        print(c("  ── DRY RUN (no se guardaron cambios) ──", "yellow"))
        print()

    if not new and not updated:
        print(c("  Todo al día — no hay partidos nuevos.", "green"))
        print()
        return

    if new:
        label = "NUEVOS PARTIDOS ENCONTRADOS" if not dry_run else "PARTIDOS QUE SE AGREGARÍAN"
        print(c(f"  {len(new)} {label}:", "bold"))
        print()
        for m in new:
            hg = m["home_goals"]
            ag = m["away_goals"]
            home = m["home_team"]
            away = m["away_team"]
            group = m.get("group", "?")
            date = m.get("date", "?")

            if hg > ag:
                rc = "green"
                status_char = "+"
            elif ag > hg:
                rc = "red"
                status_char = "-"
            else:
                rc = "yellow"
                status_char = "="

            score_str = c(f"{hg}-{ag}", rc)
            print(f"    {status_char} [{group}] {date}  {home:20s} {score_str}  {away}")

        print()

    if updated:
        print(c(f"  {len(updated)} RESULTADOS ACTUALIZADOS:", "yellow"))
        for m in updated:
            print(f"    {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}")
        print()

    if not dry_run:
        total_local = len(load_local_results())
        print(c(f"  Archivo actualizado: {RESULTS_FILE.name} ({total_local} partidos)", "green"))
        print(c(f"  Fuente: football-data.org — {total_api} partidos en API", "dim"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MODO WATCH
# ═══════════════════════════════════════════════════════════════════════════════

def watch_mode(interval_min: int = 5, dry_run: bool = False):
    """Re-sincroniza cada N minutos."""
    print()
    print(c(f"  MODO WATCH — Sincronizando cada {interval_min} minutos", "bold"))
    print(c(f"     Presiona Ctrl+C para detener", "dim"))
    print()

    try:
        while True:
            now = datetime.now().strftime("%H:%M:%S")
            print(c(f"  ─── {now} ───────────────────────────────────────", "dim"))
            new, updated, _ = sync(dry_run=dry_run)

            if new:
                print(c(f"  ¡{len(new)} nuevos resultados detectados!", "green"))

            print(c(f"  Próxima revisión en {interval_min} minuto(s)...", "dim"))
            print()
            time.sleep(interval_min * 60)
    except KeyboardInterrupt:
        print()
        print(c("  Watch mode detenido.", "cyan"))
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TABLA DE POSICIONES RÁPIDA (post-sync)
# ═══════════════════════════════════════════════════════════════════════════════

def print_quick_standings(group: str):
    """Muestra una tabla de posiciones rápida basada en resultados reales."""
    from simulation.wc2026_fixtures import WC2026_GROUPS

    teams = WC2026_GROUPS.get(group, [])
    if not teams:
        print(c(f"  Grupo '{group}' no encontrado.", "red"))
        return

    results = load_local_results()
    teams_set = set(teams)

    # Filtrar resultados de este grupo
    group_results = [r for r in results
                     if r["home_team"] in teams_set and r["away_team"] in teams_set]

    # Calcular tabla
    table = {}
    for t in teams:
        table[t] = {"team": t, "pj": 0, "pg": 0, "pe": 0, "pp": 0,
                     "gf": 0, "ga": 0, "gd": 0, "pts": 0}

    for r in group_results:
        h = r["home_team"]
        a = r["away_team"]
        hg = r["home_goals"]
        ag = r["away_goals"]

        table[h]["pj"] += 1
        table[a]["pj"] += 1
        table[h]["gf"] += hg
        table[h]["ga"] += ag
        table[a]["gf"] += ag
        table[a]["ga"] += hg

        if hg > ag:
            table[h]["pg"] += 1
            table[h]["pts"] += 3
            table[a]["pp"] += 1
        elif ag > hg:
            table[a]["pg"] += 1
            table[a]["pts"] += 3
            table[h]["pp"] += 1
        else:
            table[h]["pe"] += 1
            table[a]["pe"] += 1
            table[h]["pts"] += 1
            table[a]["pts"] += 1

    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]

    standings = sorted(table.values(),
                       key=lambda x: (x["pts"], x["gd"], x["gf"]),
                       reverse=True)

    n_played = len(group_results)
    total_matches = 6  # C(4,2) = 6

    print()
    print(c(f"  ┌───────────────────────────────────────────────────────────┐", "cyan"))
    print(c(f"  │         TABLA GRUPO {group} — {n_played}/{total_matches} partidos jugados               │", "cyan"))
    print(c(f"  └───────────────────────────────────────────────────────────┘", "cyan"))
    print()

    header = f"  {'#':>2s}  {'Equipo':22s} {'PJ':>3s} {'PG':>3s} {'PE':>3s} {'PP':>3s} {'GF':>3s} {'GA':>3s} {'GD':>4s} {'Pts':>4s}"
    print(c(header, "bold"))
    print(f"  {'─'*2}  {'─'*22} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*4} {'─'*4}")

    for i, s in enumerate(standings):
        if i < 2:
            tc = "green"
            mark = "^"
        elif i == 2:
            tc = "yellow"
            mark = "o"
        else:
            tc = "red"
            mark = "v"

        gd_str = f"+{s['gd']}" if s["gd"] > 0 else str(s["gd"])

        line = (f"  {c(mark, tc)} {i+1}  "
                f"{c(s['team'], tc):>32s} "
                f"{s['pj']:>3d} {s['pg']:>3d} {s['pe']:>3d} {s['pp']:>3d} "
                f"{s['gf']:>3d} {s['ga']:>3d} {gd_str:>4s} "
                f"{c(str(s['pts']), 'bold'):>13s}")
        print(line)

    print()
    print(c("  ^ = Clasifica  o = 3ro (posible)  v = Eliminado", "dim"))
    if n_played < total_matches:
        print(c(f"  Faltan {total_matches - n_played} partido(s) por jugar", "yellow"))
    else:
        print(c(f"  Todos los partidos del grupo jugados", "green"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sincroniza resultados del Mundial 2026 desde football-data.org"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar partidos nuevos sin guardar")
    parser.add_argument("--watch", action="store_true",
                        help="Modo watch: re-sincroniza periódicamente")
    parser.add_argument("-i", "--interval", type=int, default=5,
                        help="Intervalo en minutos para modo watch (default: 5)")
    parser.add_argument("--tabla", type=str, default=None,
                        help="Mostrar tabla de un grupo después de sincronizar (A-L)")
    parser.add_argument("--all-tables", action="store_true",
                        help="Mostrar tablas de todos los grupos después de sincronizar")

    args = parser.parse_args()

    if args.watch:
        watch_mode(interval_min=args.interval, dry_run=args.dry_run)
    else:
        sync(dry_run=args.dry_run)

        if args.tabla:
            print_quick_standings(args.tabla.upper())

        if args.all_tables:
            for g in "ABCDEFGHIJKL":
                print_quick_standings(g)


if __name__ == "__main__":
    main()
