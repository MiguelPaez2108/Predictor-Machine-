"""
update_r32_results.py — Actualización rápida de resultados del R32
====================================================================
Agrega los resultados ya conocidos del R32 y cualquier resultado nuevo
que pases por línea de comandos, luego opcionalmente re-corre la simulación.

Uso:
  python update_r32_results.py                          # agrega resultados hardcodeados
  python update_r32_results.py --add "France 2-0 Sweden"
  python update_r32_results.py --resim                  # re-corre Monte Carlo después
  python update_r32_results.py --sync-api               # pull desde football-data.org
  python update_r32_results.py --list                   # ver todos los resultados R32
"""

import sys
import json
import re
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

ROOT         = Path(__file__).parent
RESULTS_FILE = ROOT / "data" / "resultados_reales.json"

# ─── Colores ──────────────────────────────────────────────────────────────────
def c(text, code):
    codes = {
        "green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
        "cyan":"\033[96m","bold":"\033[1m","dim":"\033[2m","reset":"\033[0m",
    }
    return codes.get(code, "") + text + codes["reset"]


# ─── Resultados ya conocidos del R32 ─────────────────────────────────────────
# Actualizar aquí a medida que se juegan los partidos
R32_KNOWN_RESULTS = [
    # (home, away, home_goals, away_goals, date)
    ("Brazil",       "Japan",        2, 1, "2026-06-29"),
    ("South Africa", "Canada",       0, 1, "2026-06-29"),
    # Agregar más aquí cuando se jueguen:
    # ("Germany",      "Paraguay",     X, Y, "2026-XX-XX"),
    # ("France",       "Sweden",       X, Y, "2026-XX-XX"),
]

# Mapeo de nombres alternativos → canónicos del proyecto
NAME_MAP = {
    "brasil": "Brazil", "japan": "Japan", "japón": "Japan",
    "sudáfrica": "South Africa", "sudafrica": "South Africa",
    "canadá": "Canada", "canada": "Canada",
    "alemania": "Germany", "germany": "Germany",
    "paraguay": "Paraguay",
    "francia": "France", "france": "France",
    "suecia": "Sweden", "sweden": "Sweden",
    "países bajos": "Netherlands", "netherlands": "Netherlands",
    "marruecos": "Morocco", "morocco": "Morocco",
    "portugal": "Portugal",
    "croacia": "Croatia", "croatia": "Croatia",
    "españa": "Spain", "espana": "Spain", "spain": "Spain",
    "austria": "Austria",
    "estados unidos": "United States", "usa": "United States",
    "united states": "United States",
    "bosnia": "Bosnia and Herzegovina",
    "bélgica": "Belgium", "belgium": "Belgium",
    "senegal": "Senegal",
    "costa de marfil": "Ivory Coast", "ivory coast": "Ivory Coast",
    "noruega": "Norway", "norway": "Norway",
    "méxico": "Mexico", "mexico": "Mexico",
    "ecuador": "Ecuador",
    "inglaterra": "England", "england": "England",
    "dr congo": "DR Congo", "república democrática del congo": "DR Congo",
    "argentina": "Argentina",
    "cabo verde": "Cape Verde", "cape verde": "Cape Verde",
    "australia": "Australia",
    "egipto": "Egypt", "egypt": "Egypt",
    "suiza": "Switzerland", "switzerland": "Switzerland",
    "argelia": "Algeria", "algeria": "Algeria",
    "colombia": "Colombia",
    "ghana": "Ghana",
    "norway": "Norway",
    "ivory coast": "Ivory Coast",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    key = name.strip().lower()
    return NAME_MAP.get(key, name.strip())


def load_results() -> List[Dict]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results: List[Dict]):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def match_key(h, a):
    return "|".join(sorted([h.lower(), a.lower()]))


def parse_result(raw: str) -> Optional[Tuple[str, int, str, int]]:
    raw = raw.strip()
    m = re.match(r'^(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+)$', raw)
    if m:
        return normalize(m.group(1)), int(m.group(2)), normalize(m.group(4)), int(m.group(3))
    parts = raw.split()
    nums = [(i, int(p)) for i, p in enumerate(parts) if p.isdigit()]
    if len(nums) >= 2:
        i1, n1 = nums[0]; i2, n2 = nums[1]
        t1 = normalize(" ".join(parts[:i1]))
        t2 = normalize(" ".join(parts[i1+1:i2]))
        if t1 and t2:
            return t1, n1, t2, n2
    return None


# ─── Agregar resultado ────────────────────────────────────────────────────────
def add_result(home: str, away: str, hg: int, ag: int,
               stage: str = "R32", date: str = None,
               silent: bool = False) -> str:
    results = load_results()
    key = match_key(home, away)
    existing = {match_key(r["home_team"], r["away_team"]): i
                for i, r in enumerate(results)}

    entry = {
        "home_team":  home,
        "away_team":  away,
        "home_goals": hg,
        "away_goals": ag,
        "stage":      stage,
        "group":      stage,
        "date":       date or datetime.now().strftime("%Y-%m-%d"),
        "synced_at":  datetime.now().isoformat(),
        "source":     "manual_r32",
    }

    if key in existing:
        old = results[existing[key]]
        if old["home_goals"] == hg and old["away_goals"] == ag:
            if not silent:
                print(c(f"  [YA EXISTE] {home} {hg}-{ag} {away}", "dim"))
            return "unchanged"
        results[existing[key]] = entry
        status = "updated"
    else:
        results.append(entry)
        status = "created"

    save_results(results)

    if not silent:
        winner = home if hg > ag else (away if ag > hg else "Empate")
        color = "green" if hg != ag else "yellow"
        print(c(f"  [{status.upper()}] {home} {hg}-{ag} {away}  → {winner}", color))

    return status


# ─── Sync desde football-data.org ────────────────────────────────────────────
def sync_from_api(dry_run: bool = False) -> int:
    """Pull de resultados desde football-data.org (grupos + eliminatoria)."""
    try:
        from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL, FOOTBALL_DATA_WC_CODE
    except ImportError:
        print(c("  [ERROR] config.py no encontrado", "red"))
        return 0

    if not FOOTBALL_DATA_API_KEY:
        print(c("  [ERROR] FOOTBALL_DATA_API_KEY no configurada en .env", "red"))
        return 0

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{FOOTBALL_DATA_WC_CODE}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    stage_map = {
        "LAST_32": "R32", "ROUND_OF_32": "R32",
        "LAST_16": "R16", "ROUND_OF_16": "R16",
        "QUARTER_FINALS": "QF", "SEMI_FINALS": "SF",
        "FINAL": "F", "THIRD_PLACE": "3P",
        "GROUP_STAGE": "GROUP_STAGE",
    }
    api_name_map = {
        "Türkiye": "Turkey", "Czechia": "Czech Republic",
        "Korea Republic": "South Korea", "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast", "Congo DR": "DR Congo",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Cape Verde Islands": "Cape Verde",
    }

    try:
        resp = requests.get(url, params={"status": "FINISHED"},
                           headers=headers, timeout=15)
        if resp.status_code != 200:
            print(c(f"  [ERROR] API HTTP {resp.status_code}", "red"))
            return 0

        matches = resp.json().get("matches", [])
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(c(f"  [API] {len(matches)} partidos terminados  |  requests restantes: {remaining}", "cyan"))

        n_new = 0
        results = load_results()
        existing_keys = {match_key(r["home_team"], r["away_team"]) for r in results}

        for m in matches:
            home_raw = m["homeTeam"]["name"]
            away_raw = m["awayTeam"]["name"]
            home = api_name_map.get(home_raw, home_raw)
            away = api_name_map.get(away_raw, away_raw)

            ft = m["score"].get("fullTime", {})
            hg, ag = ft.get("home"), ft.get("away")
            if hg is None or ag is None:
                continue

            stage_api = m.get("stage", "GROUP_STAGE")
            stage = stage_map.get(stage_api, stage_api)
            date = m.get("utcDate", "")[:10]
            key = match_key(home, away)

            if key not in existing_keys:
                if dry_run:
                    print(c(f"  [DRY] [{stage}] {home} {hg}-{ag} {away}", "yellow"))
                else:
                    add_result(home, away, int(hg), int(ag), stage=stage,
                               date=date, silent=False)
                    existing_keys.add(key)
                n_new += 1

        return n_new

    except Exception as e:
        print(c(f"  [ERROR] {e}", "red"))
        return 0


# ─── Listar resultados ────────────────────────────────────────────────────────
def list_r32_results():
    results = load_results()
    r32 = [r for r in results if r.get("stage") in ("R32",) or
           (r.get("group") in ("R32",))]

    print(c(f"\n  Resultados R32 registrados: {len(r32)}/16\n", "bold"))

    from simulation.knockout_stage import WC2026_R32_REAL_BRACKET
    for i, (home, away) in enumerate(WC2026_R32_REAL_BRACKET, 1):
        key = match_key(home, away)
        found = None
        for r in results:
            if match_key(r["home_team"], r["away_team"]) == key:
                found = r
                break

        side = "IZQ" if i <= 8 else "DER"
        if found:
            hg, ag = found["home_goals"], found["away_goals"]
            winner = found["home_team"] if hg > ag else found["away_team"]
            col = "green"
            score_str = c(f"{hg}-{ag}", col)
            print(f"  {i:>2d}. [{side}] {home:22s} {score_str}  {away:22s}  → {c(winner, 'green')}")
        else:
            print(c(f"  {i:>2d}. [{side}] {home:22s} vs  {away:22s}  (pendiente)", "dim"))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Actualización de resultados R32 WC2026"
    )
    parser.add_argument("--add", type=str, default=None,
                        help="Agregar resultado: 'France 2-0 Sweden'")
    parser.add_argument("--stage", type=str, default="R32",
                        help="Etapa (R32, R16, QF, SF, F). Default: R32")
    parser.add_argument("--date", type=str, default=None,
                        help="Fecha YYYY-MM-DD")
    parser.add_argument("--sync-api", action="store_true",
                        help="Sincronizar desde football-data.org")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar sin guardar")
    parser.add_argument("--list", action="store_true",
                        help="Mostrar resultados R32")
    parser.add_argument("--resim", action="store_true",
                        help="Re-correr simulación Monte Carlo (10k runs) después")
    parser.add_argument("--sims", type=int, default=10_000,
                        help="Número de simulaciones si --resim (default: 10000)")
    args = parser.parse_args()

    print(c("\n  WC2026 — ACTUALIZACIÓN RESULTADOS R32", "bold"))
    print(c("  " + "─"*42 + "\n", "cyan"))

    if args.list:
        list_r32_results()
        return

    if args.sync_api:
        n = sync_from_api(dry_run=args.dry_run)
        print(c(f"\n  Total nuevos: {n}", "green" if n > 0 else "dim"))
        if n > 0 and args.resim:
            _run_resim(args.sims)
        return

    if args.add:
        parsed = parse_result(args.add)
        if not parsed:
            print(c(f"  [ERROR] No se pudo parsear: '{args.add}'", "red"))
            print(c("  Formato: 'France 2-0 Sweden' o 'France 2 Sweden 0'", "dim"))
            return
        home, hg, away, ag = parsed
        if not args.dry_run:
            add_result(home, away, hg, ag,
                      stage=args.stage.upper(),
                      date=args.date)
        else:
            print(c(f"  [DRY] {home} {hg}-{ag} {away} [{args.stage}]", "yellow"))
        if args.resim:
            _run_resim(args.sims)
        return

    # Sin argumentos: aplicar los resultados conocidos hardcodeados
    print(c("  Aplicando resultados conocidos del R32...\n", "cyan"))
    n_updated = 0
    for home, away, hg, ag, date in R32_KNOWN_RESULTS:
        status = add_result(home, away, hg, ag, stage="R32", date=date)
        if status in ("created", "updated"):
            n_updated += 1

    print()
    if n_updated > 0:
        print(c(f"  [OK] {n_updated} resultado(s) nuevo(s) agregados/actualizados", "green"))
        if args.resim:
            _run_resim(args.sims)
    else:
        print(c("  Todos los resultados ya estaban registrados.", "dim"))

    print()
    list_r32_results()


def _run_resim(n_sims: int):
    """Re-corre la simulación Monte Carlo con los nuevos resultados."""
    print(c(f"\n  Re-corriendo simulación ({n_sims:,} runs)...", "cyan"))
    try:
        import simulation.match_simulator as ms
        ms._LOADED_MODELS = None
        ms._LAMBDA_CACHE.clear()
        ms._MATCH_PROB_CACHE.clear()
        ms._LAMBDA_CACHE_CAL.clear()
        ms._STATIC_DATA = None

        from simulation.tournament import simulate_wc2026, print_champion_table, save_results as save_sim
        results = simulate_wc2026(n_sims=n_sims, seed=42, verbose=True,
                                  use_real_bracket=True)
        if results:
            print_champion_table(results, top_n=16)
            save_sim(results)
            print(c("  [OK] Simulación actualizada y guardada", "green"))
    except Exception as e:
        print(c(f"  [ERROR] No se pudo re-simular: {e}", "red"))
        print(c("  Correr manualmente: python run_final.py --sims 10000", "dim"))


if __name__ == "__main__":
    main()
