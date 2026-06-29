"""
sync_resultados_from_sofascore.py — Sincroniza resultados_reales.json con
los resultados oficiales scrapeados de Sofascore
(data/raw/sofascore_wc2026/matches_index.parquet).

Por qué hace falta:
  resultados_reales.json se venía llenando manualmente / vía
  ingestion/sync_results.py (football-data.org), y quedó desactualizado a
  mitad de la fase de grupos (~41 partidos). Sofascore ya tiene los 72
  partidos de grupos confirmados. Mientras no se sincronice esto,
  simulation/group_stage.py sigue SIMULANDO partidos que ya se jugaron
  en la realidad (load_real_results() en ese módulo lee justamente
  resultados_reales.json para decidir qué partidos no simular).

Qué hace:
  - Lee matches_index.parquet (todos los partidos con resultado, grupos
    o eliminatoria, que Sofascore ya tenga confirmados).
  - Para cada partido: si NO existe en resultados_reales.json, lo agrega.
    Si existe pero el marcador difiere, lo corrige.
  - No borra ni toca entradas que no tengan equivalente en Sofascore
    (ej. resultados cargados a mano para partidos que Sofascore no cubra).

Uso:
  python ingestion/sync_resultados_from_sofascore.py             # sincroniza
  python ingestion/sync_resultados_from_sofascore.py --dry-run   # solo mostrar, no guarda
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, RESULTS_FILE

SOFASCORE_DIR = DATA_RAW / "sofascore_wc2026"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_group(stage: str) -> str:
    """'FIFA World Cup, Group I' -> 'I'. Si no matchea, devuelve '?'."""
    m = re.search(r"Group\s+([A-L])\b", str(stage))
    return m.group(1) if m else "?"


def load_local_results() -> list:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_local_results(results: list):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _match_key(home: str, away: str) -> str:
    """Clave única por partido, insensible al orden local/visitante."""
    return "|".join(sorted([home, away]))


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de entradas desde Sofascore
# ─────────────────────────────────────────────────────────────────────────────

def build_sofascore_entries(idx_df: pd.DataFrame) -> list:
    entries = []
    for _, r in idx_df.iterrows():
        if pd.isna(r.get("home_goals")) or pd.isna(r.get("away_goals")):
            continue  # partido sin resultado todavía (eliminatoria futura, etc.)

        stage_raw = str(r.get("stage", ""))
        entries.append({
            "home_team"          : r["home_team"],
            "away_team"          : r["away_team"],
            "home_goals"         : int(r["home_goals"]),
            "away_goals"         : int(r["away_goals"]),
            "group"              : parse_group(stage_raw),
            "date"               : r.get("match_date"),
            "stage"              : "GROUP_STAGE" if "Group" in stage_raw else "KNOCKOUT",
            "sofascore_match_id" : int(r["sofascore_match_id"]),
            "synced_at"          : datetime.now().isoformat(),
            "source"             : "sofascore",
        })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Sincronización
# ─────────────────────────────────────────────────────────────────────────────

def sync(dry_run: bool = False):
    idx_path = SOFASCORE_DIR / "matches_index.parquet"
    if not idx_path.exists():
        print(f"ERROR: no existe {idx_path}.")
        print("  Correr primero: python ingestion/08_scrape_sofascore_wc2026.py")
        return

    idx_df = pd.read_parquet(idx_path)
    sofa_entries = build_sofascore_entries(idx_df)
    print(f"Partidos con resultado confirmado en Sofascore: {len(sofa_entries)}")

    local = load_local_results()
    local_by_key = {_match_key(r["home_team"], r["away_team"]): r for r in local}
    print(f"Partidos ya en resultados_reales.json:          {len(local)}")
    print()

    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for se in sofa_entries:
        key = _match_key(se["home_team"], se["away_team"])
        existing = local_by_key.get(key)

        if existing is None:
            print(f"  [NUEVO]      {se['home_team']:22s} {se['home_goals']}-{se['away_goals']} "
                  f"{se['away_team']:22s} (Grupo {se['group']})")
            if not dry_run:
                local.append(se)
                local_by_key[key] = se
            new_count += 1

        elif (existing.get("home_goals") != se["home_goals"] or
              existing.get("away_goals") != se["away_goals"]):
            # El marcador difiere -> corregir con el dato de Sofascore.
            # Atención al orden home/away: si en el JSON local el partido
            # quedó guardado con home/away invertido respecto a Sofascore,
            # comparamos también la versión invertida antes de pisar nada.
            same_orientation = existing.get("home_team") == se["home_team"]
            old_score = f"{existing.get('home_goals')}-{existing.get('away_goals')}"

            if same_orientation:
                new_score = f"{se['home_goals']}-{se['away_goals']}"
            else:
                # orientación invertida: comparar y guardar invertido también
                new_score = f"{se['away_goals']}-{se['home_goals']}"
                if old_score == new_score:
                    unchanged_count += 1
                    continue

            print(f"  [CORREGIDO]  {se['home_team']:22s} {old_score} -> {new_score} "
                  f"{se['away_team']:22s}")
            if not dry_run:
                if same_orientation:
                    existing["home_goals"] = se["home_goals"]
                    existing["away_goals"] = se["away_goals"]
                else:
                    existing["home_goals"] = se["away_goals"]
                    existing["away_goals"] = se["home_goals"]
                existing["updated_at"] = datetime.now().isoformat()
                existing["source"] = "sofascore"
            updated_count += 1
        else:
            unchanged_count += 1

    print()
    print(f"Nuevos: {new_count}  |  Corregidos: {updated_count}  |  Sin cambios: {unchanged_count}")

    if dry_run:
        print("\n[DRY RUN] No se guardó nada en disco.")
        return

    save_local_results(local)
    print(f"\n[OK] resultados_reales.json actualizado: {len(local)} partidos totales")
    print(f"     Guardado en: {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sincroniza resultados_reales.json con los partidos confirmados por Sofascore"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar qué cambiaría, sin guardar")
    args = parser.parse_args()
    sync(dry_run=args.dry_run)