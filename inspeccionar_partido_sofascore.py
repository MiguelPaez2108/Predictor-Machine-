"""
inspeccionar_partido_sofascore.py — Inspecciona los tiros de UN partido
específico en match_shots.parquet, para diagnosticar por qué match_goals
detecta más goles que el marcador oficial en ese partido.

Uso:
  python inspeccionar_partido_sofascore.py --match-id 15186751
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW

OUT_DIR = DATA_RAW / "sofascore_wc2026"


def main(match_id: int):
    shots = pd.read_parquet(OUT_DIR / "match_shots.parquet")
    idx   = pd.read_parquet(OUT_DIR / "matches_index.parquet")

    info = idx[idx["sofascore_match_id"] == match_id]
    if not info.empty:
        row = info.iloc[0]
        print(f"Partido: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']}"
              f"  (id={match_id})")
    else:
        print(f"[AVISO] match_id={match_id} no está en matches_index.")

    m = shots[shots["match_id"] == match_id].copy()
    print(f"\nTotal filas en match_shots para este match_id: {len(m)}")

    if "id" in m.columns:
        n_unique_ids = m["id"].nunique()
        print(f"IDs de tiro únicos: {n_unique_ids}  "
              f"({'[OK] sin duplicados' if n_unique_ids == len(m) else '[!] HAY FILAS DUPLICADAS'})")

        dup_ids = m[m.duplicated(subset=["id"], keep=False)]
        if not dup_ids.empty:
            print(f"\n[!] {len(dup_ids)} filas con 'id' de tiro repetido — esto confirmaría duplicación:")
            cols = [c for c in ["id", "player", "isHome", "time", "addedTime",
                                 "goalType", "incidentType", "xg"] if c in dup_ids.columns]
            print(dup_ids[cols].sort_values("id").to_string(index=False))
    else:
        print("[AVISO] No hay columna 'id' en match_shots — no se puede chequear duplicados por id.")
        print("        Revisando duplicados por (player, time, xg) en su lugar:")
        check_cols = [c for c in ["player", "time", "xg", "isHome"] if c in m.columns]
        if check_cols:
            dup_rows = m[m.duplicated(subset=check_cols, keep=False)]
            if not dup_rows.empty:
                print(f"  [!] {len(dup_rows)} filas duplicadas por {check_cols}")
                print(dup_rows[check_cols].sort_values(check_cols).to_string(index=False))

    # Mismo criterio que extract_goals() en el script principal
    mask = pd.Series(False, index=m.index)
    if "goalType" in m.columns:
        mask = mask | m["goalType"].notna()
    if "incidentType" in m.columns:
        mask = mask | (m["incidentType"].astype(str).str.lower() == "goal")

    goals = m[mask].copy()
    print(f"\nFilas detectadas como gol (goalType notna OR incidentType=='goal'): {len(goals)}")

    cols = [c for c in ["id", "player", "isHome", "time", "addedTime",
                         "goalType", "incidentType", "xg", "xgot", "shotType", "situation"]
            if c in goals.columns]
    if "time" in goals.columns:
        goals = goals.sort_values("time")
    print(goals[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", type=int, default=15186751,
                        help="match_id de Sofascore a inspeccionar (default: Austria vs Jordan)")
    args = parser.parse_args()
    main(args.match_id)