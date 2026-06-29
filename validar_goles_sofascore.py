"""
validar_goles_sofascore.py — Cruza match_goals.parquet contra el marcador
oficial de matches_index.parquet, partido por partido.

Sofascore puede dejar goles ANULADOS (VAR, offside) marcados con goalType
en el shot map, aunque no cuenten en el marcador final. Este script detecta
exactamente en qué partidos pasa eso, comparando:
  goles_detectados (filas en match_goals.parquet por match_id)
  vs
  goles_oficiales (home_goals + away_goals en matches_index.parquet)

Uso:
  python validar_goles_sofascore.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW

OUT_DIR = DATA_RAW / "sofascore_wc2026"


def main():
    idx_path   = OUT_DIR / "matches_index.parquet"
    goals_path = OUT_DIR / "match_goals.parquet"

    if not idx_path.exists() or not goals_path.exists():
        print(f"ERROR: faltan archivos en {OUT_DIR}")
        return

    idx   = pd.read_parquet(idx_path)
    goals = pd.read_parquet(goals_path)

    print(f"Partidos en matches_index: {len(idx)}")
    print(f"Filas en match_goals:      {len(goals)}")

    total_oficial = int((idx["home_goals"].fillna(0) + idx["away_goals"].fillna(0)).sum())
    print(f"\nGoles totales según marcador final : {total_oficial}")
    print(f"Goles totales según match_goals     : {len(goals)}")
    print(f"Diferencia                          : {len(goals) - total_oficial}")

    # Conteo de goles detectados por partido
    goals_per_match = goals.groupby("match_id").size().rename("goals_detected")

    idx2 = idx.set_index("sofascore_match_id").copy()
    idx2["goals_official"] = (idx2["home_goals"].fillna(0) + idx2["away_goals"].fillna(0)).astype(int)
    idx2 = idx2.join(goals_per_match, how="left")
    idx2["goals_detected"] = idx2["goals_detected"].fillna(0).astype(int)

    mismatches = idx2[idx2["goals_official"] != idx2["goals_detected"]]

    if mismatches.empty:
        print("\n[OK] Todos los partidos matchean goles oficiales == goles detectados.")
    else:
        print(f"\n[AVISO] {len(mismatches)} partido(s) con diferencia "
              f"(probable gol anulado por VAR/offside aún tageado como goalType):")
        for mid, row in mismatches.iterrows():
            diff = row["goals_detected"] - row["goals_official"]
            signo = "+" if diff > 0 else ""
            print(f"  id={mid}  {row['home_team']} {row['home_goals']}-{row['away_goals']} "
                  f"{row['away_team']}   oficial={row['goals_official']}  "
                  f"detectados={row['goals_detected']}  ({signo}{diff})")

        print("\n  Para esos match_id, revisá match_shots.parquet filtrando por ese")
        print("  match_id y goalType notna — el/los tiro(s) extra probablemente tengan")
        print("  algo distintivo (mirá columnas no vistas en el diagnóstico, ej. un flag")
        print("  de offside/disallowed) que se pueda usar para excluirlos en extract_goals().")


if __name__ == "__main__":
    main()