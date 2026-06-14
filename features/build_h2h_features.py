"""
build_h2h_features.py
Calcula estadísticas head-to-head entre cada par de equipos (últimos 5 años).
Salida: data/features/h2h_history.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_FEATURES, ensure_dirs


H2H_YEARS = 10  # ventana de años para H2H


def build_h2h_features():
    ensure_dirs()

    print("Cargando international_results.parquet...")
    ir = pd.read_parquet(DATA_RAW / "international_results.parquet")
    ir["date"] = pd.to_datetime(ir["date"])
    ir = ir.sort_values("date").reset_index(drop=True)

    # Formato largo: una fila por (team, opponent, date) en cada partido
    home = ir[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "gf", "away_score": "ga"})
    home["is_home"] = True

    away = ir[["date", "away_team", "home_team", "away_score", "home_score", "neutral"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "gf", "home_score": "ga"})
    away["is_home"] = False

    long = pd.concat([home, away], ignore_index=True)
    long["gf"] = long["gf"].fillna(0)
    long["ga"] = long["ga"].fillna(0)
    long["win"]  = (long["gf"] > long["ga"]).astype(int)
    long["draw"] = (long["gf"] == long["ga"]).astype(int)
    long["loss"] = (long["gf"] < long["ga"]).astype(int)
    long = long.sort_values(["team", "opponent", "date"]).reset_index(drop=True)

    print(f"  Total enfrentamientos: {len(long):,}")
    print("  Calculando H2H rolling...")

    records = []
    cutoff_delta = pd.DateOffset(years=H2H_YEARS)

    for (team, opponent), group in long.groupby(["team", "opponent"]):
        group = group.sort_values("date").reset_index(drop=True)

        for i, row in group.iterrows():
            as_of = row["date"]
            cutoff = as_of - cutoff_delta
            past = group[(group["date"] < as_of) & (group["date"] >= cutoff)]

            if past.empty:
                h2h = {
                    "h2h_matches":     0,
                    "h2h_wins":        0,
                    "h2h_draws":       0,
                    "h2h_losses":      0,
                    "h2h_win_rate":    np.nan,
                    "h2h_gf_avg":      np.nan,
                    "h2h_ga_avg":      np.nan,
                    "h2h_gd_avg":      np.nan,
                    "h2h_pts_avg":     np.nan,
                }
            else:
                n = len(past)
                wins   = int(past["win"].sum())
                draws  = int(past["draw"].sum())
                losses = int(past["loss"].sum())
                pts_avg = (wins * 3 + draws * 1) / n
                h2h = {
                    "h2h_matches":  n,
                    "h2h_wins":     wins,
                    "h2h_draws":    draws,
                    "h2h_losses":   losses,
                    "h2h_win_rate": round(wins / n, 4),
                    "h2h_gf_avg":   round(float(past["gf"].mean()), 4),
                    "h2h_ga_avg":   round(float(past["ga"].mean()), 4),
                    "h2h_gd_avg":   round(float((past["gf"] - past["ga"]).mean()), 4),
                    "h2h_pts_avg":  round(float(pts_avg), 4),
                }

            records.append({
                "date":     as_of,
                "team":     team,
                "opponent": opponent,
                **h2h,
            })

    h2h_df = pd.DataFrame(records)
    h2h_df = h2h_df.sort_values(["date", "team"]).reset_index(drop=True)

    out_path = DATA_FEATURES / "h2h_history.parquet"
    h2h_df.to_parquet(out_path, index=False)

    print(f"\n✓ h2h_history.parquet: {len(h2h_df):,} filas")
    print(f"  Pares únicos team×opp: {h2h_df.groupby(['team','opponent']).ngroups:,}")
    print(f"  H2H matches promedio (cuando hay historial): "
          f"{h2h_df[h2h_df['h2h_matches']>0]['h2h_matches'].mean():.1f}")


if __name__ == "__main__":
    build_h2h_features()
