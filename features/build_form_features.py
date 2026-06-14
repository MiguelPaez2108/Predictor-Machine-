"""
build_form_features.py
Calcula la forma reciente de cada equipo con ventana ponderada exponencialmente.
Fórmula: Forma = Σ(resultado_i × λ^(N-i)) / Σ(λ^i),  λ=0.85, N=10
Salida: data/features/form_rolling.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_FEATURES, FORM_WINDOW, FORM_DECAY, ensure_dirs


def compute_form(team_history: pd.DataFrame, as_of_date: pd.Timestamp,
                 window: int = FORM_WINDOW, lam: float = FORM_DECAY) -> dict:
    """
    Dado el historial de un equipo hasta as_of_date (excluido),
    calcula la forma ponderada exponencialmente.
    """
    past = team_history[team_history["date"] < as_of_date].tail(window)
    if past.empty:
        return {
            "form_weighted":      np.nan,
            "form_pts_last6":     np.nan,
            "form_pts_last10":    np.nan,
            "form_gf_avg":        np.nan,
            "form_ga_avg":        np.nan,
            "form_gd_avg":        np.nan,
            "form_wins":          np.nan,
            "form_draws":         np.nan,
            "form_losses":        np.nan,
            "momentum_trend":     np.nan,
            "n_matches_form":     0,
        }

    N = len(past)
    weights = np.array([lam ** (N - 1 - i) for i in range(N)])
    weights /= weights.sum()

    results = past["result_numeric"].values
    form_weighted = float(np.dot(results, weights))

    # Últimas 6 y 10 para momentum
    last6  = past.tail(6)
    last10 = past.tail(10)
    pts6   = last6["pts"].mean() if len(last6) > 0 else np.nan
    pts10  = last10["pts"].mean() if len(last10) > 0 else np.nan
    momentum = float(pts6 - pts10) if not (np.isnan(pts6) or np.isnan(pts10)) else np.nan

    return {
        "form_weighted":   round(form_weighted, 4),
        "form_pts_last6":  round(float(pts6), 4) if not np.isnan(pts6) else np.nan,
        "form_pts_last10": round(float(pts10), 4) if not np.isnan(pts10) else np.nan,
        "form_gf_avg":     round(float(past["gf"].mean()), 4),
        "form_ga_avg":     round(float(past["ga"].mean()), 4),
        "form_gd_avg":     round(float(past["gd"].mean()), 4),
        "form_wins":       int((past["result_numeric"] == 1.0).sum()),
        "form_draws":      int((past["result_numeric"] == 0.5).sum()),
        "form_losses":     int((past["result_numeric"] == 0.0).sum()),
        "momentum_trend":  round(float(momentum), 4) if not np.isnan(momentum) else np.nan,
        "n_matches_form":  N,
    }


def build_form_features():
    ensure_dirs()

    print("Cargando international_results.parquet...")
    ir = pd.read_parquet(DATA_RAW / "international_results.parquet")
    ir["date"] = pd.to_datetime(ir["date"])
    ir = ir.sort_values("date").reset_index(drop=True)

    # Expand a formato largo: una fila por equipo × partido
    home = ir[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "neutral", "is_competitive"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "gf", "away_score": "ga"})
    home["is_home"] = True

    away = ir[["date", "away_team", "home_team", "away_score", "home_score",
               "tournament", "neutral", "is_competitive"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "gf", "home_score": "ga"})
    away["is_home"] = False

    long = pd.concat([home, away], ignore_index=True)
    long["gf"] = long["gf"].fillna(0)
    long["ga"] = long["ga"].fillna(0)
    long["gd"] = long["gf"] - long["ga"]

    # Resultado numérico: 1=victoria, 0.5=empate, 0=derrota
    long["result_numeric"] = np.where(long["gf"] > long["ga"], 1.0,
                             np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["pts"] = np.where(long["gf"] > long["ga"], 3.0,
                  np.where(long["gf"] == long["ga"], 1.0, 0.0))

    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    print(f"  Equipos únicos: {long['team'].nunique()}")
    print(f"  Partidos totales (largo): {len(long):,}")
    print("  Calculando forma rolling...")

    # Para cada fila, calcular forma con partidos ANTERIORES (sin leakage)
    records = []
    for team, group in long.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        for i, row in group.iterrows():
            form_stats = compute_form(group, as_of_date=row["date"])
            rec = {
                "date":        row["date"],
                "team":        team,
                "opponent":    row["opponent"],
                "tournament":  row["tournament"],
                "is_home":     row["is_home"],
                "neutral":     row["neutral"],
                "gf":          row["gf"],
                "ga":          row["ga"],
                "gd":          row["gd"],
                "pts":         row["pts"],
                "result":      row["result_numeric"],
                **form_stats
            }
            records.append(rec)

    form_df = pd.DataFrame(records)
    form_df = form_df.sort_values(["date", "team"]).reset_index(drop=True)

    out_path = DATA_FEATURES / "form_rolling.parquet"
    form_df.to_parquet(out_path, index=False)

    print(f"\n✓ form_rolling.parquet: {len(form_df):,} filas")
    print(f"  Equipos: {form_df['team'].nunique()}")
    print(f"  Forma weighted promedio: {form_df['form_weighted'].mean():.4f}")
    print(f"  Rango fechas: {form_df['date'].min().date()} → {form_df['date'].max().date()}")


if __name__ == "__main__":
    build_form_features()
