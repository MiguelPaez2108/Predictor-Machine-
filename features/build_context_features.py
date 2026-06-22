"""
build_context_features.py
Features de contexto por partido: descanso, fatiga, fase eliminatoria,
importancia del torneo, localía relativa.
Salida: data/features/context_features.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_FEATURES, ensure_dirs


# Pesos de importancia de torneos (para el índice de fatiga y contexto)
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":            1.0,
    "UEFA Euro":                 0.9,
    "Copa América":              0.85,
    "Copa America":              0.85,
    "Africa Cup of Nations":     0.80,
    "AFCON":                     0.80,
    "AFC Asian Cup":             0.75,
    "CONCACAF Gold Cup":         0.70,
    "UEFA Nations League":       0.65,
    "FIFA Confederations Cup":   0.85,
    "UEFA World Cup qualification": 0.55,
    "CONMEBOL World Cup qualification": 0.55,
    "Friendly":                  0.20,
}

# Torneos que cuentan como eliminatorias internacionales de alta intensidad
KNOCKOUT_TOURNAMENTS = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "Copa America",
    "Africa Cup of Nations", "AFCON", "AFC Asian Cup", "CONCACAF Gold Cup",
    "FIFA Confederations Cup",
}


def get_tournament_weight(tournament: str) -> float:
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament.lower():
            return weight
    return 0.35  # clasificatorias genéricas


def is_major_tournament(tournament: str) -> bool:
    for key in KNOCKOUT_TOURNAMENTS:
        if key.lower() in tournament.lower():
            return True
    return False


def build_context_features():
    ensure_dirs()

    print("Cargando international_results.parquet...")
    ir = pd.read_parquet(DATA_RAW / "international_results.parquet")
    ir["date"] = pd.to_datetime(ir["date"])
    ir = ir.sort_values("date").reset_index(drop=True)

    # Formato largo
    home = ir[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "city", "country", "neutral", "is_competitive"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "gf", "away_score": "ga"})
    home["is_home"] = True

    away = ir[["date", "away_team", "home_team", "away_score", "home_score",
               "tournament", "city", "country", "neutral", "is_competitive"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "gf", "home_score": "ga"})
    away["is_home"] = False

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Peso del torneo
    long["tournament_weight"] = long["tournament"].apply(get_tournament_weight)
    long["is_major"]          = long["tournament"].apply(is_major_tournament)

    print(f"  Calculando contexto para {len(long):,} registros...")

    records = []
    for team, group in long.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        dates = group["date"].values
        opponents = group["opponent"].values
        tournaments = group["tournament"].values
        countries = group["country"].values
        is_homes = group["is_home"].values
        neutrals = group["neutral"].values
        is_competitives = group["is_competitive"].values
        t_weights = group["tournament_weight"].values
        is_majors = group["is_major"].values

        n_rows = len(group)
        for i in range(n_rows):
            as_of = dates[i]

            # Descanso y fatiga
            if i == 0:
                days_rest = np.nan
                fatigue_index = 0.0
                matches_last_30 = 0
                matches_last_60 = 0
                matches_wc_cycle = 0
            else:
                last_match = dates[i-1]
                days_rest = float((as_of - last_match) / np.timedelta64(1, 'D'))

                # Índice de fatiga: Σ(peso_torneo × e^(-días/7)) de últimos 5 partidos
                start5 = max(0, i - 5)
                l5_dates = dates[start5:i]
                l5_weights = t_weights[start5:i]
                
                days_ago = (as_of - l5_dates) / np.timedelta64(1, 'D')
                decay = l5_weights * np.exp(-days_ago / 7)
                fatigue_index = round(float(decay.sum()), 4)

                # Partidos en últimos 30, 60 y 730 días (2 años)
                since_30 = as_of - np.timedelta64(30, 'D')
                since_60 = as_of - np.timedelta64(60, 'D')
                since_2y = as_of - np.timedelta64(730, 'D')

                idx_30 = np.searchsorted(dates[:i], since_30)
                idx_60 = np.searchsorted(dates[:i], since_60)
                idx_2y = np.searchsorted(dates[:i], since_2y)

                matches_last_30 = int(i - idx_30)
                matches_last_60 = int(i - idx_60)
                matches_wc_cycle = int(i - idx_2y)

            # Contexto del torneo en curso (dentro de una ventana de 50 días para evitar leakage entre ediciones)
            current_tourn = tournaments[i]
            since_50d = as_of - np.timedelta64(50, 'D')
            
            # Buscar el índice inicial de la ventana de 50 días en dates[:i]
            idx_50d = np.searchsorted(dates[:i], since_50d)
            matches_in_tournament = int((tournaments[idx_50d:i] == current_tourn).sum())

            # Inferir fase
            if matches_in_tournament == 0:
                phase = "group_1"
            elif matches_in_tournament <= 2:
                phase = f"group_{matches_in_tournament + 1}"
            elif matches_in_tournament == 3:
                phase = "round_of_16"
            elif matches_in_tournament == 4:
                phase = "quarter_final"
            elif matches_in_tournament == 5:
                phase = "semi_final"
            else:
                phase = "final"

            is_knockout_phase = matches_in_tournament >= 3

            records.append({
                "date":                 pd.Timestamp(as_of),
                "team":                 team,
                "opponent":             opponents[i],
                "tournament":           tournaments[i],
                "country_host":         countries[i],
                "is_home":              is_homes[i],
                "neutral":              neutrals[i],
                "is_competitive":       is_competitives[i],
                "tournament_weight":    t_weights[i],
                "is_major_tournament":  is_majors[i],
                "days_rest":            days_rest,
                "fatigue_index":        fatigue_index,
                "matches_last_30d":     matches_last_30,
                "matches_last_60d":     matches_last_60,
                "matches_wc_cycle":     matches_wc_cycle,
                "matches_in_tournament":matches_in_tournament,
                "phase":                phase,
                "is_knockout_phase":    is_knockout_phase,
            })

    ctx_df = pd.DataFrame(records)
    ctx_df = ctx_df.sort_values(["date", "team"]).reset_index(drop=True)

    out_path = DATA_FEATURES / "context_features.parquet"
    ctx_df.to_parquet(out_path, index=False)

    print(f"\n[OK] context_features.parquet: {len(ctx_df):,} filas")
    print(f"  Descanso promedio: {ctx_df['days_rest'].mean():.1f} días")
    print(f"  Fatiga index promedio: {ctx_df['fatigue_index'].mean():.3f}")
    print(f"  Fases: {ctx_df['phase'].value_counts().to_dict()}")


if __name__ == "__main__":
    build_context_features()
