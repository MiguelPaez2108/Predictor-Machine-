"""
build_elo_features.py
Construye el snapshot de ELO por equipo × partido desde el histórico calculado.
También extrae ELO diferencial y probabilidad esperada de victoria.
Salida: data/features/team_snapshot.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_FEATURES, ensure_dirs


def build_elo_features():
    ensure_dirs()

    print("Cargando elo_historical.parquet...")
    elo = pd.read_parquet(DATA_RAW / "elo_historical.parquet")
    elo["date"] = pd.to_datetime(elo["date"])
    elo = elo.sort_values("date").reset_index(drop=True)

    print("Cargando international_results.parquet...")
    ir = pd.read_parquet(DATA_RAW / "international_results.parquet")
    ir["date"] = pd.to_datetime(ir["date"])

    print(f"  ELO histórico: {len(elo):,} partidos | IR: {len(ir):,} partidos")

    # ── 1. Construir tabla ELO actual por equipo (última entrada por equipo)
    # Para cada equipo, su ELO post-partido más reciente es su ELO actual
    elo_home = elo[["date", "home_team", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_post": "elo_post"}
    )
    elo_away = elo[["date", "away_team", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_post": "elo_post"}
    )
    elo_long = pd.concat([elo_home, elo_away], ignore_index=True)
    elo_long = elo_long.sort_values(["team", "date"])

    # ELO actual = último valor registrado por equipo
    elo_current = (
        elo_long.groupby("team")["elo_post"]
        .last()
        .reset_index()
        .rename(columns={"elo_post": "elo_current"})
    )

    # ── 2. Para cada partido en international_results, obtener ELO pre-partido
    # Usamos elo_historical que ya tiene elo_home_pre y elo_away_pre
    # Construimos snapshot por equipo × partido (desde ambos lados)
    snap_home = elo[["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre",
                       "expected_home", "tournament"]].copy()
    snap_home = snap_home.rename(columns={
        "home_team":    "team",
        "away_team":    "opponent",
        "elo_home_pre": "elo_pre",
        "elo_away_pre": "opp_elo_pre",
        "expected_home":"expected_win",
    })
    snap_home["is_home"] = True

    snap_away = elo[["date", "away_team", "home_team", "elo_away_pre", "elo_home_pre",
                       "expected_home", "tournament"]].copy()
    snap_away["expected_away"] = 1 - snap_away["expected_home"]
    snap_away = snap_away.rename(columns={
        "away_team":    "team",
        "home_team":    "opponent",
        "elo_away_pre": "elo_pre",
        "elo_home_pre": "opp_elo_pre",
        "expected_away":"expected_win",
    })
    snap_away = snap_away.drop(columns=["expected_home"])
    snap_away["is_home"] = False

    snapshot = pd.concat([snap_home, snap_away], ignore_index=True)
    snapshot["elo_diff"] = snapshot["elo_pre"] - snapshot["opp_elo_pre"]
    snapshot["date"] = pd.to_datetime(snapshot["date"])
    snapshot = snapshot.sort_values(["team", "date"]).reset_index(drop=True)

    # ── 3. Rolling ELO stats: media móvil de ELO últimos 365 días
    def rolling_elo_mean(group, window_days=365):
        group = group.sort_values("date")
        group["elo_rolling_1y"] = (
            group.set_index("date")["elo_pre"]
            .rolling(f"{window_days}D", min_periods=3)
            .mean()
            .values
        )
        return group

    print("  Calculando ELO rolling (1 año)...")
    snapshot = snapshot.groupby("team", group_keys=False).apply(rolling_elo_mean)

    # ── 4. Añadir FIFA rankings y valor de mercado al snapshot actual
    fifa = pd.read_parquet(DATA_RAW / "fifa_rankings.parquet")
    sv   = pd.read_parquet(DATA_RAW / "squad_values.parquet")

    # Merge por team name
    elo_current = elo_current.merge(
        fifa[["team", "fifa_rank", "fifa_points"]], on="team", how="left"
    )
    elo_current = elo_current.merge(
        sv[["team", "squad_value_eur", "squad_value_log"]], on="team", how="left"
    )

    # Guardar
    out_snap = DATA_FEATURES / "team_snapshot.parquet"
    snapshot.to_parquet(out_snap, index=False)

    out_curr = DATA_FEATURES / "elo_current.parquet"
    elo_current.to_parquet(out_curr, index=False)

    print(f"\n[OK] team_snapshot.parquet: {len(snapshot):,} filas")
    print(f"[OK] elo_current.parquet:   {len(elo_current)} equipos")
    print(f"  ELO range: {snapshot['elo_pre'].min():.0f} – {snapshot['elo_pre'].max():.0f}")
    print(f"  Equipos con FIFA rank: {elo_current['fifa_rank'].notna().sum()}/{len(elo_current)}")
    print(f"  Equipos con squad value: {elo_current['squad_value_eur'].notna().sum()}/{len(elo_current)}")


if __name__ == "__main__":
    build_elo_features()
