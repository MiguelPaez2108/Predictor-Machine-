"""
build_wc2026_live_features.py — Features en vivo del WC2026
============================================================
Lee los datos de Sofascore de la fase de grupos completada y genera
un snapshot por equipo con métricas promedio (xG, xGA, goles, posesión, etc.).

Entrada:
  data/raw/sofascore_wc2026/match_team_stats.parquet   (144 filas: 72×2)
  data/raw/sofascore_wc2026/matches_index.parquet      (72 filas)

Salida:
  data/features/wc2026_live_snapshot.parquet            (48 filas: 1 por equipo)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_FEATURES


def build_wc2026_live_features() -> pd.DataFrame:
    """
    Construye un snapshot de features en vivo por equipo a partir de los
    datos de Sofascore de la fase de grupos del WC2026.
    """
    sofascore_dir = DATA_RAW / "sofascore_wc2026"

    # ── Cargar stats de equipo por partido ────────────────────────────────────
    stats_path = sofascore_dir / "match_team_stats.parquet"
    if not stats_path.exists():
        raise FileNotFoundError(f"No encontrado: {stats_path}")
    stats = pd.read_parquet(stats_path)
    print(f"  match_team_stats: {len(stats)} filas, {len(stats.columns)} cols")

    # ── Cargar índice de partidos (para goles reales) ─────────────────────────
    index_path = sofascore_dir / "matches_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"No encontrado: {index_path}")
    matches = pd.read_parquet(index_path)
    print(f"  matches_index: {len(matches)} filas")

    # ── Identificar columna de match_id ───────────────────────────────────────
    # stats usa 'match_id', matches usa 'sofascore_match_id'
    stats_id_col = "match_id" if "match_id" in stats.columns else "sofascore_match_id"
    matches_id_col = "sofascore_match_id" if "sofascore_match_id" in matches.columns else "match_id"

    # ── Calcular xGA (xG del rival) mediante self-join ────────────────────────
    # Para cada fila (team X en match M), el xGA es el xG del OTRO equipo en match M
    if "expected_goals" in stats.columns:
        xg_per_match = stats[[stats_id_col, "team", "expected_goals"]].copy()
        xg_per_match = xg_per_match.rename(columns={stats_id_col: "mid"})

        # Self-join: para cada team, encontrar el xG del oponente en el mismo partido
        xga_lookup = xg_per_match.rename(columns={
            "team": "opponent", "expected_goals": "xga"
        })
        merged = xg_per_match.merge(xga_lookup, on="mid", how="left")
        merged = merged[merged["team"] != merged["opponent"]]

        xga_by_team = merged.groupby("team")["xga"].mean()
    else:
        xga_by_team = pd.Series(dtype=float)

    # ── Calcular goles a favor y en contra desde matches_index ────────────────
    goals_records = []
    for _, row in matches.iterrows():
        mid = row[matches_id_col]
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]
        goals_records.append({"team": home, "gf": hg, "ga": ag, "mid": mid})
        goals_records.append({"team": away, "gf": ag, "ga": hg, "mid": mid})
    goals_df = pd.DataFrame(goals_records)
    goals_agg = goals_df.groupby("team").agg(
        gf_avg=("gf", "mean"),
        ga_avg=("ga", "mean"),
        gf_total=("gf", "sum"),
        ga_total=("ga", "sum"),
        matches_played=("mid", "count"),
    )

    # ── Agregar métricas numéricas de match_team_stats por equipo ─────────────
    # Seleccionar solo columnas numéricas (excluyendo IDs y flags)
    exclude_cols = {stats_id_col, "team", "is_home"}
    numeric_cols = [
        c for c in stats.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(stats[c])
    ]

    team_avgs = stats.groupby("team")[numeric_cols].mean()

    # Renombrar 'expected_goals' a 'xg_avg' para claridad
    if "expected_goals" in team_avgs.columns:
        team_avgs = team_avgs.rename(columns={"expected_goals": "xg_avg"})

    # ── Combinar todo ─────────────────────────────────────────────────────────
    snapshot = team_avgs.copy()

    # Agregar xGA
    if not xga_by_team.empty:
        snapshot["xga_avg"] = xga_by_team

    # Agregar goles reales
    for col in ["gf_avg", "ga_avg", "gf_total", "ga_total", "matches_played"]:
        if col in goals_agg.columns:
            snapshot[col] = goals_agg[col]

    # Limpiar: resetear índice
    snapshot = snapshot.reset_index()
    snapshot = snapshot.rename(columns={"index": "team"} if "index" in snapshot.columns else {})

    # Asegurar que 'team' es columna (no índice)
    if "team" not in snapshot.columns and snapshot.index.name == "team":
        snapshot = snapshot.reset_index()

    print(f"\n  Snapshot generado: {len(snapshot)} equipos, {len(snapshot.columns)} columnas")
    print(f"  Columnas: {list(snapshot.columns)}")

    return snapshot


def main():
    print("=" * 60)
    print("  BUILD WC2026 LIVE FEATURES (Sofascore)")
    print("=" * 60)

    snapshot = build_wc2026_live_features()

    # Guardar
    out_path = DATA_FEATURES / "wc2026_live_snapshot.parquet"
    snapshot.to_parquet(out_path, index=False)
    print(f"\n  [OK] Guardado: {out_path}")

    # Resumen
    print(f"\n  --- Resumen xG (top 10) ---")
    if "xg_avg" in snapshot.columns:
        top_xg = snapshot.nlargest(10, "xg_avg")[["team", "xg_avg", "xga_avg", "gf_avg", "ga_avg"]]
        print(top_xg.to_string(index=False))
    else:
        print("  (columna xg_avg no disponible)")

    return snapshot


if __name__ == "__main__":
    main()
