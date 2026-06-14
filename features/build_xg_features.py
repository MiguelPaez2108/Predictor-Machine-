"""
build_xg_features.py — Extracción avanzada de features desde StatsBomb
=======================================================================
Basado en la chuleta de statsbombpy (Datogami).

Extrae por partido × equipo:
  BÁSICO:
    xg, xga, goals, goals_against, shots, shots_on_target
    xg_conversion, xga_prevention

  AVANZADO (desde eventos detallados):
    xg_open_play      → xG solo de juego abierto (excluye penaltis/faltas)
    xg_set_piece      → xG de balón parado (corners, faltas directas)
    ppda              → Passes Per Defensive Action (presión defensiva)
    shot_quality      → xG por remate (calidad de ocasiones creadas)
    pass_completion   → % pases completados (pass_outcome NaN = completado)
    progressive_passes→ pases que avanzan ≥10 unidades hacia portería rival
    pressure_rate     → acciones de presión por minuto de posesión rival
    under_pressure_shots → tiros bajo presión / tiros totales

Salida: data/features/xg_derived.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_SB, DATA_FEATURES, STATSBOMB_COMPS, ensure_dirs


TOURNAMENT_LABELS = {
    "wc_2018"          : ("FIFA World Cup", 2018),
    "wc_2022"          : ("FIFA World Cup", 2022),
    "euro_2020"        : ("UEFA Euro",      2020),
    "euro_2024"        : ("UEFA Euro",      2024),
    "copa_america_2024": ("Copa America",   2024),
    "afcon_2023"       : ("AFCON",          2023),
}

# Campo de juego StatsBomb: 120 × 80 unidades
FIELD_LENGTH = 120.0
FIELD_WIDTH  = 80.0
# Una portería está en x=120; el área grande va hasta x=102
# Un pase es "progresivo" si avanza ≥10 unidades hacia x=120
PROGRESSIVE_THRESHOLD = 10.0


def extract_location(loc):
    """Extrae [x, y] de la columna location (lista, tupla, numpy array o NaN)."""
    try:
        if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
            return float(loc[0]), float(loc[1])
    except Exception:
        pass
    return np.nan, np.nan


def compute_ppda(events_df: pd.DataFrame, team: str, opponent: str) -> float:
    """
    PPDA (Passes Per Defensive Action).
    = pases_completados_rival / acciones_defensivas_propias_en_campo_rival

    Mide la intensidad de la presión defensiva alta.
    PPDA bajo → presión alta (estilo Klopp/Guardiola)
    PPDA alto → bloque bajo

    Acciones defensivas: Pressure, Tackle, Interception en campo rival (x < 60)
    """
    if events_df.empty:
        return np.nan

    # Pases completados del rival en su propio campo (x < 60 desde su perspectiva)
    opp_passes = events_df[
        (events_df["team"] == opponent) &
        (events_df["type"] == "Pass")
    ]
    if "pass_outcome" in opp_passes.columns:
        opp_completed_passes = opp_passes[opp_passes["pass_outcome"].isna()]
    else:
        opp_completed_passes = opp_passes

    # Acciones defensivas del equipo analizado en campo rival
    # (campo rival = x > 60 en el sistema de coordenadas StatsBomb)
    defensive_actions = events_df[
        (events_df["team"] == team) &
        (events_df["type"].isin(["Pressure", "Tackle", "Interception",
                                  "Ball Recovery", "Duel"]))
    ]

    # Filtrar acciones en campo rival (x > 60 = mitad del campo contrario)
    if "location" in defensive_actions.columns and not defensive_actions.empty:
        xs = defensive_actions["location"].apply(
            lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 1 else np.nan
        )
        defensive_in_opp_half = defensive_actions[xs > 60]
    else:
        defensive_in_opp_half = defensive_actions

    n_def = len(defensive_in_opp_half)
    n_pass = len(opp_completed_passes)

    if n_def == 0:
        return np.nan
    return round(n_pass / n_def, 4)


def compute_progressive_passes(passes_df: pd.DataFrame) -> int:
    """
    Pases progresivos: pases completados que avanzan ≥10 unidades hacia x=120.
    Basado en la definición de StatsBomb / FBref.
    """
    if passes_df.empty or "location" not in passes_df.columns:
        return 0

    completed = passes_df[passes_df["pass_outcome"].isna()] \
        if "pass_outcome" in passes_df.columns else passes_df

    if "pass_end_location" not in completed.columns:
        return 0

    prog = 0
    for _, row in completed.iterrows():
        x_start, _ = extract_location(row.get("location"))
        x_end,   _ = extract_location(row.get("pass_end_location"))
        if pd.notna(x_start) and pd.notna(x_end):
            if (x_end - x_start) >= PROGRESSIVE_THRESHOLD:
                prog += 1
    return prog


def process_tournament(folder: Path, tournament_name: str, year: int) -> pd.DataFrame:
    """
    Extrae features de xG y avanzadas de un torneo StatsBomb.
    Lee los archivos events_{match_id}.parquet ya descargados.
    """
    index_path = folder / "index.parquet"
    if not index_path.exists():
        print(f"  [SKIP] {folder.name} — sin index.parquet")
        return pd.DataFrame()

    index   = pd.read_parquet(index_path)
    matches_path = folder / "matches.parquet"
    matches = pd.read_parquet(matches_path) if matches_path.exists() else pd.DataFrame()

    # Lookup match_id → fecha
    date_map  = {}
    stage_map = {}
    if not matches.empty:
        if "match_id" in matches.columns and "match_date" in matches.columns:
            date_map = dict(zip(
                matches["match_id"].astype(int),
                pd.to_datetime(matches["match_date"])
            ))
        if "match_id" in matches.columns and "competition_stage" in matches.columns:
            stage_map = dict(zip(
                matches["match_id"].astype(int),
                matches["competition_stage"].astype(str)
            ))

    records = []

    for _, row in index.iterrows():
        mid      = int(row["match_id"])
        evt_path = folder / f"events_{mid}.parquet"
        if not evt_path.exists():
            continue

        evts       = pd.read_parquet(evt_path)
        match_date = date_map.get(mid, pd.NaT)
        stage      = stage_map.get(mid, "Unknown")
        teams      = [row["home_team"], row["away_team"]]

        # ── Filtros globales ──────────────────────────────────────────────────
        shots_all = evts[evts["type"] == "Shot"].copy() \
            if "type" in evts.columns else pd.DataFrame()
        passes_all = evts[evts["type"] == "Pass"].copy() \
            if "type" in evts.columns else pd.DataFrame()

        for team in teams:
            opp = teams[1] if team == teams[0] else teams[0]

            team_shots = shots_all[shots_all["team"] == team] \
                if not shots_all.empty else pd.DataFrame()
            opp_shots  = shots_all[shots_all["team"] == opp] \
                if not shots_all.empty else pd.DataFrame()
            team_passes = passes_all[passes_all["team"] == team] \
                if not passes_all.empty else pd.DataFrame()

            # ── xG total ──────────────────────────────────────────────────────
            xg_col = "shot_statsbomb_xg"
            xg  = float(team_shots[xg_col].sum()) if xg_col in team_shots.columns else 0.0
            xga = float(opp_shots[xg_col].sum())  if xg_col in opp_shots.columns  else 0.0

            # ── Goles ─────────────────────────────────────────────────────────
            goals = goals_against = 0
            if "shot_outcome" in team_shots.columns:
                goals = int((team_shots["shot_outcome"] == "Goal").sum())
            if "shot_outcome" in opp_shots.columns:
                goals_against = int((opp_shots["shot_outcome"] == "Goal").sum())

            # ── Tiros ─────────────────────────────────────────────────────────
            shots_total = len(team_shots)
            shots_ot    = 0
            if "shot_outcome" in team_shots.columns:
                on_target = {"Goal", "Saved", "Saved To Post"}
                shots_ot  = int(team_shots["shot_outcome"].isin(on_target).sum())

            # ── xG por tipo (Open Play vs Set Piece) ─────────────────────────
            xg_open_play = xg_set_piece = 0.0
            if xg_col in team_shots.columns and "shot_type" in team_shots.columns:
                open_play_types = {"Open Play", "From Corner"}
                set_piece_types = {"Free Kick", "Penalty"}
                xg_open_play = float(
                    team_shots[team_shots["shot_type"].isin(open_play_types)][xg_col].sum()
                )
                xg_set_piece = float(
                    team_shots[team_shots["shot_type"].isin(set_piece_types)][xg_col].sum()
                )

            # ── Calidad de remates (xG por tiro) ─────────────────────────────
            shot_quality = round(xg / shots_total, 4) if shots_total > 0 else np.nan

            # ── Tiros bajo presión ────────────────────────────────────────────
            shots_under_pressure = 0
            if "under_pressure" in team_shots.columns:
                shots_under_pressure = int(team_shots["under_pressure"].fillna(False).sum())
            pressure_shot_rate = round(
                shots_under_pressure / shots_total, 4
            ) if shots_total > 0 else np.nan

            # ── Completion rate de pases ──────────────────────────────────────
            pass_completion = np.nan
            if not team_passes.empty and "pass_outcome" in team_passes.columns:
                n_total     = len(team_passes)
                n_completed = int(team_passes["pass_outcome"].isna().sum())
                pass_completion = round(n_completed / n_total, 4) if n_total > 0 else np.nan

            # ── Pases progresivos ─────────────────────────────────────────────
            progressive_passes = compute_progressive_passes(team_passes)

            # ── PPDA (presión defensiva alta) ─────────────────────────────────
            ppda = compute_ppda(evts, team, opp)

            # ── Acciones bajo presión del equipo ─────────────────────────────
            pressure_actions = 0
            if "type" in evts.columns:
                pressure_actions = int(
                    ((evts["team"] == team) & (evts["type"] == "Pressure")).sum()
                )

            records.append({
                # Identificadores
                "match_id"           : mid,
                "date"               : match_date,
                "tournament"         : tournament_name,
                "year"               : year,
                "stage"              : stage,
                "team"               : team,
                "opponent"           : opp,
                # xG básico
                "xg"                 : round(xg, 4),
                "xga"                : round(xga, 4),
                "goals"              : goals,
                "goals_against"      : goals_against,
                "shots"              : shots_total,
                "shots_on_target"    : shots_ot,
                "xg_conversion"      : round(goals / xg, 4) if xg > 0 else np.nan,
                "xga_prevention"     : round(goals_against / xga, 4) if xga > 0 else np.nan,
                # xG avanzado
                "xg_open_play"       : round(xg_open_play, 4),
                "xg_set_piece"       : round(xg_set_piece, 4),
                "shot_quality"       : shot_quality,
                "shots_under_pressure_rate": pressure_shot_rate,
                # Pases
                "pass_completion"    : pass_completion,
                "progressive_passes" : progressive_passes,
                # Presión
                "ppda"               : ppda,
                "pressure_actions"   : pressure_actions,
            })

    return pd.DataFrame(records)


def build_xg_features():
    ensure_dirs()
    all_dfs = []

    for comp_id, season_id, out_dir in STATSBOMB_COMPS:
        folder = Path(out_dir)
        label  = folder.name
        t_name, t_year = TOURNAMENT_LABELS.get(label, (label, 0))
        print(f"→ Procesando {label} ({t_name} {t_year})...")
        df = process_tournament(folder, t_name, t_year)
        if not df.empty:
            print(f"  {len(df)//2} partidos | {len(df)} filas | "
                  f"xG_mean={df['xg'].mean():.3f} | "
                  f"PPDA_mean={df['ppda'].mean():.2f}")
            all_dfs.append(df)
        else:
            print(f"  ⚠ Sin datos (ejecutar 02_download_statsbomb.py primero)")

    if not all_dfs:
        print("ERROR: No hay datos de StatsBomb descargados.")
        return

    result = pd.concat(all_dfs, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["date", "match_id", "team"]).reset_index(drop=True)

    out_path = DATA_FEATURES / "xg_derived.parquet"
    result.to_parquet(out_path, index=False)

    print(f"\n✓ xg_derived.parquet — {len(result):,} filas")
    print(f"  Torneos: {result['tournament'].unique().tolist()}")
    print(f"  Rango: {result['date'].min().date()} → {result['date'].max().date()}")
    print(f"\n  Cobertura de features avanzadas:")
    for col in ["xg_open_play", "ppda", "pass_completion", "progressive_passes", "shot_quality"]:
        pct = result[col].notna().mean() * 100
        print(f"    {col:35s}: {pct:.1f}%")


if __name__ == "__main__":
    build_xg_features()
