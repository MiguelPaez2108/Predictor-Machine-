"""
build_master_features.py
Une todos los features en un training set limpio por partido.
Formato: una fila por PARTIDO (perspectiva home vs away).
Features: delta_elo, delta_xg, forma, H2H, contexto, ranking, mercado.
Salidas:
  data/model/training_set.parquet   → hasta TRAIN_END_DATE
  data/model/validation_set.parquet → hasta VALIDATION_END_DATE
  data/model/wc2026_prediction_input.parquet → para predicción
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_RAW, DATA_FEATURES, DATA_MODEL,
                    TRAIN_END_DATE, VALIDATION_END_DATE, ensure_dirs)


def load_features():
    """Carga todos los parquets de features."""
    print("Cargando features...")

    ir     = pd.read_parquet(DATA_RAW / "international_results.parquet")
    elo_h  = pd.read_parquet(DATA_RAW / "elo_historical.parquet")
    fifa   = pd.read_parquet(DATA_RAW / "fifa_rankings.parquet")
    sv     = pd.read_parquet(DATA_RAW / "squad_values.parquet")

    form_path  = DATA_FEATURES / "form_rolling.parquet"
    ctx_path   = DATA_FEATURES / "context_features.parquet"
    h2h_path   = DATA_FEATURES / "h2h_history.parquet"
    xg_path    = DATA_FEATURES / "xg_derived.parquet"

    form = pd.read_parquet(form_path)  if form_path.exists() else pd.DataFrame()
    ctx  = pd.read_parquet(ctx_path)   if ctx_path.exists()  else pd.DataFrame()
    h2h  = pd.read_parquet(h2h_path)   if h2h_path.exists()  else pd.DataFrame()
    xg   = pd.read_parquet(xg_path)    if xg_path.exists()   else pd.DataFrame()

    print(f"  IR: {len(ir):,} | ELO: {len(elo_h):,} | Form: {len(form):,} | "
          f"Ctx: {len(ctx):,} | H2H: {len(h2h):,} | xG: {len(xg):,}")

    return ir, elo_h, fifa, sv, form, ctx, h2h, xg


def merge_team_features(df_base: pd.DataFrame, form, ctx, h2h, xg, fifa, sv):
    """
    Para cada partido en df_base (formato IR wide: home vs away),
    fusiona los features de AMBOS equipos en una sola fila.
    """
    result = df_base.copy()

    # ── Form features ───────────────────────────────────────────────────────
    form_cols = ["form_weighted", "form_pts_last6", "form_pts_last10",
                 "form_gf_avg", "form_ga_avg", "form_gd_avg",
                 "form_wins", "form_draws", "form_losses", "momentum_trend"]

    if not form.empty:
        # Merge para home
        form_h = form[["date", "team", "opponent"] + form_cols].rename(
            columns={c: f"home_{c}" for c in form_cols}
        )
        result = result.merge(
            form_h,
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "team", "opponent"],
            how="left"
        ).drop(columns=["team", "opponent"], errors="ignore")

        # Merge para away
        form_a = form[["date", "team", "opponent"] + form_cols].rename(
            columns={c: f"away_{c}" for c in form_cols}
        )
        result = result.merge(
            form_a,
            left_on=["date", "away_team", "home_team"],
            right_on=["date", "team", "opponent"],
            how="left",
            suffixes=("", "_dup")
        ).drop(columns=["team", "opponent"] + [c for c in result.columns if c.endswith("_dup")],
               errors="ignore")

    # ── ELO features ─────────────────────────────────────────────────────────
    # Ya están en elo_historical con home/away
    # (ya tiene elo_home_pre, elo_away_pre, expected_home)

    # ── Context features ─────────────────────────────────────────────────────
    ctx_cols = ["days_rest", "fatigue_index", "matches_last_30d",
                "matches_in_tournament", "is_knockout_phase", "tournament_weight"]

    if not ctx.empty:
        ctx_h = ctx[["date", "team", "opponent"] + ctx_cols].rename(
            columns={c: f"home_{c}" for c in ctx_cols}
        )
        result = result.merge(
            ctx_h,
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "team", "opponent"],
            how="left"
        ).drop(columns=["team", "opponent"], errors="ignore")

        ctx_a = ctx[["date", "team", "opponent"] + ctx_cols].rename(
            columns={c: f"away_{c}" for c in ctx_cols}
        )
        result = result.merge(
            ctx_a,
            left_on=["date", "away_team", "home_team"],
            right_on=["date", "team", "opponent"],
            how="left",
            suffixes=("", "_dup")
        ).drop(columns=["team", "opponent"] + [c for c in result.columns if c.endswith("_dup")],
               errors="ignore")

    # ── H2H features ─────────────────────────────────────────────────────────
    h2h_cols = ["h2h_matches", "h2h_win_rate", "h2h_gf_avg", "h2h_ga_avg",
                "h2h_gd_avg", "h2h_pts_avg"]

    if not h2h.empty:
        h2h_m = h2h[["date", "team", "opponent"] + h2h_cols]
        result = result.merge(
            h2h_m,
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "team", "opponent"],
            how="left"
        ).drop(columns=["team", "opponent"], errors="ignore")

    # ── xG features (solo partidos con StatsBomb) ─────────────────────────────
    xg_cols = ["xg", "xga", "shots", "shots_on_target", "xg_conversion", "xga_prevention"]

    if not xg.empty:
        # xG es per equipo per partido → necesitamos agrupar últimos 5 partidos pre-fecha
        # Calculamos rolling xG por equipo antes del join
        xg_sorted = xg.sort_values(["team", "date"])
        xg_rolling = []
        for team_name, grp in xg_sorted.groupby("team"):
            grp = grp.reset_index(drop=True)
            for i, row in grp.iterrows():
                past5 = grp[grp["date"] < row["date"]].tail(5)
                xg_rolling.append({
                    "date":         row["date"],
                    "team":         team_name,
                    "xg_avg5":      round(float(past5["xg"].mean()), 4) if len(past5) > 0 else np.nan,
                    "xga_avg5":     round(float(past5["xga"].mean()), 4) if len(past5) > 0 else np.nan,
                    "xg_conv_avg5": round(float(past5["xg_conversion"].mean()), 4) if len(past5) > 0 else np.nan,
                })
        xg_roll_df = pd.DataFrame(xg_rolling)

        xg_h = xg_roll_df.rename(columns={c: f"home_{c}" for c in ["xg_avg5", "xga_avg5", "xg_conv_avg5"]})
        result = result.merge(
            xg_h, left_on=["date", "home_team"], right_on=["date", "team"],
            how="left"
        ).drop(columns=["team"], errors="ignore")

        xg_a = xg_roll_df.rename(columns={c: f"away_{c}" for c in ["xg_avg5", "xga_avg5", "xg_conv_avg5"]})
        result = result.merge(
            xg_a, left_on=["date", "away_team"], right_on=["date", "team"],
            how="left",
            suffixes=("", "_dup")
        ).drop(columns=[c for c in result.columns if c.endswith("_dup")],
               errors="ignore")

    # ── FIFA rankings ─────────────────────────────────────────────────────────
    fifa_latest = fifa.copy()
    fifa_latest["date"] = pd.to_datetime(fifa_latest["date"])
    # Usamos siempre el ranking más reciente disponible
    fifa_snap = (
        fifa_latest.sort_values("date")
        .groupby("team")[["fifa_rank", "fifa_points"]]
        .last()
        .reset_index()
    )

    result = result.merge(
        fifa_snap.rename(columns={"team": "home_team",
                                  "fifa_rank": "home_fifa_rank",
                                  "fifa_points": "home_fifa_pts"}),
        on="home_team", how="left"
    )
    result = result.merge(
        fifa_snap.rename(columns={"team": "away_team",
                                  "fifa_rank": "away_fifa_rank",
                                  "fifa_points": "away_fifa_pts"}),
        on="away_team", how="left"
    )

    # ── Squad values ───────────────────────────────────────────────────────────
    result = result.merge(
        sv[["team", "squad_value_log"]].rename(columns={"team": "home_team",
                                                          "squad_value_log": "home_sv_log"}),
        on="home_team", how="left"
    )
    result = result.merge(
        sv[["team", "squad_value_log"]].rename(columns={"team": "away_team",
                                                          "squad_value_log": "away_sv_log"}),
        on="away_team", how="left"
    )

    return result


def compute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features diferenciales (home - away)."""
    df = df.copy()

    if "elo_home_pre" in df.columns and "elo_away_pre" in df.columns:
        df["delta_elo"]      = df["elo_home_pre"] - df["elo_away_pre"]
        df["expected_home_elo"] = df.get("expected_home", np.nan)

    if "home_form_weighted" in df.columns and "away_form_weighted" in df.columns:
        df["delta_form"]     = df["home_form_weighted"] - df["away_form_weighted"]
        df["delta_form_pts6"]= df["home_form_pts_last6"] - df["away_form_pts_last6"]
        df["delta_gf_avg"]   = df["home_form_gf_avg"] - df["away_form_gf_avg"]
        df["delta_ga_avg"]   = df["home_form_ga_avg"] - df["away_form_ga_avg"]

    if "home_fifa_rank" in df.columns and "away_fifa_rank" in df.columns:
        df["delta_fifa_rank"]= df["away_fifa_rank"] - df["home_fifa_rank"]  # invertido: menor es mejor
        df["delta_fifa_pts"] = df["home_fifa_pts"] - df["away_fifa_pts"]

    if "home_sv_log" in df.columns and "away_sv_log" in df.columns:
        df["delta_sv_log"]   = df["home_sv_log"] - df["away_sv_log"]

    if "home_xg_avg5" in df.columns and "away_xg_avg5" in df.columns:
        df["delta_xg"]       = df["home_xg_avg5"] - df["away_xg_avg5"]
        df["delta_xga"]      = df["home_xga_avg5"] - df["away_xga_avg5"]

    if "home_days_rest" in df.columns and "away_days_rest" in df.columns:
        df["delta_rest"]     = df["home_days_rest"] - df["away_days_rest"]
        df["delta_fatigue"]  = df["home_fatigue_index"] - df["away_fatigue_index"]

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Añade variable objetivo: resultado del partido."""
    df = df.copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["target_result"] = np.where(df["home_score"] > df["away_score"], "H",
                          np.where(df["home_score"] == df["away_score"], "D", "A"))
    df["target_result_num"] = np.where(df["home_score"] > df["away_score"], 1,
                              np.where(df["home_score"] == df["away_score"], 0, -1))
    df["target_home_goals"]  = df["home_score"]
    df["target_away_goals"]  = df["away_score"]
    df["target_total_goals"] = df["home_score"] + df["away_score"]
    df["target_over25"]      = (df["target_total_goals"] > 2.5).astype(int)
    df["target_btts"]        = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    return df


def build_master_features():
    ensure_dirs()

    ir, elo_h, fifa, sv, form, ctx, h2h, xg = load_features()

    ir["date"]   = pd.to_datetime(ir["date"])
    elo_h["date"] = pd.to_datetime(elo_h["date"])

    # Base: merge IR + ELO (tienen el mismo esquema de date/home_team/away_team)
    base = ir.merge(
        elo_h[["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre",
               "elo_home_post", "elo_away_post", "expected_home"]],
        on=["date", "home_team", "away_team"],
        how="left"
    )

    print(f"\nBase tras merge IR+ELO: {len(base):,} partidos")

    # Merge de todos los features
    print("Merging features por equipo...")
    full = merge_team_features(base, form, ctx, h2h, xg, fifa, sv)

    # Deltas
    print("Calculando features diferenciales...")
    full = compute_delta_features(full)

    # Target (solo para partidos ya jugados)
    full = add_target(full)

    # Limpiar columnas duplicadas o vacías
    full = full.loc[:, ~full.columns.duplicated()]

    # Filtrar solo partidos con resultado (no NaN en score)
    complete = full.dropna(subset=["home_score", "away_score"]).copy()
    print(f"Partidos con resultado completo: {len(complete):,}")

    # ── Splits temporales ──────────────────────────────────────────────────────
    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end   = pd.Timestamp(VALIDATION_END_DATE)

    train = complete[complete["date"] <= train_end].copy()
    val   = complete[(complete["date"] > train_end) & (complete["date"] <= val_end)].copy()
    test  = complete[complete["date"] > val_end].copy()

    print(f"\nSplits temporales:")
    print(f"  Train (hasta {TRAIN_END_DATE}):     {len(train):,} partidos")
    print(f"  Validation ({TRAIN_END_DATE}→{VALIDATION_END_DATE}): {len(val):,} partidos")
    print(f"  Test (desde {VALIDATION_END_DATE}):  {len(test):,} partidos")

    # Guardar
    train.to_parquet(DATA_MODEL / "training_set.parquet", index=False)
    val.to_parquet(DATA_MODEL / "validation_set.parquet", index=False)
    test.to_parquet(DATA_MODEL / "test_set.parquet", index=False)

    # Reporte de cobertura de features
    key_feats = ["delta_elo", "delta_form", "delta_fifa_rank",
                 "delta_sv_log", "delta_xg", "delta_rest"]
    print("\nCobertura de features clave (% no-nulos en training):")
    for f in key_feats:
        if f in train.columns:
            pct = train[f].notna().mean() * 100
            print(f"  {f:25s}: {pct:.1f}%")
        else:
            print(f"  {f:25s}: [no disponible]")

    print(f"\n✓ training_set.parquet:   {len(train):,} filas")
    print(f"✓ validation_set.parquet: {len(val):,} filas")
    print(f"✓ test_set.parquet:       {len(test):,} filas")


if __name__ == "__main__":
    build_master_features()
