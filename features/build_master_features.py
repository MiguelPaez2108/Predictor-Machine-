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


def build_wc2026_prediction_input(fixtures_df: pd.DataFrame, prediction_as_of_start: pd.Timestamp) -> pd.DataFrame:
    """
    Construye data/model/wc2026_prediction_input.parquet para usar con:
      python models/ensemble.py --wc2026

    Suposición operativa:
    - Solo partidos de fase de grupos (48 equipos / 12 grupos).
    - Fecha de predicción = prediction_as_of_start + offset por orden del fixture.
    - Para features 'as-of' se usan los valores más recientes con date < as_of_date
      (sin leakage hacia adelante).
    """
    # --- Fecha por partido (as-of) ---
    fixtures = fixtures_df.copy().reset_index(drop=True)
    fixtures["date"] = prediction_as_of_start + pd.to_timedelta(fixtures.index, unit="D")

    # --- ELO (proxy futuro usando snapshot actual por equipo) ---
    # Para predicción, 'neutral' en fixtures => home_advantage = 0.
    elo_current = pd.read_parquet(DATA_RAW / "elo_historical.parquet").copy()
    # Si no existe aquí, usamos elo_current.parquet si está.
    from config import DATA_FEATURES
    elo_curr_path = DATA_FEATURES / "elo_current.parquet"
    if elo_curr_path.exists():
        elo_current = pd.read_parquet(elo_curr_path)
    else:
        # fallback: intentar tomar último elo_home_post/elo_away_post desde elo_historical
        elo = pd.read_parquet(DATA_RAW / "elo_historical.parquet").copy()
        elo["date"] = pd.to_datetime(elo["date"])
        elo = elo.sort_values("date").reset_index(drop=True)
        elo_home = elo[["date", "home_team", "elo_home_post"]].rename(
            columns={"home_team": "team", "elo_home_post": "elo_current"}
        )
        elo_away = elo[["date", "away_team", "elo_away_post"]].rename(
            columns={"away_team": "team", "elo_away_post": "elo_current"}
        )
        elo_long = pd.concat([elo_home, elo_away], ignore_index=True)
        elo_current = elo_long.groupby("team")["elo_current"].last().reset_index()

    elo_map = elo_current.set_index("team")["elo_current"].to_dict()

    fixtures["elo_home_pre"] = fixtures["home_team"].map(elo_map)
    fixtures["elo_away_pre"] = fixtures["away_team"].map(elo_map)

    # Expected home con neutral=true (home_advantage=0)
    # expected_home = 1/(1+10^((elo_away - elo_home)/400))
    dd = fixtures["elo_away_pre"] - fixtures["elo_home_pre"]
    fixtures["expected_home_elo"] = 1.0 / (1.0 + np.power(10.0, dd / 400.0))
    fixtures["expected_home"] = fixtures["expected_home_elo"]  # compat

    # --- FIFA ranking (último snapshot por team) ---
    fifa = pd.read_parquet(DATA_RAW / "fifa_rankings.parquet").copy()
    fifa["date"] = pd.to_datetime(fifa["date"])
    fifa_snap = (
        fifa.sort_values("date")
            .groupby("team")[["fifa_rank", "fifa_points"]]
            .last()
            .reset_index()
    )
    fixtures = fixtures.merge(
        fifa_snap.rename(columns={"team": "home_team", "fifa_rank": "home_fifa_rank", "fifa_points": "home_fifa_pts"}),
        on="home_team", how="left"
    )
    fixtures = fixtures.merge(
        fifa_snap.rename(columns={"team": "away_team", "fifa_rank": "away_fifa_rank", "fifa_points": "away_fifa_pts"}),
        on="away_team", how="left"
    )

    # --- Squad values (snapshot por team) ---
    sv = pd.read_parquet(DATA_RAW / "squad_values.parquet").copy()
    if "date" in sv.columns:
        sv["date"] = pd.to_datetime(sv["date"])
        sv = (
            sv.sort_values("date")
              .groupby("team")[["squad_value_log"]].last()
              .reset_index()
        )
    fixtures = fixtures.merge(
        sv[["team", "squad_value_log"]].rename(columns={"team": "home_team", "squad_value_log": "home_sv_log"}),
        on="home_team", how="left"
    )
    fixtures = fixtures.merge(
        sv[["team", "squad_value_log"]].rename(columns={"team": "away_team", "squad_value_log": "away_sv_log"}),
        on="away_team", how="left"
    )

    # --- Form rolling / H2H (as-of determinista sin merge_asof) ---
    # Para WC solo hay 72 partidos, así que iterar es aceptable y evita
    # problemas de ordenación de merge_asof en pandas.
    form_path = DATA_FEATURES / "form_rolling.parquet"
    form = pd.read_parquet(form_path).copy()
    form["date"] = pd.to_datetime(form["date"])

    form_cols = [
        "form_weighted", "form_pts_last6", "form_pts_last10", "form_gf_avg", "form_ga_avg",
        "form_gd_avg", "form_wins", "form_draws", "form_losses", "momentum_trend"
    ]

    h2h = pd.read_parquet(DATA_FEATURES / "h2h_history.parquet").copy()
    h2h["date"] = pd.to_datetime(h2h["date"])
    h2h_cols = ["h2h_matches", "h2h_win_rate", "h2h_gf_avg", "h2h_ga_avg",
                "h2h_gd_avg", "h2h_pts_avg"]

    fixtures_sorted = fixtures.sort_values("date").reset_index(drop=True)

    # Asegurar strings consistentes
    fixtures_sorted["home_team"] = fixtures_sorted["home_team"].astype(str)
    fixtures_sorted["away_team"] = fixtures_sorted["away_team"].astype(str)
    form["team"] = form["team"].astype(str)
    form["opponent"] = form["opponent"].astype(str)
    h2h["team"] = h2h["team"].astype(str)
    h2h["opponent"] = h2h["opponent"].astype(str)

    # Pre-crear columnas
    for c in form_cols:
        fixtures_sorted[f"home_{c}"] = 0.0
        fixtures_sorted[f"away_{c}"] = 0.0
    for c in h2h_cols:
        fixtures_sorted[f"home_{c}"] = 0.0
        fixtures_sorted[f"away_{c}"] = 0.0

    # Indexaciones para acelerar (por team, opponent)
    form_g = form.groupby(["team", "opponent"])
    h2h_g  = h2h.groupby(["team", "opponent"])

    for idx, r in fixtures_sorted.iterrows():
        as_of = r["date"]
        ht = r["home_team"]
        at = r["away_team"]

        # Form home (ht vs at)
        key = (ht, at)
        if key in form_g.indices:
            sub = form_g.get_group(key)
            sub = sub[sub["date"] < as_of].sort_values("date")
            if not sub.empty:
                last = sub.iloc[-1]
                for c in form_cols:
                    fixtures_sorted.at[idx, f"home_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

        # Form away (at vs ht)
        key = (at, ht)
        if key in form_g.indices:
            sub = form_g.get_group(key)
            sub = sub[sub["date"] < as_of].sort_values("date")
            if not sub.empty:
                last = sub.iloc[-1]
                for c in form_cols:
                    fixtures_sorted.at[idx, f"away_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

        # H2H home
        key = (ht, at)
        if key in h2h_g.indices:
            sub = h2h_g.get_group(key)
            sub = sub[sub["date"] < as_of].sort_values("date")
            if not sub.empty:
                last = sub.iloc[-1]
                for c in h2h_cols:
                    fixtures_sorted.at[idx, f"home_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

        # H2H away
        key = (at, ht)
        if key in h2h_g.indices:
            sub = h2h_g.get_group(key)
            sub = sub[sub["date"] < as_of].sort_values("date")
            if not sub.empty:
                last = sub.iloc[-1]
                for c in h2h_cols:
                    fixtures_sorted.at[idx, f"away_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

    # --- xG + Context (as-of determinista sin merge_asof) ---
    # Para WC (72 partidos) la iteración es aceptable y evita fallos de ordenación.

    # xG rolling avg5 por equipo
    xg = pd.read_parquet(DATA_FEATURES / "xg_derived.parquet").copy()
    xg["date"] = pd.to_datetime(xg["date"])
    if "team" not in xg.columns:
        raise ValueError("xg_derived.parquet no contiene columna 'team'")

    xg = xg.sort_values(["team", "date"])

    # Preindex: por team, guardar arrays para búsquedas
    xg_groups = {k: g for k, g in xg.groupby("team")}

    # Context por (team, opponent)
    ctx = pd.read_parquet(DATA_FEATURES / "context_features.parquet").copy()
    ctx["date"] = pd.to_datetime(ctx["date"])
    ctx_cols = ["days_rest", "fatigue_index", "matches_last_30d", "matches_in_tournament",
                "is_knockout_phase", "tournament_weight"]
    ctx = ctx.sort_values(["team", "opponent", "date"])
    ctx_groups = {k: g for k, g in ctx.groupby(["team", "opponent"])}

    for idx, r in fixtures_sorted.iterrows():
        as_of = r["date"]
        ht = r["home_team"]
        at = r["away_team"]

        # ---- xG home (team=ht) ----
        g = xg_groups.get(ht)
        if g is not None and not g.empty:
            past = g[g["date"] < as_of].tail(5)
            fixtures_sorted.at[idx, "home_xg_avg5"] = float(past["xg"].mean()) if len(past) > 0 else 0.0
            fixtures_sorted.at[idx, "home_xga_avg5"] = float(past["xga"].mean()) if len(past) > 0 else 0.0
            fixtures_sorted.at[idx, "home_xg_conv_avg5"] = float(past["xg_conversion"].mean()) if len(past) > 0 else 0.0

        # ---- xG away (team=at) ----
        g = xg_groups.get(at)
        if g is not None and not g.empty:
            past = g[g["date"] < as_of].tail(5)
            fixtures_sorted.at[idx, "away_xg_avg5"] = float(past["xg"].mean()) if len(past) > 0 else 0.0
            fixtures_sorted.at[idx, "away_xga_avg5"] = float(past["xga"].mean()) if len(past) > 0 else 0.0
            fixtures_sorted.at[idx, "away_xg_conv_avg5"] = float(past["xg_conversion"].mean()) if len(past) > 0 else 0.0

        # ---- Context home (team=ht, opponent=at) ----
        cg = ctx_groups.get((ht, at))
        if cg is not None and not cg.empty:
            past = cg[cg["date"] < as_of].sort_values("date")
            if not past.empty:
                last = past.iloc[-1]
                for c in ctx_cols:
                    fixtures_sorted.at[idx, f"home_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

        # ---- Context away (team=at, opponent=ht) ----
        cg = ctx_groups.get((at, ht))
        if cg is not None and not cg.empty:
            past = cg[cg["date"] < as_of].sort_values("date")
            if not past.empty:
                last = past.iloc[-1]
                for c in ctx_cols:
                    fixtures_sorted.at[idx, f"away_{c}"] = float(last[c]) if pd.notna(last[c]) else 0.0

    # --- delta features (match training schema) ---
    df = fixtures_sorted

    df["delta_elo"] = df["elo_home_pre"] - df["elo_away_pre"]
    df["delta_form"] = df["home_form_weighted"] - df["away_form_weighted"]

    df["delta_xg"]  = df["home_xg_avg5"] - df["away_xg_avg5"]
    df["delta_xga"] = df["home_xga_avg5"] - df["away_xga_avg5"]

    df["delta_fifa_rank"] = df["away_fifa_rank"] - df["home_fifa_rank"]
    df["delta_sv_log"] = df["home_sv_log"] - df["away_sv_log"]

    df["delta_rest"] = df["home_days_rest"] - df["away_days_rest"]

    # fill NaNs conservatively (RF/XGB tend to need numeric)
    for c in ["delta_elo", "expected_home_elo", "delta_form", "delta_xg", "delta_xga",
              "delta_fifa_rank", "delta_sv_log", "delta_rest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # columnas adicionales útiles
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    # salida mínima compatible con predict_match(meta_input loop)
    out_cols = [
        "date",
        "home_team", "away_team",
        "expected_home_elo",
        "delta_elo", "delta_form",
        "delta_xg", "delta_xga",
        "delta_fifa_rank", "delta_sv_log",
        "delta_rest",
    ]
    # si existen, mantener también h2h del partido (a veces útil)
    for c in ["h2h_matches", "h2h_win_rate", "h2h_gf_avg", "h2h_ga_avg", "h2h_gd_avg", "h2h_pts_avg"]:
        if f"home_{c}" in df.columns and f"away_{c}" in df.columns:
            out_cols.extend([f"home_{c}", f"away_{c}"])

    return df[out_cols].copy()


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

    # ── Generar wc2026_prediction_input.parquet (solo grupos) ─────────────
    wc_out = DATA_MODEL / "wc2026_prediction_input.parquet"
    import traceback
    try:
        from simulation.wc2026_fixtures import generate_group_fixtures
        fixtures = generate_group_fixtures()
        fixtures_df = pd.DataFrame(fixtures)

        # generate_group_fixtures incluye home/away + group; no necesitamos neutral/venue para el modelo
        if "neutral" in fixtures_df.columns:
            fixtures_df = fixtures_df.drop(columns=["neutral"], errors="ignore")

        as_of_start = pd.Timestamp("2026-06-11")
        wc_pred_input = build_wc2026_prediction_input(fixtures_df, as_of_start)
        print(f"\nDEBUG: wc_pred_input generado con {len(wc_pred_input):,} filas")
        wc_pred_input.to_parquet(wc_out, index=False)

        print(f"\n[OK] wc2026_prediction_input.parquet: {len(wc_pred_input):,} filas en {wc_out}")
    except Exception:
        print("\nWARN: no se pudo generar wc2026_prediction_input.parquet (ver traceback):")
        traceback.print_exc()

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

    print(f"\n[OK] training_set.parquet:   {len(train):,} filas")
    print(f"[OK] validation_set.parquet: {len(val):,} filas")
    print(f"[OK] test_set.parquet:       {len(test):,} filas")


if __name__ == "__main__":
    build_master_features()
