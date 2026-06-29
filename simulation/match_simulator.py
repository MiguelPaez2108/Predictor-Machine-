"""
match_simulator.py — Simulador Estocástico de Partidos
=======================================================
Núcleo de la simulación Monte Carlo.

Para cada partido produce:
  1. Un resultado 1X2 muestreado (H/D/A) desde las probabilidades del ensemble.
  2. Un marcador concreto (home_goals, away_goals) muestreado desde
     la distribución de Poisson bivariada de Dixon-Coles / Bayesian MAP.
  3. En fases eliminatorias: resolución de empates (ET + penaltis).

Diseño de rendimiento:
  - Carga modelos una sola vez (singleton) y los reutiliza.
  - Usa numpy puro para PMF de Poisson (sin scipy en el loop hot-path).
  - Cache de lambdas por par de equipos → cada par se computa una sola vez.
  - Soporta 100 000 simulaciones en < 2 min.
"""

import sys
import math
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_MODEL, DATA_RAW, DATA_FEATURES

# ── Pre-computar tabla de factoriales para Poisson rápido ─────────────────────
_MAX_GOALS = 9
_FACTORIALS = np.array([float(math.factorial(k)) for k in range(_MAX_GOALS + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# Singleton de modelos
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_MODELS: Optional[Dict] = None
_LAMBDA_CACHE: Dict[Tuple[str, str], Tuple[float, float]] = {}
_MATCH_PROB_CACHE: Dict[Tuple[str, str, bool], np.ndarray] = {}


def precompute_all_match_probabilities(models: Dict):
    """
    Precalcula en un lote (batch) optimizado las probabilidades ensemble
    calibradas de las 2,256 combinaciones posibles para el Mundial 2026.
    Evita evaluar secuencialmente el Random Forest, acelerando la simulación.
    """
    global _MATCH_PROB_CACHE
    try:
        from predict_match import ALL_TEAMS
    except ImportError:
        return

    # 1. Generar todas las combinaciones posibles
    rows = []
    keys = []
    for i in range(len(ALL_TEAMS)):
        for j in range(i + 1, len(ALL_TEAMS)):
            for is_ko in [False, True]:
                row = build_static_row(ALL_TEAMS[i], ALL_TEAMS[j], is_knockout=is_ko)
                rows.append(row)
                keys.append((ALL_TEAMS[i], ALL_TEAMS[j], is_ko))

    df = pd.DataFrame(rows)

    # 2. Dixon-Coles y Bayesian en lote (batch) usando sus parámetros vectorizados
    dc = models.get("dc")
    if dc and dc.is_fitted:
        att_dc = dc.attack_
        def_dc = dc.defense_
        int_dc = dc.intercept_
        rho_dc = dc.rho_
        mean_att_dc = np.mean(list(att_dc.values())) if att_dc else 0.0
        mean_def_dc = np.mean(list(def_dc.values())) if def_dc else 0.0
    else:
        att_dc, def_dc, int_dc, rho_dc = {}, {}, 0.0, -0.10
        mean_att_dc, mean_def_dc = 0.0, 0.0

    bay = models.get("bay")
    if bay and bay.is_fitted:
        att_bay = bay.attack_
        def_bay = bay.defense_
        int_bay = bay.intercept_
        rho_bay = bay.rho
        mean_att_bay = np.mean(list(att_bay.values())) if att_bay else 0.0
        mean_def_bay = np.mean(list(def_bay.values())) if def_bay else 0.0
    else:
        att_bay, def_bay, int_bay, rho_bay = {}, {}, 0.0, -0.10
        mean_att_bay, mean_def_bay = 0.0, 0.0

    p_h_dc_list, p_d_dc_list, p_a_dc_list = [], [], []
    p_h_bay_list, p_d_bay_list, p_a_bay_list = [], [], []

    for h, a, _ in keys:
        # Dixon-Coles
        lh_dc = np.exp(int_dc + att_dc.get(h, mean_att_dc) + def_dc.get(a, mean_def_dc))
        la_dc = np.exp(int_dc + att_dc.get(a, mean_att_dc) + def_dc.get(h, mean_def_dc))
        ph_dc, pd_dc, pa_dc = poisson_outcome_probs(lh_dc, la_dc, rho_dc)
        p_h_dc_list.append(ph_dc)
        p_d_dc_list.append(pd_dc)
        p_a_dc_list.append(pa_dc)

        # Bayesian
        lh_bay = np.exp(int_bay + att_bay.get(h, mean_att_bay) + def_bay.get(a, mean_def_bay))
        la_bay = np.exp(int_bay + att_bay.get(a, mean_att_bay) + def_bay.get(h, mean_def_bay))
        ph_bay, pd_bay, pa_bay = poisson_outcome_probs(lh_bay, la_bay, rho_bay)
        p_h_bay_list.append(ph_bay)
        p_d_bay_list.append(pd_bay)
        p_a_bay_list.append(pa_bay)

    # 3. Elo Logistic en lote (batch)
    elo = models.get("elo")
    if elo and elo.is_fitted:
        p_elo = elo.predict_proba_df(df)
        p_h_elo = p_elo["p_home_win"].values
        p_d_elo = p_elo["p_draw"].values
        p_a_elo = p_elo["p_away_win"].values
    else:
        p_h_elo = np.full(len(df), 0.333)
        p_d_elo = np.full(len(df), 0.333)
        p_a_elo = np.full(len(df), 0.334)

    # 4. Random Forest en lote (batch)
    rf = models.get("rf")
    if rf and rf.is_fitted:
        p_rf = rf.predict_proba_df(df)
        p_h_rf = p_rf["p_home_win"].values
        p_d_rf = p_rf["p_draw"].values
        p_a_rf = p_rf["p_away_win"].values
    else:
        p_h_rf = np.full(len(df), 0.333)
        p_d_rf = np.full(len(df), 0.333)
        p_a_rf = np.full(len(df), 0.334)

    # 5. XGBoost Meta-modelo en lote (batch)
    meta_rows = []
    for idx, (h, a, is_ko) in enumerate(keys):
        meta_row = {
            "home_team" : h, "away_team" : a,
            "p_home_dc" : p_h_dc_list[idx],  "p_draw_dc" : p_d_dc_list[idx],  "p_away_dc" : p_a_dc_list[idx],
            "p_home_elo": p_h_elo[idx],      "p_draw_elo": p_d_elo[idx],      "p_away_elo": p_a_elo[idx],
            "p_home_rf" : p_h_rf[idx],       "p_draw_rf" : p_d_rf[idx],       "p_away_rf" : p_a_rf[idx],
            "p_home_bay": p_h_bay_list[idx], "p_draw_bay": p_d_bay_list[idx], "p_away_bay": p_a_bay_list[idx],
        }
        row = df.iloc[idx]
        for feat in ["delta_elo", "delta_form", "delta_xg", "delta_fifa_rank", "delta_sv_log", "delta_rest", "expected_home_elo"]:
            val = row.get(feat, np.nan)
            meta_row[feat] = float(val) if not pd.isna(val) else 0.0
        meta_rows.append(meta_row)

    df_meta = pd.DataFrame(meta_rows)
    meta = models.get("meta")
    if meta and meta.is_fitted:
        p_meta = meta.predict_proba_df(df_meta)
        meta_h = p_meta["p_home_win"].values
        meta_d = p_meta["p_draw"].values
        meta_a = p_meta["p_away_win"].values
    else:
        meta_h = np.mean([p_h_dc_list, p_h_elo, p_h_rf, p_h_bay_list], axis=0)
        meta_d = np.mean([p_d_dc_list, p_d_elo, p_d_rf, p_d_bay_list], axis=0)
        meta_a = np.mean([p_a_dc_list, p_a_elo, p_a_rf, p_a_bay_list], axis=0)

    # 6. Calibración en lote (batch)
    cal = models.get("cal")
    ensemble_raw = np.column_stack([meta_h, meta_d, meta_a])
    if cal and cal.is_fitted:
        ensemble_final = cal.transform(ensemble_raw)
    else:
        ensemble_final = ensemble_raw

    # Guardar en cache para accesos O(1)
    for idx, (h, a, is_ko) in enumerate(keys):
        probs = ensemble_final[idx]
        s = probs.sum()
        if s > 0:
            probs = probs / s
        
        # Guardar en ambos sentidos
        _MATCH_PROB_CACHE[(h, a, is_ko)] = probs
        _MATCH_PROB_CACHE[(a, h, is_ko)] = np.array([probs[2], probs[1], probs[0]])


def get_models() -> Dict:
    """Carga los modelos del ensemble una sola vez por proceso."""
    global _LOADED_MODELS
    if _LOADED_MODELS is not None:
        return _LOADED_MODELS

    import pickle

    models = {}
    model_files = {
        "dc" : DATA_MODEL / "dixon_coles.pkl",
        "bay": DATA_MODEL / "bayesian_map.pkl",
        "elo": DATA_MODEL / "elo_logistic.pkl",
        "rf" : DATA_MODEL / "random_forest.pkl",
        "meta": DATA_MODEL / "meta_xgboost.pkl",
        "cal" : DATA_MODEL / "calibrator.pkl",
    }
    for key, path in model_files.items():
        if path.exists():
            try:
                with open(path, "rb") as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                print(f"  WARN: no se pudo cargar {key}: {e}")
    _LOADED_MODELS = models
    
    # Precalcular todas las probabilidades para la simulación
    try:
        print("  Precalculando todas las probabilidades ensemble (2,256 combinaciones)...")
        t_start = time.time()
        precompute_all_match_probabilities(models)
        print(f"  [OK] Precalculo completado en {time.time() - t_start:.2f}s")
    except Exception as ex:
        print(f"  WARN: no se pudo precalcular las probabilidades: {ex}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Carga lazy de los snapshots estáticos (una sola vez por proceso)
# ─────────────────────────────────────────────────────────────────────────────

_STATIC_DATA: Optional[Dict] = None


def _load_static_data() -> Dict:
    """Carga Elo actual, FIFA ranking, valor de plantilla y forma reciente."""
    global _STATIC_DATA
    if _STATIC_DATA is not None:
        return _STATIC_DATA

    data: Dict = {}

    # Elo actual por equipo
    elo_path = DATA_FEATURES / "elo_current.parquet"
    if not elo_path.exists():
        elo_path = DATA_RAW / "elo_current.parquet"
    if elo_path.exists():
        df = pd.read_parquet(elo_path)
        col = "elo_current"
        data["elo"] = dict(zip(df["team"], df[col]))
    else:
        data["elo"] = {}

    # FIFA ranking (snapshot más reciente por equipo)
    fifa_path = DATA_RAW / "fifa_rankings.parquet"
    if fifa_path.exists():
        df = pd.read_parquet(fifa_path)
        df["date"] = pd.to_datetime(df.get("date", pd.Timestamp.now()))
        snap = df.sort_values("date").groupby("team").last()
        data["fifa_rank"]   = snap["fifa_rank"].to_dict()   if "fifa_rank" in snap else {}
        data["fifa_points"] = snap["fifa_points"].to_dict() if "fifa_points" in snap else {}
    else:
        data["fifa_rank"], data["fifa_points"] = {}, {}

    # Valor de plantilla (log)
    sv_path = DATA_RAW / "squad_values.parquet"
    if sv_path.exists():
        df = pd.read_parquet(sv_path)
        if "squad_value_log" in df.columns:
            data["sv_log"] = dict(zip(df["team"], df["squad_value_log"]))
        elif "squad_value_eur" in df.columns:
            data["sv_log"] = {
                t: math.log(v) if v and v > 0 else np.nan
                for t, v in zip(df["team"], df["squad_value_eur"])
            }
        else:
            data["sv_log"] = {}
    else:
        data["sv_log"] = {}

    # Forma reciente (última entrada por equipo)
    form_path = DATA_FEATURES / "form_rolling.parquet"
    if form_path.exists():
        df = pd.read_parquet(form_path)
        df["date"] = pd.to_datetime(df["date"])
        snap = df.sort_values("date").groupby("team").last()
        data["form_weighted"]   = snap["form_weighted"].to_dict()   if "form_weighted" in snap else {}
        data["form_pts_last6"]  = snap["form_pts_last6"].to_dict()  if "form_pts_last6" in snap else {}
        data["form_gf_avg"]     = snap["form_gf_avg"].to_dict()     if "form_gf_avg" in snap else {}
        data["form_ga_avg"]     = snap["form_ga_avg"].to_dict()     if "form_ga_avg" in snap else {}
        data["momentum_trend"]  = snap["momentum_trend"].to_dict()  if "momentum_trend" in snap else {}
    else:
        data["form_weighted"] = data["form_pts_last6"] = {}
        data["form_gf_avg"] = data["form_ga_avg"] = data["momentum_trend"] = {}

    # xG reciente (promedio de los últimos partidos disponibles en StatsBomb)
    xg_path = DATA_FEATURES / "xg_derived.parquet"
    if xg_path.exists():
        df = pd.read_parquet(xg_path)
        agg = df.groupby("team").agg(xg=("xg", "mean"), xga=("xga", "mean"))
        data["xg_avg"]  = agg["xg"].to_dict()
        data["xga_avg"] = agg["xga"].to_dict()
    else:
        data["xg_avg"], data["xga_avg"] = {}, {}

    # ── xG EN VIVO del WC2026 (Sofascore) — prioridad sobre el histórico ─────
    live_path = DATA_FEATURES / "wc2026_live_snapshot.parquet"
    if live_path.exists():
        df = pd.read_parquet(live_path)
        if "xg_avg" in df.columns:
            data["wc2026_xg_avg"] = dict(zip(df["team"], df["xg_avg"]))
        else:
            data["wc2026_xg_avg"] = {}
        if "xga_avg" in df.columns:
            data["wc2026_xga_avg"] = dict(zip(df["team"], df["xga_avg"]))
        else:
            data["wc2026_xga_avg"] = {}
        if "gf_avg" in df.columns:
            data["wc2026_gf_avg"] = dict(zip(df["team"], df["gf_avg"]))
        else:
            data["wc2026_gf_avg"] = {}
        if "ga_avg" in df.columns:
            data["wc2026_ga_avg"] = dict(zip(df["team"], df["ga_avg"]))
        else:
            data["wc2026_ga_avg"] = {}
        print(f"  [OK] xG en vivo WC2026 cargado: {len(data['wc2026_xg_avg'])} equipos")
    else:
        data["wc2026_xg_avg"]  = {}
        data["wc2026_xga_avg"] = {}
        data["wc2026_gf_avg"]  = {}
        data["wc2026_ga_avg"]  = {}

    _STATIC_DATA = data
    return data


def build_static_row(home: str, away: str, is_knockout: bool = False) -> Dict:
    """
    Construye un dict de features 'as-of-now' para un partido home vs away,
    usando los snapshots más recientes disponibles (Elo, FIFA, plantilla,
    forma, xG). Pensado para la simulación Monte Carlo, donde no hay una
    fecha concreta de partido como en el training set.
    """
    d = _load_static_data()

    elo_h = d["elo"].get(home, 1500.0)
    elo_a = d["elo"].get(away, 1500.0)
    # Campo neutral → sin ventaja de localía
    expected_home_elo = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))

    fifa_rank_h = d["fifa_rank"].get(home, np.nan)
    fifa_rank_a = d["fifa_rank"].get(away, np.nan)
    fifa_pts_h  = d["fifa_points"].get(home, np.nan)
    fifa_pts_a  = d["fifa_points"].get(away, np.nan)

    sv_h = d["sv_log"].get(home, np.nan)
    sv_a = d["sv_log"].get(away, np.nan)

    form_h = d["form_weighted"].get(home, np.nan)
    form_a = d["form_weighted"].get(away, np.nan)
    pts6_h = d["form_pts_last6"].get(home, np.nan)
    pts6_a = d["form_pts_last6"].get(away, np.nan)
    gf_h   = d["form_gf_avg"].get(home, np.nan)
    gf_a   = d["form_gf_avg"].get(away, np.nan)
    ga_h   = d["form_ga_avg"].get(home, np.nan)
    ga_a   = d["form_ga_avg"].get(away, np.nan)

    # xG: priorizar dato EN VIVO del WC2026, caer al histórico si falta
    xg_h  = d["wc2026_xg_avg"].get(home,  d["xg_avg"].get(home, np.nan))
    xg_a  = d["wc2026_xg_avg"].get(away,  d["xg_avg"].get(away, np.nan))
    xga_h = d["wc2026_xga_avg"].get(home, d["xga_avg"].get(home, np.nan))
    xga_a = d["wc2026_xga_avg"].get(away, d["xga_avg"].get(away, np.nan))

    row = {
        "home_team": home, "away_team": away,
        "elo_home_pre": elo_h, "elo_away_pre": elo_a,
        "expected_home_elo": expected_home_elo,
        "delta_elo": elo_h - elo_a,

        "home_fifa_rank": fifa_rank_h, "away_fifa_rank": fifa_rank_a,
        "delta_fifa_rank": (fifa_rank_a - fifa_rank_h)
            if not (np.isnan(fifa_rank_h) or np.isnan(fifa_rank_a)) else np.nan,
        "delta_fifa_pts": (fifa_pts_h - fifa_pts_a)
            if not (np.isnan(fifa_pts_h) or np.isnan(fifa_pts_a)) else np.nan,

        "delta_sv_log": (sv_h - sv_a) if not (np.isnan(sv_h) or np.isnan(sv_a)) else np.nan,

        "home_form_weighted": form_h, "away_form_weighted": form_a,
        "delta_form": (form_h - form_a) if not (np.isnan(form_h) or np.isnan(form_a)) else np.nan,
        "delta_form_pts6": (pts6_h - pts6_a) if not (np.isnan(pts6_h) or np.isnan(pts6_a)) else np.nan,
        "delta_gf_avg": (gf_h - gf_a) if not (np.isnan(gf_h) or np.isnan(gf_a)) else np.nan,
        "delta_ga_avg": (ga_h - ga_a) if not (np.isnan(ga_h) or np.isnan(ga_a)) else np.nan,
        "home_momentum_trend": d["momentum_trend"].get(home, np.nan),
        "away_momentum_trend": d["momentum_trend"].get(away, np.nan),

        "delta_xg": (xg_h - xg_a) if not (np.isnan(xg_h) or np.isnan(xg_a)) else np.nan,
        "delta_xga": (xga_h - xga_a) if not (np.isnan(xga_h) or np.isnan(xga_a)) else np.nan,

        "home_is_knockout": int(is_knockout),
        "home_tournament_weight": 1.0,   # Mundial = peso máximo
        "home_days_rest": np.nan, "away_days_rest": np.nan,
        "delta_rest": 0.0, "delta_fatigue": 0.0,

        "h2h_win_rate": np.nan, "h2h_gd_avg": np.nan, "h2h_matches": 0,
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Obtener lambdas de Poisson para una pareja de equipos
# ─────────────────────────────────────────────────────────────────────────────

def get_poisson_lambdas(home: str, away: str,
                        models: Optional[Dict] = None) -> Tuple[float, float]:
    """
    Devuelve (λ_home, λ_away) goles esperados para el partido.
    Usa cache para no recalcular el mismo par en cada simulación.
    Prioriza Dixon-Coles; si no, usa Bayesian MAP; si no, heurística Elo.
    """
    global _LAMBDA_CACHE
    key = (home, away)
    if key in _LAMBDA_CACHE:
        return _LAMBDA_CACHE[key]

    if models is None:
        models = get_models()

    result = (1.3, 1.0)  # fallback neutro

    # Dixon-Coles (primer candidato)
    if "dc" in models:
        try:
            pred = models["dc"].predict_match(home, away)
            lh = float(pred.get("lambda_home", 1.3))
            la = float(pred.get("lambda_away", 1.0))
            if 0.1 < lh < 10 and 0.1 < la < 10:
                result = (lh, la)
                _LAMBDA_CACHE[key] = result
                return result
        except Exception:
            pass

    # Bayesian MAP (segundo candidato)
    if "bay" in models:
        try:
            pred = models["bay"].predict_match(home, away)
            lh = float(pred.get("lambda_home", 1.3))
            la = float(pred.get("lambda_away", 1.0))
            if 0.1 < lh < 10 and 0.1 < la < 10:
                result = (lh, la)
                _LAMBDA_CACHE[key] = result
                return result
        except Exception:
            pass

    _LAMBDA_CACHE[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Obtener probabilidades 1X2 del ensemble
# ─────────────────────────────────────────────────────────────────────────────

def get_match_probabilities(home: str, away: str,
                            models: Optional[Dict] = None,
                            is_knockout: bool = False) -> np.ndarray:
    """
    Devuelve [p_home, p_draw, p_away] del ensemble calibrado.
    Usa el cache de precalculados para velocidad O(1).
    """
    global _MATCH_PROB_CACHE
    key = (home, away, is_knockout)
    if key in _MATCH_PROB_CACHE:
        return _MATCH_PROB_CACHE[key]

    if models is None:
        models = get_models()

    try:
        from models.ensemble import predict_match
        row = build_static_row(home, away, is_knockout=is_knockout)
        pred = predict_match(home, away, row=row, models=models, verbose=False)
        return np.array([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])
    except Exception:
        pass

    # Fallback: Poisson numérico
    lh, la = get_poisson_lambdas(home, away, models)
    ph, pd_, pa = poisson_outcome_probs(lh, la)
    return np.array([ph, pd_, pa])


# ─────────────────────────────────────────────────────────────────────────────
# Distribución de Poisson bivariada (Dixon-Coles style)
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_pmf_vec(lam: float, max_goals: int = _MAX_GOALS) -> np.ndarray:
    """
    PMF de Poisson vectorizada con numpy puro (sin scipy).
    Retorna array de longitud (max_goals+1).
    """
    k = np.arange(max_goals + 1, dtype=np.float64)
    log_pmf = k * np.log(max(lam, 1e-10)) - lam - np.log(_FACTORIALS[:max_goals + 1])
    return np.exp(log_pmf)


def _build_dc_table(lh: float, la: float,
                    rho: float = -0.13) -> np.ndarray:
    """
    Construye la tabla de probabilidades conjuntas Dixon-Coles con numpy puro.
    Retorna matriz (_MAX_GOALS+1) × (_MAX_GOALS+1).
    """
    ph_vec = _poisson_pmf_vec(lh)
    pa_vec = _poisson_pmf_vec(la)
    table  = np.outer(ph_vec, pa_vec)

    # Corrección DC para marcadores bajos (sólo 4 celdas)
    corr = np.array([
        [1.0 - lh * la * rho,  1.0 + lh * rho],
        [1.0 + la * rho,       1.0 - rho      ],
    ])
    table[:2, :2] *= np.maximum(corr, 1e-9)

    table = np.maximum(table, 0.0)
    s = table.sum()
    if s > 0:
        table /= s
    return table


def poisson_outcome_probs(lh: float, la: float,
                          rho: float = -0.13,
                          max_goals: int = _MAX_GOALS) -> Tuple[float, float, float]:
    """
    Calcula P(H), P(D), P(A) — numpy puro, sin scipy.
    """
    table = _build_dc_table(lh, la, rho)
    n = table.shape[0]
    idx = np.arange(n)
    mask_h = idx[:, None] > idx[None, :]   # gh > ga
    mask_d = idx[:, None] == idx[None, :]  # gh == ga
    p_h = float(table[mask_h].sum())
    p_d = float(table[mask_d].sum())
    p_a = float(1.0 - p_h - p_d)
    if p_a < 0:
        p_a = 0.0
    total = p_h + p_d + p_a
    if total < 1e-9:
        return 1/3, 1/3, 1/3
    return p_h / total, p_d / total, p_a / total


def sample_scoreline(lh: float, la: float,
                     rho: float = -0.13,
                     rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
    """
    Muestrea un marcador desde Poisson bivariada con corrección DC.
    Numpy puro — sin scipy en el hot-path.
    """
    table      = _build_dc_table(lh, la, rho)
    probs_flat = table.flatten()
    n          = table.shape[0]

    if rng is not None:
        idx = rng.choice(len(probs_flat), p=probs_flat)
    else:
        idx = np.random.choice(len(probs_flat), p=probs_flat)

    home_goals = int(idx // n)
    away_goals = int(idx %  n)
    return home_goals, away_goals


# ─────────────────────────────────────────────────────────────────────────────
# Calibración de Lambdas al Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_lambdas_to_ensemble(lh_dc, la_dc, target_p_home, target_p_away,
                                   max_iter=40):
    """
    Mantiene el total de goles esperados de Dixon-Coles (realista),
    pero redistribuye el split home/away para que el resultado 1X2
    coincida con la probabilidad del ensemble calibrado.
    """
    total = lh_dc + la_dc
    if total <= 0:
        return lh_dc, la_dc
    lo, hi = 0.02, 0.98
    target_diff = target_p_home - target_p_away
    mid = 0.5
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        lh, la = total * mid, total * (1 - mid)
        ph, _, pa = poisson_outcome_probs(lh, la)
        if (ph - pa) < target_diff:
            lo = mid
        else:
            hi = mid
    return total * mid, total * (1 - mid)


_LAMBDA_CACHE_CAL: Dict[Tuple[str, str], Tuple[float, float]] = {}

def get_calibrated_poisson_lambdas(home: str, away: str,
                                    models: Optional[Dict] = None,
                                    is_knockout: bool = False) -> Tuple[float, float]:
    """Lambdas de DC (forma realista) recalibradas al 1X2 del ensemble completo."""
    global _LAMBDA_CACHE_CAL
    key = (home, away)
    if key in _LAMBDA_CACHE_CAL:
        return _LAMBDA_CACHE_CAL[key]

    if models is None:
        models = get_models()

    lh_dc, la_dc = get_poisson_lambdas(home, away, models)
    p_h, p_d, p_a = get_match_probabilities(home, away, models, is_knockout=is_knockout)
    lh, la = calibrate_lambdas_to_ensemble(lh_dc, la_dc, p_h, p_a)

    _LAMBDA_CACHE_CAL[key] = (lh, la)
    return lh, la


# ─────────────────────────────────────────────────────────────────────────────
# Simulación de un partido (fase de grupos — sin ET)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_group_match(home: str, away: str,
                         models: Optional[Dict] = None,
                         rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Simula un partido de fase de grupos.
    Devuelve: {home_team, away_team, home_goals, away_goals, result}
    result: 'H' | 'D' | 'A'
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    lh, la = get_calibrated_poisson_lambdas(home, away, models, is_knockout=False)
    hg, ag = sample_scoreline(lh, la, rng=rng)

    result = "H" if hg > ag else ("D" if hg == ag else "A")
    return {
        "home_team"  : home,
        "away_team"  : away,
        "home_goals" : hg,
        "away_goals" : ag,
        "result"     : result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulación de un partido eliminatorio (con ET y penaltis)
# ─────────────────────────────────────────────────────────────────────────────

PENALTY_WIN_PROB = 0.5   # 50-50 en penaltis (modelo simétrico)

def simulate_knockout_match(team1: str, team2: str,
                            models: Optional[Dict] = None,
                            rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Simula un partido eliminatorio.
    Si hay empate al final del tiempo reglamentario → Tiempo Extra + Penaltis.

    Devuelve:
      {team1, team2, goals_team1, goals_team2, winner, went_to_et, went_to_penalties}
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    lh, la = get_calibrated_poisson_lambdas(team1, team2, models, is_knockout=True)
    hg, ag = sample_scoreline(lh, la, rng=rng)

    went_to_et      = False
    went_to_pens    = False
    et_home_goals   = 0
    et_away_goals   = 0

    if hg == ag:
        # ── Tiempo Extra ──────────────────────────────────────────────────────
        went_to_et = True
        # En la prórroga la media de goles es ~0.35 por equipo (estadística histórica)
        et_lh = lh * (30 / 90)   # 30 minutos adicionales
        et_la = la * (30 / 90)
        et_hg = rng.poisson(et_lh)
        et_ag = rng.poisson(et_la)
        hg += et_hg
        ag += et_ag
        et_home_goals = et_hg
        et_away_goals = et_ag

        if hg == ag:
            # ── Penaltis ──────────────────────────────────────────────────────
            went_to_pens = True
            # team1 gana con probabilidad P_penalty_win
            # Ajustamos por Elo si disponible
            p1_wins = PENALTY_WIN_PROB
            if "elo" in models and hasattr(models["elo"], "get_elo"):
                try:
                    e1 = models["elo"].get_elo(team1)
                    e2 = models["elo"].get_elo(team2)
                    p1_wins = 1 / (1 + 10 ** ((e2 - e1) / 1500))
                    p1_wins = np.clip(0.3 + 0.4 * p1_wins, 0.3, 0.7)
                except Exception:
                    pass

            winner = team1 if rng.random() < p1_wins else team2
        else:
            winner = team1 if hg > ag else team2
    else:
        winner = team1 if hg > ag else team2

    return {
        "team1"             : team1,
        "team2"             : team2,
        "goals_team1"       : hg,
        "goals_team2"       : ag,
        "et_goals_team1"    : et_home_goals,
        "et_goals_team2"    : et_away_goals,
        "winner"            : winner,
        "loser"             : team2 if winner == team1 else team1,
        "went_to_et"        : went_to_et,
        "went_to_penalties" : went_to_pens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test rápido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    print("=== Test Match Simulator ===")
    models = get_models()
    print(f"Modelos disponibles: {list(models.keys())}")

    np.random.seed(42)
    rng = np.random.default_rng(42)

    home, away = "Argentina", "France"
    print(f"\n[Grupo] {home} vs {away}")
    for _ in range(3):
        r = simulate_group_match(home, away, models, rng)
        print(f"  {r['home_goals']}–{r['away_goals']} ({r['result']})")

    print(f"\n[Eliminatoria] {home} vs {away}")
    for _ in range(3):
        r = simulate_knockout_match(home, away, models, rng)
        et = " [ET]" if r["went_to_et"] else ""
        p = " [PEN]" if r["went_to_penalties"] else ""
        print(f"  {r['goals_team1']}–{r['goals_team2']}{et}{p} → Gana: {r['winner']}")
