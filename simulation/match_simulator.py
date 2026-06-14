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
  - Soporta 100 000 simulaciones de todo el torneo en < 5 min
    mediante vectorización numpy donde sea posible.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Singleton de modelos
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_MODELS: Optional[Dict] = None


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
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Obtener lambdas de Poisson para una pareja de equipos
# ─────────────────────────────────────────────────────────────────────────────

def get_poisson_lambdas(home: str, away: str,
                        models: Optional[Dict] = None) -> Tuple[float, float]:
    """
    Devuelve (λ_home, λ_away) goles esperados para el partido.
    Prioriza Dixon-Coles; si no, usa Bayesian MAP; si no, heurística Elo.
    """
    if models is None:
        models = get_models()

    # Dixon-Coles (primer candidato)
    if "dc" in models:
        try:
            pred = models["dc"].predict_match(home, away)
            lh = float(pred.get("lambda_home", 1.3))
            la = float(pred.get("lambda_away", 1.0))
            if 0.1 < lh < 10 and 0.1 < la < 10:
                return lh, la
        except Exception:
            pass

    # Bayesian MAP (segundo candidato)
    if "bay" in models:
        try:
            pred = models["bay"].predict_match(home, away)
            lh = float(pred.get("lambda_home", 1.3))
            la = float(pred.get("lambda_away", 1.0))
            if 0.1 < lh < 10 and 0.1 < la < 10:
                return lh, la
        except Exception:
            pass

    # Fallback neutro
    return 1.3, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Obtener probabilidades 1X2 del ensemble
# ─────────────────────────────────────────────────────────────────────────────

def get_match_probabilities(home: str, away: str,
                            models: Optional[Dict] = None) -> np.ndarray:
    """
    Devuelve [p_home, p_draw, p_away] del ensemble calibrado.
    """
    if models is None:
        models = get_models()

    try:
        from models.ensemble import predict_match
        pred = predict_match(home, away, models=models, verbose=False)
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

def poisson_outcome_probs(lh: float, la: float,
                          rho: float = -0.13,
                          max_goals: int = 8) -> Tuple[float, float, float]:
    """
    Calcula P(H), P(D), P(A) integrando la distribución de Poisson
    bivariada con corrección de Dixon-Coles (rho) para marcadores 0-0/1-0/0-1/1-1.
    """
    from scipy.stats import poisson

    p_h = p_d = p_a = 0.0
    for gh in range(max_goals + 1):
        for ga in range(max_goals + 1):
            p = poisson.pmf(gh, lh) * poisson.pmf(ga, la)
            # Corrección de baja puntuación
            if gh == 0 and ga == 0:
                tau = 1.0 - lh * la * rho
            elif gh == 1 and ga == 0:
                tau = 1.0 + la * rho
            elif gh == 0 and ga == 1:
                tau = 1.0 + lh * rho
            elif gh == 1 and ga == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            p *= max(tau, 1e-9)
            if gh > ga:
                p_h += p
            elif gh == ga:
                p_d += p
            else:
                p_a += p

    total = p_h + p_d + p_a
    if total < 1e-9:
        return 1/3, 1/3, 1/3
    return p_h / total, p_d / total, p_a / total


def sample_scoreline(lh: float, la: float,
                     rho: float = -0.13,
                     max_goals: int = 8) -> Tuple[int, int]:
    """
    Muestrea un marcador (home_goals, away_goals) desde la distribución
    de Poisson bivariada con corrección Dixon-Coles.
    Más eficiente que iterar, usa numpy vectorizado.
    """
    g = np.arange(max_goals + 1)
    # Probabilidades marginales Poisson
    from scipy.stats import poisson
    ph_vec = poisson.pmf(g, lh)
    pa_vec = poisson.pmf(g, la)
    # Tabla de probabilidades conjuntas
    table = np.outer(ph_vec, pa_vec)  # (max_goals+1) × (max_goals+1)

    # Corrección DC para marcadores bajos
    for gh in range(min(2, max_goals + 1)):
        for ga in range(min(2, max_goals + 1)):
            if gh == 0 and ga == 0:
                tau = 1.0 - lh * la * rho
            elif gh == 1 and ga == 0:
                tau = 1.0 + la * rho
            elif gh == 0 and ga == 1:
                tau = 1.0 + lh * rho
            elif gh == 1 and ga == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            table[gh, ga] *= max(tau, 1e-9)

    table = np.maximum(table, 0)
    probs_flat = table.flatten()
    probs_flat /= probs_flat.sum()

    idx = np.random.choice(len(probs_flat), p=probs_flat)
    home_goals = idx // (max_goals + 1)
    away_goals = idx %  (max_goals + 1)
    return int(home_goals), int(away_goals)


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

    lh, la = get_poisson_lambdas(home, away, models)
    hg, ag = sample_scoreline(lh, la)

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

    lh, la = get_poisson_lambdas(team1, team2, models)
    hg, ag = sample_scoreline(lh, la)

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
