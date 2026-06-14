"""
dixon_coles.py — Modelo Dixon-Coles (Poisson Bivariada)
========================================================
Referencia: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market."

El modelo estima parámetros de ataque/defensa por equipo y calcula:
  - λ_home, λ_away  (tasas esperadas de goles)
  - Matriz de probabilidad de marcadores (hasta MAX_GOALS × MAX_GOALS)
  - P(Victoria local), P(Empate), P(Victoria visitante)
  - P(Over/Under N.5 goles)
  - P(BTTS - ambos marcan)
  - Marcador más probable

El parámetro ρ (rho) corrige la sobreestimación de resultados bajos
(0-0, 1-0, 0-1, 1-1) que Poisson independiente sobreestima.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_SET, VALIDATION_SET, DATA_MODEL, ensure_dirs

MAX_GOALS = 8   # Máximo de goles por equipo considerado en la matriz
MIN_MATCHES = 3  # Mínimo de partidos para tener parámetro propio


# ─────────────────────────────────────────────────────────────────────────────
# Función de corrección τ (tau) de Dixon-Coles
# ─────────────────────────────────────────────────────────────────────────────

def tau(home_goals: int, away_goals: int,
        lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Corrección τ para marcadores bajos.
    Ajusta la sobreestimación que produce Poisson independiente
    en los marcadores 0-0, 1-0, 0-1, 1-1.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    else:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Probabilidad conjunta Dixon-Coles de un marcador (x, y)
# ─────────────────────────────────────────────────────────────────────────────

def dc_score_prob(home_goals: int, away_goals: int,
                  lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    P(goles_local=x, goles_visitante=y) con corrección Dixon-Coles.
    """
    t  = tau(home_goals, away_goals, lambda_h, lambda_a, rho)
    ph = poisson.pmf(home_goals, lambda_h)
    pa = poisson.pmf(away_goals, lambda_a)
    return max(t * ph * pa, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Log-verosimilitud negativa del modelo (para minimizar)
# ─────────────────────────────────────────────────────────────────────────────

def neg_log_likelihood(params: np.ndarray,
                       home_idx: np.ndarray, away_idx: np.ndarray,
                       home_goals: np.ndarray, away_goals: np.ndarray,
                       n_teams: int,
                       weight: Optional[np.ndarray] = None) -> float:
    """
    Log-verosimilitud negativa de todo el dataset bajo Dixon-Coles (versión vectorizada).
    """
    from scipy.special import loggamma

    attack  = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    intercept = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    lambda_h = np.exp(intercept + attack[home_idx] + defense[away_idx])
    lambda_a = np.exp(intercept + attack[away_idx] + defense[home_idx])

    # Evitar desbordamientos o valores nulos
    lambda_h = np.clip(lambda_h, 1e-10, None)
    lambda_a = np.clip(lambda_a, 1e-10, None)

    # Log Poisson PMF: k * log(mu) - mu - loggamma(k + 1)
    log_pmf_h = home_goals * np.log(lambda_h) - lambda_h - loggamma(home_goals + 1)
    log_pmf_a = away_goals * np.log(lambda_a) - lambda_a - loggamma(away_goals + 1)

    # Corrección Dixon-Coles tau
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    tau_vals = np.ones_like(lambda_h)
    tau_vals[mask_00] = 1.0 - lambda_h[mask_00] * lambda_a[mask_00] * rho
    tau_vals[mask_01] = 1.0 + lambda_h[mask_01] * rho
    tau_vals[mask_10] = 1.0 + lambda_a[mask_10] * rho
    tau_vals[mask_11] = 1.0 - rho

    tau_vals = np.clip(tau_vals, 1e-10, None)

    log_prob = log_pmf_h + log_pmf_a + np.log(tau_vals)

    if weight is not None:
        log_prob = log_prob * weight

    return float(-np.sum(log_prob))


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal DixonColesModel
# ─────────────────────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles ajustado por máxima verosimilitud (MLE).
    Permite:
      - fit(df): ajustar sobre datos históricos
      - predict_match(home, away): devuelve probas 1X2, O/U, BTTS, etc.
      - score_matrix(home, away): matriz MAX_GOALS × MAX_GOALS
    """

    def __init__(self, rho_init: float = -0.10, max_goals: int = MAX_GOALS):
        self.rho_init  = rho_init
        self.max_goals = max_goals
        self.teams_     : list  = []
        self.team_idx_  : Dict  = {}
        self.attack_    : Dict  = {}
        self.defense_   : Dict  = {}
        self.intercept_ : float = 0.0
        self.rho_       : float = rho_init
        self.is_fitted   = False

    # ── Ajuste del modelo ─────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame,
            home_col: str = "home_team", away_col: str = "away_team",
            home_goals_col: str = "home_score", away_goals_col: str = "away_score",
            weight_col: Optional[str] = None,
            maxiter: int = 150) -> "DixonColesModel":
        """
        Ajusta los parámetros de ataque/defensa por MLE.

        df: DataFrame con columnas de equipos, goles y (opcionalmente) pesos.
        weight_col: columna de pesos temporales (ej. decaimiento exponencial).
        """
        df = df.copy()
        df[home_goals_col] = pd.to_numeric(df[home_goals_col], errors="coerce")
        df[away_goals_col] = pd.to_numeric(df[away_goals_col], errors="coerce")
        df = df.dropna(subset=[home_goals_col, away_goals_col])
        df = df[(df[home_goals_col] >= 0) & (df[away_goals_col] >= 0)]

        # Filtrar equipos con al menos MIN_MATCHES partidos
        all_teams = pd.concat([df[home_col], df[away_col]])
        valid_teams = all_teams.value_counts()
        valid_teams = valid_teams[valid_teams >= MIN_MATCHES].index.tolist()
        df = df[df[home_col].isin(valid_teams) & df[away_col].isin(valid_teams)]

        self.teams_    = sorted(list(set(df[home_col]).union(set(df[away_col]))))
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        n_teams        = len(self.teams_)

        print(f"  DixonColes: {len(df):,} partidos | {n_teams} equipos")

        home_idx  = np.array([self.team_idx_[t] for t in df[home_col]])
        away_idx  = np.array([self.team_idx_[t] for t in df[away_col]])
        home_gls  = df[home_goals_col].values.astype(int)
        away_gls  = df[away_goals_col].values.astype(int)
        weights   = df[weight_col].values if weight_col and weight_col in df.columns else None

        # Inicialización: ataque=0.1, defensa=0.0, intercept=0.3, rho=rho_init
        x0 = np.concatenate([
            np.full(n_teams, 0.1),   # ataque
            np.zeros(n_teams),        # defensa
            [0.3],                    # intercepto
            [self.rho_init]           # rho
        ])

        # Optimización sin restricciones (mucho más rápida con L-BFGS-B en altas dimensiones)
        result = minimize(
            neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_gls, away_gls, n_teams, weights),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-5}
        )

        if not result.success:
            print(f"  ADVERTENCIA: MLE no convergió. {result.message}")

        params = result.x

        # Normalización para identifiabilidad → suma(ataque) = 0
        # Dado que lambda_h = exp(intercept + attack_h + defense_a),
        # podemos restar la media a los ataques y sumarla al intercepto sin alterar las tasas.
        attack_raw = params[:n_teams]
        defense_raw = params[n_teams:2 * n_teams]
        intercept_raw = params[2 * n_teams]

        attack_mean = float(np.mean(attack_raw))
        attack_norm = attack_raw - attack_mean
        intercept_norm = intercept_raw + attack_mean

        for i, team in enumerate(self.teams_):
            self.attack_[team]  = float(attack_norm[i])
            self.defense_[team] = float(defense_raw[i])
        self.intercept_ = float(intercept_norm)
        self.rho_       = float(params[2 * n_teams + 1])
        self.is_fitted   = True

        print(f"  rho={self.rho_:.4f} | intercept={self.intercept_:.4f} | "
              f"NLL={result.fun:.2f}")
        return self

    # ── Tasas esperadas de goles ──────────────────────────────────────────────

    def _lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Calcula λ_home y λ_away para el partido dado."""
        att_h = self.attack_.get(home_team,  np.mean(list(self.attack_.values())))
        att_a = self.attack_.get(away_team,  np.mean(list(self.attack_.values())))
        def_h = self.defense_.get(home_team, np.mean(list(self.defense_.values())))
        def_a = self.defense_.get(away_team, np.mean(list(self.defense_.values())))

        lambda_h = np.exp(self.intercept_ + att_h + def_a)
        lambda_a = np.exp(self.intercept_ + att_a + def_h)
        return lambda_h, lambda_a

    # ── Matriz de marcadores ──────────────────────────────────────────────────

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Devuelve matriz (max_goals+1) × (max_goals+1) de probabilidades
        de marcadores P(goles_local=i, goles_visitante=j).
        """
        assert self.is_fitted, "Modelo no ajustado. Ejecutar .fit() primero."
        lh, la = self._lambdas(home_team, away_team)
        G = self.max_goals + 1
        matrix = np.zeros((G, G))
        for i in range(G):
            for j in range(G):
                matrix[i, j] = dc_score_prob(i, j, lh, la, self.rho_)
        # Normalizar para que sume 1
        matrix /= matrix.sum()
        return matrix

    # ── Predicción de partido ─────────────────────────────────────────────────

    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """
        Devuelve diccionario con todas las probabilidades del partido.
        """
        assert self.is_fitted
        lh, la = self._lambdas(home_team, away_team)
        mat = self.score_matrix(home_team, away_team)

        # 1X2
        p_home = float(np.sum(np.tril(mat, -1)))   # goles_h > goles_a
        p_draw = float(np.sum(np.diag(mat)))
        p_away = float(np.sum(np.triu(mat, 1)))

        # Over/Under
        G = self.max_goals + 1
        total = np.array([[i + j for j in range(G)] for i in range(G)])
        p_over15  = float(mat[total > 1.5].sum())
        p_over25  = float(mat[total > 2.5].sum())
        p_over35  = float(mat[total > 3.5].sum())
        p_under25 = 1.0 - p_over25

        # BTTS
        p_btts = float(mat[1:, 1:].sum())

        # Marcador más probable
        idx = np.unravel_index(mat.argmax(), mat.shape)
        most_likely_score = f"{idx[0]}-{idx[1]}"

        return {
            "home_team"          : home_team,
            "away_team"          : away_team,
            "lambda_home"        : round(lh, 4),
            "lambda_away"        : round(la, 4),
            "p_home_win"         : round(p_home, 4),
            "p_draw"             : round(p_draw, 4),
            "p_away_win"         : round(p_away, 4),
            "p_over15"           : round(p_over15, 4),
            "p_over25"           : round(p_over25, 4),
            "p_over35"           : round(p_over35, 4),
            "p_under25"          : round(p_under25, 4),
            "p_btts"             : round(p_btts, 4),
            "most_likely_score"  : most_likely_score,
            "model"              : "DixonColes",
        }

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save(self, path: Path):
        """Guarda parámetros del modelo en parquet."""
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado: {path}")

    @staticmethod
    def load(path: Path) -> "DixonColesModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Función de entrenamiento y evaluación rápida
# ─────────────────────────────────────────────────────────────────────────────

def add_time_weights(df: pd.DataFrame, half_life_days: int = 365) -> pd.DataFrame:
    """
    Añade columna 'time_weight' con decaimiento exponencial por antigüedad.
    Partidos recientes tienen mayor peso.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    days_ago = (max_date - df["date"]).dt.days
    df["time_weight"] = np.exp(-np.log(2) * days_ago / half_life_days)
    return df


def train_dixon_coles(half_life_days: int = 730) -> DixonColesModel:
    """
    Carga datos de entrenamiento, ajusta y guarda el modelo.
    """
    ensure_dirs()
    print("═" * 60)
    print("DIXON-COLES: Entrenamiento")
    print("═" * 60)

    train = pd.read_parquet(TRAINING_SET)
    val   = pd.read_parquet(VALIDATION_SET) if VALIDATION_SET.exists() else pd.DataFrame()

    # Unir train + val para máximo de datos
    df = pd.concat([train, val], ignore_index=True) if not val.empty else train
    df = add_time_weights(df, half_life_days=half_life_days)

    model = DixonColesModel(rho_init=-0.10)
    model.fit(df, weight_col="time_weight")

    # Guardar
    model_path = DATA_MODEL / "dixon_coles.pkl"
    model.save(model_path)

    # Test rápido
    print("\nTest de predicción rápida:")
    teams = list(model.attack_.keys())[:2]
    if len(teams) >= 2:
        pred = model.predict_match(teams[0], teams[1])
        print(f"  {teams[0]} vs {teams[1]}")
        print(f"  P(H/D/A): {pred['p_home_win']:.1%} / {pred['p_draw']:.1%} / {pred['p_away_win']:.1%}")
        print(f"  λ_home={pred['lambda_home']} | λ_away={pred['lambda_away']}")
        print(f"  Marcador más probable: {pred['most_likely_score']}")
        print(f"  Over 2.5: {pred['p_over25']:.1%} | BTTS: {pred['p_btts']:.1%}")

    return model


if __name__ == "__main__":
    model = train_dixon_coles()
    print("\nDixon-Coles listo.")
