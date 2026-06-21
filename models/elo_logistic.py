"""
elo_logistic.py — Modelo Elo + Regresión Logística
====================================================
Capa 1 del ensemble: usa el diferencial de Elo como predictor principal
y añade features adicionales (forma, ranking FIFA, valor de mercado)
para producir probabilidades 1X2 calibradas.

También implementa la actualización dinámica del Elo partido a partido.

Ventaja sobre Dixon-Coles: robusto con pocos datos (solo necesita resultados),
excelente predictor para equipos sin xG disponible.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TRAINING_SET, VALIDATION_SET, DATA_MODEL,
                    ELO_K, ELO_HOME_ADVANTAGE, ensure_dirs)


# ─────────────────────────────────────────────────────────────────────────────
# Motor Elo puro (para actualización y predicción simple)
# ─────────────────────────────────────────────────────────────────────────────

class EloEngine:
    """
    Sistema Elo dinámico para selecciones nacionales.

    Fórmulas:
      E_A = 1 / (1 + 10^((R_B - R_A) / 400))
      ΔR_A = K × (S_A - E_A)

    K varía según importancia del partido:
      60  → Copa del Mundo
      40  → Clasificatorias + Copa continental
      20  → Amistosos
      30  → Resto
    """

    DEFAULT_ELO = 1500.0

    def __init__(self, k_factors: Dict = None, home_advantage: float = 100.0):
        self.k_factors      = k_factors or ELO_K
        self.home_advantage = home_advantage
        self.ratings_: Dict[str, float] = {}

    def _k_factor(self, tournament: str) -> float:
        t_low = str(tournament).lower()
        if any(x in t_low for x in ["world cup", "mundial", "fifa world"]):
            return self.k_factors["world_cup"]
        elif any(x in t_low for x in ["qualifier", "classification", "clasificatorio"]):
            return self.k_factors["qualifier"]
        elif any(x in t_low for x in ["friendly", "amistoso"]):
            return self.k_factors["friendly"]
        return self.k_factors["default"]

    def get_rating(self, team: str) -> float:
        return self.ratings_.get(team, self.DEFAULT_ELO)

    def expected_score(self, rating_a: float, rating_b: float,
                       is_home: bool = False) -> float:
        """Probabilidad esperada de victoria de A sobre B."""
        adj_b = rating_b - (self.home_advantage if is_home else 0.0)
        return 1.0 / (1.0 + 10.0 ** ((adj_b - rating_a) / 400.0))

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int,
               tournament: str = "default",
               neutral: bool = False) -> Tuple[float, float]:
        """
        Actualiza los ratings tras un partido y devuelve los nuevos ratings.
        """
        from typing import Tuple  # import local para evitar circular
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)
        k   = self._k_factor(tournament)

        # Ventaja de localía solo si no es campo neutral
        is_home = not neutral
        e_h = self.expected_score(r_h, r_a, is_home=is_home)
        e_a = 1.0 - e_h

        # Resultado real
        if home_goals > away_goals:
            s_h, s_a = 1.0, 0.0
        elif home_goals == away_goals:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        self.ratings_[home_team] = r_h + k * (s_h - e_h)
        self.ratings_[away_team] = r_a + k * (s_a - e_a)

        return self.ratings_[home_team], self.ratings_[away_team]

    def build_from_history(self, df: pd.DataFrame,
                           home_col: str = "home_team",
                           away_col: str = "away_team",
                           home_goals: str = "home_score",
                           away_goals: str = "away_score",
                           tournament_col: str = "tournament",
                           neutral_col: str = "neutral") -> pd.DataFrame:
        """
        Recorre el historial en orden cronológico y construye el Elo histórico.
        Devuelve el dataframe con columnas elo_home_pre y elo_away_pre.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df[home_goals] = pd.to_numeric(df[home_goals], errors="coerce")
        df[away_goals] = pd.to_numeric(df[away_goals], errors="coerce")
        df = df.dropna(subset=[home_goals, away_goals])

        records = []
        for _, row in df.iterrows():
            ht  = row[home_col]
            at  = row[away_col]
            hg  = int(row[home_goals])
            ag  = int(row[away_goals])
            tour = str(row.get(tournament_col, "default"))
            neut = bool(row.get(neutral_col, False))

            r_h_pre = self.get_rating(ht)
            r_a_pre = self.get_rating(at)
            e_h     = self.expected_score(r_h_pre, r_a_pre, is_home=(not neut))

            r_h_post, r_a_post = self.update(ht, at, hg, ag, tour, neut)

            records.append({
                "date":          row["date"],
                "home_team":     ht,
                "away_team":     at,
                "elo_home_pre":  round(r_h_pre, 2),
                "elo_away_pre":  round(r_a_pre, 2),
                "elo_home_post": round(r_h_post, 2),
                "elo_away_post": round(r_a_post, 2),
                "expected_home": round(e_h, 4),
                "delta_elo":     round(r_h_pre - r_a_pre, 2),
            })

        return pd.DataFrame(records)

    def predict_proba(self, home_team: str, away_team: str,
                      neutral: bool = True) -> Dict:
        """
        Predicción 1X2 directa desde Elo puro (sin ML).
        Usa el modelo de probabilidades de Elo con ajuste empírico para empate.
        """
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)
        is_home = not neutral

        e_h = self.expected_score(r_h, r_a, is_home=is_home)

        # Ajuste empírico: en fútbol ~25% de partidos son empate
        # El Elo puro no modela empates; usamos la distribución Elo → 1X2
        # basada en el paper de Hvattum & Arntzen (2010)
        draw_prob = max(0.08, 0.32 - 0.5 * abs(e_h - 0.5))
        home_prob = e_h * (1.0 - draw_prob)
        away_prob = (1.0 - e_h) * (1.0 - draw_prob)

        total = home_prob + draw_prob + away_prob
        return {
            "home_team"   : home_team,
            "away_team"   : away_team,
            "p_home_win"  : round(home_prob / total, 4),
            "p_draw"      : round(draw_prob / total, 4),
            "p_away_win"  : round(away_prob / total, 4),
            "elo_home"    : round(r_h, 1),
            "elo_away"    : round(r_a, 1),
            "model"       : "EloRaw",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Modelo Elo + Regresión Logística Multinomial
# ─────────────────────────────────────────────────────────────────────────────

# Features usadas por la regresión logística (en orden de importancia empírica)
LOGISTIC_FEATURES = [
    "delta_elo",          # ~28% importancia
    "delta_form",         # ~11%
    "delta_form_pts6",    # corr con forma
    "delta_fifa_rank",    # ~12%
    "delta_fifa_pts",     # corr con ranking
    "delta_sv_log",       # ~9%  (valor de mercado)
    "delta_xg",           # ~19% (solo si disponible)
    "delta_xga",          # corr con xG
    "delta_rest",         # ~6%
    "home_is_knockout",   # contexto eliminatorio
    "expected_home_elo",  # probabilidad Elo previa
]


class EloLogisticModel:
    """
    Combina el Elo pre-partido como feature principal de una
    regresión logística multinomial (H/D/A).

    Output: P(H), P(D), P(A) como probabilidades calibradas.
    """

    def __init__(self, features: List[str] = None, C: float = 0.5):
        self.features  = features or LOGISTIC_FEATURES
        self.C         = C
        self.pipeline_ = None
        self.is_fitted  = False
        self.classes_   = ["H", "D", "A"]
        self.available_features_: List[str] = []

    def get_elo(self, team: str) -> float:
        """Devuelve el Elo actual de un equipo para resolver empates en penaltis."""
        if not hasattr(self, "_elo_cache_dict") or self._elo_cache_dict is None:
            try:
                import pandas as pd
                from config import DATA_FEATURES
                p = DATA_FEATURES / "elo_current.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    self._elo_cache_dict = dict(zip(df["team"], df["elo_current"]))
                else:
                    self._elo_cache_dict = {}
            except Exception:
                self._elo_cache_dict = {}
        return float(self._elo_cache_dict.get(team, 1500.0))


    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona features disponibles y rellena NaN con medianas."""
        avail = [f for f in self.features if f in df.columns]
        if not avail:
            raise ValueError("Ninguna feature disponible en el dataframe.")
        self.available_features_ = avail
        X = df[avail].copy()
        X = X.fillna(X.median())
        return X

    def fit(self, df: pd.DataFrame,
            target_col: str = "target_result") -> "EloLogisticModel":
        """
        Entrena la regresión logística multinomial.
        target_col: columna con 'H', 'D' o 'A'.
        """
        df = df.copy()
        df = df.dropna(subset=[target_col])
        df = df[df[target_col].isin(["H", "D", "A"])]

        X = self._prepare_X(df)
        y = df[target_col].values

        print(f"  EloLogistic: {len(X):,} muestras | "
              f"{len(self.available_features_)} features")
        print(f"  Features: {self.available_features_}")

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                C=self.C,
                max_iter=1000,
                class_weight="balanced"
            ))
        ])
        self.pipeline_.fit(X, y)
        self.is_fitted = True

        # Log de clases
        classes = self.pipeline_["clf"].classes_.tolist()
        print(f"  Clases: {classes}")
        return self

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice probabilidades para un DataFrame de partidos.
        Devuelve columnas p_home_win, p_draw, p_away_win.
        """
        assert self.is_fitted
        X = df[self.available_features_].fillna(
            df[self.available_features_].median()
        )
        proba = self.pipeline_.predict_proba(X)
        classes = self.pipeline_["clf"].classes_.tolist()

        result = df[["home_team", "away_team"]].copy() if "home_team" in df.columns else df.iloc[:, :2].copy()
        result.columns = ["home_team", "away_team"]

        idx_h = classes.index("H") if "H" in classes else 0
        idx_d = classes.index("D") if "D" in classes else 1
        idx_a = classes.index("A") if "A" in classes else 2

        result["p_home_win"] = proba[:, idx_h].round(4)
        result["p_draw"]     = proba[:, idx_d].round(4)
        result["p_away_win"] = proba[:, idx_a].round(4)
        result["model"]      = "EloLogistic"
        return result

    def predict_match(self, row: pd.Series) -> Dict:
        """Predice un partido individual dado como pd.Series."""
        assert self.is_fitted
        df_single = pd.DataFrame([row])
        result    = self.predict_proba_df(df_single)
        return result.iloc[0].to_dict()

    def save(self, path: Path):
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado: {path}")

    @staticmethod
    def load(path: Path) -> "EloLogisticModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_elo_logistic() -> EloLogisticModel:
    ensure_dirs()
    print("═" * 60)
    print("ELO + LOGISTIC: Entrenamiento")
    print("═" * 60)

    train = pd.read_parquet(TRAINING_SET)
    print(f"  Training set: {len(train):,} partidos")
    print(f"  Columnas disponibles: {list(train.columns)[:15]}...")

    model = EloLogisticModel(C=0.5)
    model.fit(train)

    # Guardar
    path = DATA_MODEL / "elo_logistic.pkl"
    model.save(path)

    # Validación rápida
    if VALIDATION_SET.exists():
        val = pd.read_parquet(VALIDATION_SET)
        val = val.dropna(subset=["target_result"])
        if len(val) > 0:
            preds = model.predict_proba_df(val)
            print(f"\n  Validation: {len(preds):,} predicciones generadas")
            print(f"  P(H) media: {preds['p_home_win'].mean():.3f}")
            print(f"  P(D) media: {preds['p_draw'].mean():.3f}")
            print(f"  P(A) media: {preds['p_away_win'].mean():.3f}")

    return model


if __name__ == "__main__":
    model = train_elo_logistic()
    print("\nElo + Logistic listo.")
