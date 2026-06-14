"""
random_forest.py — Modelo Random Forest para predicción 1X2
=============================================================
Capa 1 del ensemble: captura interacciones no lineales entre features
que la regresión logística no puede modelar.

Configuración optimizada para datos de fútbol internacional:
  - n_estimators=500, max_depth=6, min_samples_leaf=50
  - Evita overfit con pocos datos de selecciones
  - Incluye calibración Platt (isotonic regression)
  - Output: P(H), P(D), P(A) + importancia de features
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_SET, VALIDATION_SET, DATA_MODEL, ensure_dirs


# Features disponibles — mismas que logistic, pero RF puede aprovechar más
RF_FEATURES = [
    # Core predictors (alta importancia documentada)
    "delta_elo",
    "delta_form",
    "delta_form_pts6",
    "delta_form_pts10" if False else "delta_form",  # fallback
    "delta_fifa_rank",
    "delta_fifa_pts",
    "delta_sv_log",
    "delta_xg",
    "delta_xga",
    "delta_rest",
    "delta_fatigue",
    "delta_gf_avg",
    "delta_ga_avg",
    # Features absolutas (no solo deltas)
    "home_form_weighted",
    "away_form_weighted",
    "home_momentum_trend",
    "away_momentum_trend",
    "home_fifa_rank",
    "away_fifa_rank",
    "elo_home_pre",
    "elo_away_pre",
    "expected_home_elo",
    # Contexto
    "home_is_knockout",
    "home_tournament_weight",
    "home_days_rest",
    "away_days_rest",
    # H2H
    "h2h_win_rate",
    "h2h_gd_avg",
    "h2h_matches",
]


class RandomForestModel:
    """
    Random Forest calibrado para predicción 1X2 en fútbol de selecciones.

    El RF sin calibración tiende a producir probabilidades comprimidas
    (poco extremas). La calibración isotónica corrige esto.
    """

    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 min_samples_leaf: int = 30,
                 features: List[str] = None):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.features         = features or RF_FEATURES

        self.pipeline_         = None
        self.is_fitted          = False
        self.available_features_: List[str] = []
        self.feature_importances_: pd.Series = pd.Series(dtype=float)

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        """Filtra features disponibles en el dataframe."""
        avail = [f for f in self.features if f in df.columns]
        if not avail:
            raise ValueError("Ninguna feature RF disponible en el dataframe.")
        return avail

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        avail = self._get_features(df)
        self.available_features_ = avail
        X = df[avail].copy()
        # Rellenar NaN con mediana del training set
        X = X.fillna(X.median())
        return X

    def fit(self, df: pd.DataFrame,
            target_col: str = "target_result") -> "RandomForestModel":
        """
        Entrena el Random Forest con calibración isotónica via CV.
        """
        df = df.copy()
        df = df.dropna(subset=[target_col])
        df = df[df[target_col].isin(["H", "D", "A"])]

        X = self._prepare(df)
        y = df[target_col].values

        print(f"  RandomForest: {len(X):,} muestras | "
              f"{len(self.available_features_)} features")

        # Distribución de clases
        for cls in ["H", "D", "A"]:
            pct = (y == cls).mean() * 100
            print(f"    {cls}: {pct:.1f}%")

        # Modelo base
        rf_base = RandomForestClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            max_features     = "sqrt",   # √n_features
            bootstrap        = True,
            oob_score        = True,
            class_weight     = "balanced",
            random_state     = 42,
            n_jobs           = -1
        )

        # Calibración isotónica (mejor que Platt para RF)
        calibrated = CalibratedClassifierCV(
            rf_base,
            method  = "isotonic",
            cv      = 5         # 5-fold CV para calibrar out-of-fold
        )

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),   # necesario aunque RF no lo usa directamente
            ("clf",    calibrated)
        ])
        self.pipeline_.fit(X, y)
        self.is_fitted = True

        # Importancia de features del RF base (antes de calibración)
        rf_fitted = None
        clf = self.pipeline_["clf"]
        if hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
            cal_clf = clf.calibrated_classifiers_[0]
            # En scikit-learn >= 1.2 es .estimator, en versiones anteriores es .base_estimator
            rf_fitted = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        elif hasattr(clf, "estimator"):
            rf_fitted = clf.estimator

        if rf_fitted is not None and hasattr(rf_fitted, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                rf_fitted.feature_importances_,
                index=self.available_features_
            ).sort_values(ascending=False)
        else:
            self.feature_importances_ = pd.Series(0.0, index=self.available_features_)

        # OOB score (si disponible)
        if rf_fitted is not None:
            try:
                oob = rf_fitted.oob_score_
                print(f"  OOB Score: {oob:.4f}")
            except Exception:
                pass

        print(f"\n  Top-10 features por importancia:")
        for feat, imp in self.feature_importances_.head(10).items():
            bar = "█" * int(imp * 100)
            print(f"    {feat:35s}: {imp:.4f} {bar}")

        return self

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice probabilidades para un DataFrame de partidos.
        Rellena features faltantes con la mediana del training.
        """
        assert self.is_fitted, "Modelo no ajustado."
        X = df[self.available_features_].fillna(
            df[self.available_features_].median()
        )
        proba = self.pipeline_.predict_proba(X)
        classes = self.pipeline_["clf"].classes_.tolist()

        result = pd.DataFrame()
        result["home_team"] = df["home_team"].values if "home_team" in df.columns else range(len(df))
        result["away_team"] = df["away_team"].values if "away_team" in df.columns else range(len(df))

        for cls, col in [("H", "p_home_win"), ("D", "p_draw"), ("A", "p_away_win")]:
            if cls in classes:
                result[col] = proba[:, classes.index(cls)].round(4)
            else:
                result[col] = 0.333

        result["model"] = "RandomForest"
        return result

    def predict_match(self, row: pd.Series) -> Dict:
        """Predice un partido individual dado como pd.Series."""
        assert self.is_fitted
        result = self.predict_proba_df(pd.DataFrame([row]))
        return result.iloc[0].to_dict()

    def evaluate(self, df: pd.DataFrame,
                 target_col: str = "target_result") -> Dict:
        """
        Evalúa el modelo en un conjunto de validación.
        Devuelve Log Loss, Brier Score y Accuracy.
        """
        assert self.is_fitted
        df = df.dropna(subset=[target_col])
        df = df[df[target_col].isin(["H", "D", "A"])]

        preds = self.predict_proba_df(df)
        classes = ["H", "D", "A"]

        y_true_ohe = np.zeros((len(df), 3))
        y_pred     = np.zeros((len(df), 3))

        for i, cls in enumerate(classes):
            y_true_ohe[:, i] = (df[target_col].values == cls).astype(float)
            col_map = {"H": "p_home_win", "D": "p_draw", "A": "p_away_win"}
            y_pred[:, i] = preds[col_map[cls]].values

        ll     = log_loss(y_true_ohe, y_pred)
        bs_h   = brier_score_loss(y_true_ohe[:, 0], y_pred[:, 0])
        bs_d   = brier_score_loss(y_true_ohe[:, 1], y_pred[:, 1])
        bs_a   = brier_score_loss(y_true_ohe[:, 2], y_pred[:, 2])
        bs_avg = (bs_h + bs_d + bs_a) / 3

        y_pred_cls = classes[np.argmax(y_pred, axis=1).astype(int)[0]]
        acc = (np.argmax(y_pred, axis=1) == np.argmax(y_true_ohe, axis=1)).mean()

        metrics = {
            "log_loss"   : round(ll, 4),
            "brier_score": round(bs_avg, 4),
            "accuracy"   : round(acc, 4),
            "n_samples"  : len(df)
        }
        return metrics

    def save(self, path: Path):
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado: {path}")

    @staticmethod
    def load(path: Path) -> "RandomForestModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest() -> RandomForestModel:
    ensure_dirs()
    print("═" * 60)
    print("RANDOM FOREST: Entrenamiento")
    print("═" * 60)

    train = pd.read_parquet(TRAINING_SET)
    print(f"  Training set: {len(train):,} partidos")

    model = RandomForestModel(
        n_estimators     = 500,
        max_depth        = 6,
        min_samples_leaf = 30
    )
    model.fit(train)

    # Guardar
    path = DATA_MODEL / "random_forest.pkl"
    model.save(path)

    # Validación
    if VALIDATION_SET.exists():
        val = pd.read_parquet(VALIDATION_SET)
        metrics = model.evaluate(val)
        print(f"\n  Métricas en validación:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    return model


if __name__ == "__main__":
    model = train_random_forest()
    print("\nRandom Forest listo.")
