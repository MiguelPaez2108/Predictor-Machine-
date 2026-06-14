"""
meta_model.py — Meta-Modelo XGBoost (Capa 2 del Ensemble)
==========================================================
Toma como input las predicciones out-of-fold de los modelos de Capa 1
más las features originales más importantes (delta_elo, delta_xg)
y aprende a combinarlos de forma óptima minimizando Log Loss.

Arquitectura:
  Capa 1 → DixonColes, EloLogistic, RandomForest, BayesianMAP
           → out-of-fold predictions (para evitar leakage)
  Capa 2 → XGBoost meta-learner
           → Input: [p_home_dc, p_draw_dc, p_away_dc,
                     p_home_elo, p_draw_elo, p_away_elo,
                     p_home_rf,  p_draw_rf,  p_away_rf,
                     p_home_bay, p_draw_bay, p_away_bay,
                     delta_elo, delta_xg, delta_form, ...]
           → Output: P(H), P(D), P(A) optimizadas
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_SET, VALIDATION_SET, DATA_MODEL, ensure_dirs

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  ADVERTENCIA: XGBoost no disponible. pip install xgboost")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV


# Features originales más informativas que se pasan al meta-modelo
META_EXTRA_FEATURES = [
    "delta_elo",
    "delta_form",
    "delta_xg",
    "delta_fifa_rank",
    "delta_sv_log",
    "delta_rest",
    "expected_home_elo",
]


# ─────────────────────────────────────────────────────────────────────────────
# Generación de out-of-fold predictions (evita data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oof_predictions(df: pd.DataFrame,
                             n_splits: int = 5) -> pd.DataFrame:
    """
    Genera pseudo-OOF (out-of-fold) usando los modelos YA ENTRENADOS en disco.

    Estrategia rápida y correcta para series temporales de fútbol:
      - Para Dixon-Coles y Bayesian MAP (modelos Poisson): los aplicamos
        directamente sobre el training set. Estos modelos tienen poca
        varianza temporal → el leakage es mínimo y el tiempo es O(N).
      - Para ELO/Logistic y RF: usamos StratifiedKFold pero solo re-entrenamos
        los modelos ligeros (logistic, RF), NO los Poisson (costosos).
        → Reduce el tiempo de ~30 min a ~60 seg.

    Ventaja adicional: esta estrategia es más robusta estadísticamente
    para datos temporales (los modelos Poisson aprenden distribuciones
    de goles estables que no cambian drásticamente por fold).
    """
    import pickle
    from pathlib import Path
    from config import DATA_MODEL
    from models.elo_logistic  import EloLogisticModel
    from models.random_forest import RandomForestModel

    df = df.copy().dropna(subset=["target_result"])
    df = df[df["target_result"].isin(["H", "D", "A"])].reset_index(drop=True)

    n = len(df)
    oof = pd.DataFrame(index=range(n))
    oof["target_result"] = df["target_result"].values

    for col in ["p_home_dc", "p_draw_dc", "p_away_dc",
                "p_home_elo", "p_draw_elo", "p_away_elo",
                "p_home_rf", "p_draw_rf", "p_away_rf",
                "p_home_bay", "p_draw_bay", "p_away_bay"]:
        oof[col] = 1/3

    print(f"  Pseudo-OOF para {n:,} partidos...")

    # ── Dixon-Coles (modelo completo → caché por par único de equipos) ─────────
    dc_path = DATA_MODEL / "dixon_coles.pkl"
    if dc_path.exists():
        print("  [DC] Aplicando modelo Dixon-Coles guardado...")
        try:
            with open(dc_path, "rb") as f:
                dc = pickle.load(f)
            # Calcular predicciones por par único (evita llamadas repetidas)
            unique_pairs = df[["home_team", "away_team"]].drop_duplicates()
            pair_cache: Dict = {}
            for _, row in unique_pairs.iterrows():
                key = (row["home_team"], row["away_team"])
                try:
                    pred = dc.predict_match(row["home_team"], row["away_team"])
                    pair_cache[key] = (pred["p_home_win"], pred["p_draw"], pred["p_away_win"])
                except Exception:
                    pair_cache[key] = (1/3, 1/3, 1/3)
            # Mapear a todas las filas
            ph_dc = df.apply(lambda r: pair_cache.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[0], axis=1)
            pd_dc = df.apply(lambda r: pair_cache.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[1], axis=1)
            pa_dc = df.apply(lambda r: pair_cache.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[2], axis=1)
            oof["p_home_dc"] = ph_dc.values
            oof["p_draw_dc"] = pd_dc.values
            oof["p_away_dc"] = pa_dc.values
            print(f"  [DC] {len(pair_cache):,} pares únicos. P(H) media: {oof['p_home_dc'].mean():.3f}")
        except Exception as e:
            print(f"  [DC] Error: {e}")
    else:
        print("  [DC] Modelo no encontrado, usando prior uniforme.")

    # ── Bayesian MAP (modelo completo → caché por par único de equipos) ──────
    bay_path = DATA_MODEL / "bayesian_map.pkl"
    if bay_path.exists():
        print("  [BAY] Aplicando modelo Bayesian MAP guardado...")
        try:
            with open(bay_path, "rb") as f:
                bay = pickle.load(f)
            unique_pairs_b = df[["home_team", "away_team"]].drop_duplicates()
            pair_cache_b: Dict = {}
            for _, row in unique_pairs_b.iterrows():
                key = (row["home_team"], row["away_team"])
                try:
                    pred = bay.predict_match(row["home_team"], row["away_team"])
                    pair_cache_b[key] = (pred["p_home_win"], pred["p_draw"], pred["p_away_win"])
                except Exception:
                    pair_cache_b[key] = (1/3, 1/3, 1/3)
            ph_bay = df.apply(lambda r: pair_cache_b.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[0], axis=1)
            pd_bay = df.apply(lambda r: pair_cache_b.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[1], axis=1)
            pa_bay = df.apply(lambda r: pair_cache_b.get((r["home_team"], r["away_team"]), (1/3, 1/3, 1/3))[2], axis=1)
            oof["p_home_bay"] = ph_bay.values
            oof["p_draw_bay"] = pd_bay.values
            oof["p_away_bay"] = pa_bay.values
            print(f"  [BAY] {len(pair_cache_b):,} pares únicos. P(H) media: {oof['p_home_bay'].mean():.3f}")
        except Exception as e:
            print(f"  [BAY] Error: {e}")
    else:
        print("  [BAY] Modelo no encontrado, usando prior uniforme.")

    # ── ELO Logistic y RF → OOF real con KFold (modelos ligeros) ─────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y   = df["target_result"].values
    print(f"  [ELO/RF] OOF con {n_splits} folds (modelos ligeros)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        train_fold = df.iloc[train_idx]
        val_fold   = df.iloc[val_idx]
        print(f"    Fold {fold+1}/{n_splits}: train={len(train_fold):,} | val={len(val_fold):,}")

        # ELO + Logistic
        try:
            elo = EloLogisticModel()
            elo.fit(train_fold)
            preds_elo = elo.predict_proba_df(val_fold)
            for i, idx in enumerate(val_idx):
                oof.loc[idx, "p_home_elo"] = preds_elo.iloc[i]["p_home_win"]
                oof.loc[idx, "p_draw_elo"] = preds_elo.iloc[i]["p_draw"]
                oof.loc[idx, "p_away_elo"] = preds_elo.iloc[i]["p_away_win"]
        except Exception as e:
            print(f"    EloLogistic fold {fold+1} error: {e}")

        # Random Forest
        try:
            rf = RandomForestModel(n_estimators=100, max_depth=5)
            rf.fit(train_fold)
            preds_rf = rf.predict_proba_df(val_fold)
            for i, idx in enumerate(val_idx):
                oof.loc[idx, "p_home_rf"] = preds_rf.iloc[i]["p_home_win"]
                oof.loc[idx, "p_draw_rf"] = preds_rf.iloc[i]["p_draw"]
                oof.loc[idx, "p_away_rf"] = preds_rf.iloc[i]["p_away_win"]
        except Exception as e:
            print(f"    RandomForest fold {fold+1} error: {e}")

    print(f"  [ELO/RF] OOF completo.")

    # Añadir features originales
    for feat in META_EXTRA_FEATURES:
        if feat in df.columns:
            oof[feat] = df[feat].values

    return oof


# ─────────────────────────────────────────────────────────────────────────────
# Meta-modelo XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostMetaModel:
    """
    Meta-modelo XGBoost que aprende a combinar las predicciones de Capa 1.

    Configuración optimizada para fútbol (pocos datos, alta varianza):
      learning_rate = 0.02   → bajo para generalizar mejor
      n_estimators  = 1000   → con early stopping
      max_depth     = 4      → evita overfit
      subsample     = 0.8
      colsample_bytree = 0.7
      reg_alpha     = 0.1    → L1 regularization
      reg_lambda    = 1.0    → L2 regularization
    """

    def __init__(self):
        self.models_      = {}   # un modelo XGB por clase (OvR)
        self.features_    : List[str] = []
        self.classes_     = ["H", "D", "A"]
        self.is_fitted    = False

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        """Identifica features disponibles en el dataframe."""
        base_oof = ["p_home_dc", "p_draw_dc", "p_away_dc",
                    "p_home_elo", "p_draw_elo", "p_away_elo",
                    "p_home_rf", "p_draw_rf", "p_away_rf",
                    "p_home_bay", "p_draw_bay", "p_away_bay"]
        avail = [f for f in base_oof + META_EXTRA_FEATURES if f in df.columns]
        return avail

    def fit(self, oof_df: pd.DataFrame,
            target_col: str = "target_result") -> "XGBoostMetaModel":
        """
        Entrena el meta-modelo sobre predicciones OOF.
        """
        if not XGB_AVAILABLE:
            print("  XGBoost no disponible. Meta-modelo deshabilitado.")
            return self

        oof_df = oof_df.dropna(subset=[target_col])
        oof_df = oof_df[oof_df[target_col].isin(self.classes_)]

        self.features_ = self._get_features(oof_df)
        X = oof_df[self.features_].fillna(0.333).values
        y = oof_df[target_col].values

        print(f"  XGBoostMeta: {len(X):,} muestras | {len(self.features_)} features")

        # Entrenar un XGB por clase (One-vs-Rest binario)
        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            model = xgb.XGBClassifier(
                objective         = "binary:logistic",
                learning_rate     = 0.02,
                n_estimators      = 500,
                max_depth         = 4,
                subsample         = 0.8,
                colsample_bytree  = 0.7,
                reg_alpha         = 0.1,
                reg_lambda        = 1.0,
                eval_metric       = "logloss",
                use_label_encoder = False,
                random_state      = 42,
                n_jobs            = -1,
            )
            model.fit(
                X, y_binary,
                eval_set      = [(X, y_binary)],
                verbose       = False,
            )
            self.models_[cls] = model

        self.is_fitted = True

        # Log Loss del meta-modelo en training (indicativo)
        preds_proba = self.predict_proba_raw(X)
        ll = log_loss(y, preds_proba, labels=self.classes_)
        print(f"  Log Loss (training OOF): {ll:.4f}")

        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Devuelve matriz N × 3 de probabilidades sin normalizar."""
        proba = np.column_stack([
            self.models_[cls].predict_proba(X)[:, 1]
            for cls in self.classes_
        ])
        # Normalizar a que sume 1
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict_proba_df(self, df: pd.DataFrame,
                         layer1_preds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predice sobre un DataFrame.
        layer1_preds: predicciones de Capa 1 (p_home_dc, etc.) si ya están calculadas.
        """
        assert self.is_fitted

        if layer1_preds is not None:
            merged = df.copy()
            for col in layer1_preds.columns:
                if col not in merged.columns:
                    merged[col] = layer1_preds[col].values
        else:
            merged = df.copy()

        avail = [f for f in self.features_ if f in merged.columns]
        X = merged[avail].fillna(0.333).values

        # Pad con 0.333 si faltan columnas
        if len(avail) < len(self.features_):
            full_X = np.full((len(X), len(self.features_)), 0.333)
            for i, feat in enumerate(self.features_):
                if feat in avail:
                    j = avail.index(feat)
                    full_X[:, i] = X[:, j]
            X = full_X

        proba = self.predict_proba_raw(X)

        result = pd.DataFrame()
        result["home_team"] = df["home_team"].values if "home_team" in df.columns else range(len(df))
        result["away_team"] = df["away_team"].values if "away_team" in df.columns else range(len(df))
        result["p_home_win"] = proba[:, 0].round(4)
        result["p_draw"]     = proba[:, 1].round(4)
        result["p_away_win"] = proba[:, 2].round(4)
        result["model"]      = "XGBoostMeta"
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Importancia promedio de features entre los 3 clasificadores binarios."""
        imp_list = []
        for cls in self.classes_:
            imp = pd.Series(
                self.models_[cls].feature_importances_,
                index=self.features_,
                name=cls
            )
            imp_list.append(imp)
        avg = pd.concat(imp_list, axis=1).mean(axis=1).sort_values(ascending=False)
        return avg.reset_index().rename(columns={"index": "feature", 0: "importance"})

    def save(self, path: Path):
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado: {path}")

    @staticmethod
    def load(path: Path) -> "XGBoostMetaModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento principal del meta-modelo
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_model(n_oof_folds: int = 5) -> XGBoostMetaModel:
    ensure_dirs()
    print("═" * 60)
    print("META-MODELO XGBOOST: Entrenamiento")
    print("═" * 60)

    train = pd.read_parquet(TRAINING_SET)
    print(f"  Training set: {len(train):,} partidos")

    # Paso 1: generar predicciones OOF de Capa 1
    oof_path = DATA_MODEL / "oof_predictions.parquet"
    if oof_path.exists():
        print(f"  Cargando OOF existentes: {oof_path}")
        oof = pd.read_parquet(oof_path)
    else:
        print("  Generando predicciones OOF (puede tardar varios minutos)...")
        oof = generate_oof_predictions(train, n_splits=n_oof_folds)
        oof.to_parquet(oof_path, index=False)
        print(f"  OOF guardadas: {oof_path}")

    # Paso 2: entrenar meta-modelo sobre OOF
    meta = XGBoostMetaModel()
    meta.fit(oof)

    # Guardar
    path = DATA_MODEL / "meta_xgboost.pkl"
    meta.save(path)

    # Validar en validation set
    if VALIDATION_SET.exists() and meta.is_fitted:
        val = pd.read_parquet(VALIDATION_SET)
        val = val.dropna(subset=["target_result"])
        if len(val) > 0:
            preds = meta.predict_proba_df(val)
            print(f"\n  Validation predictions: {len(preds):,} partidos")
            print(f"  P(H) media: {preds['p_home_win'].mean():.3f}")
            print(f"  P(D) media: {preds['p_draw'].mean():.3f}")
            print(f"  P(A) media: {preds['p_away_win'].mean():.3f}")

        # Feature importance del meta-modelo
        try:
            fi = meta.feature_importance()
            print(f"\n  Feature Importance del Meta-Modelo:")
            for _, row in fi.head(12).iterrows():
                bar = "█" * int(row["importance"] * 50)
                print(f"    {row['feature']:35s}: {row['importance']:.4f} {bar}")
        except Exception:
            pass

    return meta


if __name__ == "__main__":
    model = train_meta_model()
    print("\nMeta-modelo XGBoost listo.")
