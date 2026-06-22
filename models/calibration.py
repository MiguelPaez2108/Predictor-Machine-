"""
calibration.py — Capa 3: Calibración isotónica de probabilidades
=================================================================
Garantiza que si el modelo dice P(H)=0.60, efectivamente
el equipo local gana en el 60% de los casos.

Sin calibración → probabilidades bien rankeadas pero mal escaladas.
Con calibración → probabilidades válidas para Kelly Criterion y apuestas.

Métodos implementados:
  1. Isotonic Regression (no-paramétrica, más flexible)
  2. Platt Scaling (regresión logística sobre las salidas del modelo)
  3. Temperature Scaling (un solo parámetro, más robusta con pocos datos)

Para fútbol se recomienda Isotonic Regression con suficientes datos (>1000 muestras)
y Temperature Scaling con datos escasos (<500 muestras).

También incluye:
  - Curvas de calibración (reliability diagrams)
  - Expected Calibration Error (ECE)
  - Comparación antes/después de calibrar
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VALIDATION_SET, DATA_MODEL, ensure_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error (ECE) — métricas de calibración
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    ECE: mide el error promedio de calibración en N bins de probabilidad.
    ECE=0 → calibración perfecta.
    ECE=0.05 → error medio de 5 puntos porcentuales.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(y_true)

    for b in range(n_bins):
        mask = (y_pred >= bins[b]) & (y_pred < bins[b + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()   # frecuencia real
        conf = y_pred[mask].mean()   # confianza predicha
        ece += mask.sum() / n * abs(acc - conf)

    return round(ece, 5)


def reliability_data(y_true: np.ndarray, y_pred: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Datos para graficar la curva de calibración (reliability diagram)."""
    bins   = np.linspace(0, 1, n_bins + 1)
    rows   = []

    for b in range(n_bins):
        mask = (y_pred >= bins[b]) & (y_pred < bins[b + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center"  : (bins[b] + bins[b + 1]) / 2,
            "freq_real"   : y_true[mask].mean(),
            "mean_pred"   : y_pred[mask].mean(),
            "count"       : mask.sum(),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaling:
    """
    Divide los logits por una temperatura T optimizada.
    T>1 → suaviza las probabilidades (más incertidumbre)
    T<1 → agudiza las probabilidades (más confianza)

    Un solo parámetro → muy poco overfit, ideal con pocos datos.
    """

    def __init__(self):
        self.T_ = 1.0

    def _softmax(self, logits: np.ndarray, T: float) -> np.ndarray:
        scaled = logits / T
        exp    = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _logits_from_proba(self, proba: np.ndarray) -> np.ndarray:
        """Recupera logits aproximados desde probabilidades."""
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        return np.log(proba)

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """
        proba: N × 3 array de probabilidades [H, D, A]
        y_true: array de strings o enteros con clases
        """
        classes = ["H", "D", "A"]
        if isinstance(y_true[0], str):
            y_int = np.array([classes.index(c) for c in y_true])
        else:
            y_int = y_true.astype(int)

        logits = self._logits_from_proba(proba)

        def nll(T):
            if T <= 0:
                return 1e10
            p = self._softmax(logits, T)
            return log_loss(y_int, p)

        result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
        self.T_ = result.x
        print(f"  Temperature Scaling: T={self.T_:.4f} "
              f"(>1=más suave, <1=más confiado)")
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logits = self._logits_from_proba(proba)
        return self._softmax(logits, self.T_)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrador por clase (isotonic o platt)
# ─────────────────────────────────────────────────────────────────────────────

class MulticlassCalibrator:
    """
    Calibrador multinomial que aplica calibración independiente por clase.
    Estrategia OvR (One-vs-Rest) con isotonic regression.

    Garantiza: P(H) + P(D) + P(A) = 1.00 después de calibrar.
    """

    def __init__(self, method: str = "isotonic"):
        """
        method: "isotonic" | "platt" | "temperature"
        """
        self.method      = method
        self.calibrators_: Dict = {}
        self.temp_scaler_: Optional[TemperatureScaling] = None
        self.classes_    = ["H", "D", "A"]
        self.is_fitted   = False

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "MulticlassCalibrator":
        """
        proba: N × 3 array [p_H, p_D, p_A]
        y_true: array de 'H','D','A'
        """
        if self.method == "temperature":
            self.temp_scaler_ = TemperatureScaling()
            self.temp_scaler_.fit(proba, y_true)
            self.is_fitted = True
            return self

        for i, cls in enumerate(self.classes_):
            y_binary = (y_true == cls).astype(float)
            p_class  = proba[:, i]

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_class, y_binary)
            elif self.method == "platt":
                cal = LogisticRegression(C=1.0)
                cal.fit(p_class.reshape(-1, 1), y_binary)
            else:
                raise ValueError(f"Método desconocido: {self.method}")

            self.calibrators_[cls] = cal

        self.is_fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Calibra y re-normaliza las probabilidades."""
        assert self.is_fitted

        if self.method == "temperature":
            return self.temp_scaler_.transform(proba)

        calibrated = np.zeros_like(proba)
        for i, cls in enumerate(self.classes_):
            cal   = self.calibrators_[cls]
            p_raw = proba[:, i]
            if self.method == "isotonic":
                calibrated[:, i] = cal.predict(p_raw)
            elif self.method == "platt":
                calibrated[:, i] = cal.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        # Re-normalizar para que sume exactamente 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = np.where(row_sums > 0, calibrated / row_sums, 1.0 / 3)
        return calibrated

    def fit_transform(self, proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(proba, y_true).transform(proba)

    def evaluate(self, proba_raw: np.ndarray, proba_cal: np.ndarray,
                 y_true: np.ndarray) -> Dict:
        """Compara métricas antes y después de calibrar."""
        classes = ["H", "D", "A"]

        def compute_metrics(proba, y):
            ll = log_loss(y, proba, labels=classes)
            ece_vals = []
            for i, cls in enumerate(classes):
                y_bin = (y == cls).astype(float)
                ece_vals.append(expected_calibration_error(y_bin, proba[:, i]))
            return {"log_loss": round(ll, 4), "ece_avg": round(np.mean(ece_vals), 5)}

        before = compute_metrics(proba_raw, y_true)
        after  = compute_metrics(proba_cal, y_true)

        return {"before": before, "after": after,
                "delta_ll":  round(before["log_loss"] - after["log_loss"], 4),
                "delta_ece": round(before["ece_avg"]  - after["ece_avg"],  5)}

    def save(self, path: Path):
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Calibrador guardado: {path}")

    @staticmethod
    def load(path: Path) -> "MulticlassCalibrator":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de predicción final calibrado
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedEnsemble:
    """
    Pipeline completo:
      Capa 1 (Dixon-Coles, Elo, RF, Bayesian)
      → Capa 2 (XGBoost meta-modelo)
      → Capa 3 (MulticlassCalibrator)
      → Probabilidades finales calibradas

    Uso:
      ensemble = CalibratedEnsemble.load(DATA_MODEL / "ensemble_final.pkl")
      proba = ensemble.predict("Argentina", "France", row_features)
    """

    def __init__(self, method: str = "isotonic"):
        self.dc_model_     = None
        self.elo_model_    = None
        self.rf_model_     = None
        self.bay_model_    = None
        self.meta_model_   = None
        self.calibrator_   = MulticlassCalibrator(method=method)
        self.is_fitted     = False

    def _get_layer1_proba(self, home_team: str, away_team: str,
                          row: Optional[pd.Series] = None) -> np.ndarray:
        """Obtiene predicciones de Capa 1 para un partido."""
        proba = np.full((1, 4, 3), 0.333)   # 4 modelos × 3 clases

        # Dixon-Coles
        if self.dc_model_ is not None:
            try:
                p = self.dc_model_.predict_match(home_team, away_team)
                proba[0, 0] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
            except Exception:
                pass

        # Bayesian MAP
        if self.bay_model_ is not None:
            try:
                p = self.bay_model_.predict_match(home_team, away_team)
                proba[0, 3] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
            except Exception:
                pass

        # Elo Logistic + RF necesitan features → requieren row
        if row is not None:
            df_row = pd.DataFrame([row])
            df_row["home_team"] = home_team
            df_row["away_team"] = away_team

            if self.elo_model_ is not None and self.elo_model_.is_fitted:
                try:
                    p = self.elo_model_.predict_proba_df(df_row).iloc[0]
                    proba[0, 1] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
                except Exception:
                    pass

            if self.rf_model_ is not None and self.rf_model_.is_fitted:
                try:
                    p = self.rf_model_.predict_proba_df(df_row).iloc[0]
                    proba[0, 2] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
                except Exception:
                    pass

        return proba[0]   # shape: 4 × 3

    def predict(self, home_team: str, away_team: str,
                row: Optional[pd.Series] = None) -> Dict:
        """
        Predicción completa para un partido.
        Devuelve probabilidades finales calibradas.
        """
        assert self.is_fitted

        layer1 = self._get_layer1_proba(home_team, away_team, row)

        # Promedio simple de Capa 1 si no hay meta-modelo
        ensemble_raw = layer1.mean(axis=0)

        # Meta-modelo si disponible
        if self.meta_model_ is not None and self.meta_model_.is_fitted:
            meta_input = {
                "p_home_dc" : layer1[0, 0], "p_draw_dc" : layer1[0, 1],
                "p_away_dc" : layer1[0, 2],
                "p_home_elo": layer1[1, 0], "p_draw_elo": layer1[1, 1],
                "p_away_elo": layer1[1, 2],
                "p_home_rf" : layer1[2, 0], "p_draw_rf" : layer1[2, 1],
                "p_away_rf" : layer1[2, 2],
                "p_home_bay": layer1[3, 0], "p_draw_bay": layer1[3, 1],
                "p_away_bay": layer1[3, 2],
            }
            if row is not None:
                for feat in ["delta_elo", "delta_form", "delta_xg",
                             "delta_fifa_rank", "delta_sv_log"]:
                    meta_input[feat] = row.get(feat, 0.0) if isinstance(row, dict) else getattr(row, feat, 0.0)

            df_meta = pd.DataFrame([meta_input])
            df_meta["home_team"] = home_team
            df_meta["away_team"] = away_team
            pred_meta = self.meta_model_.predict_proba_df(df_meta)
            ensemble_raw = np.array([
                pred_meta.iloc[0]["p_home_win"],
                pred_meta.iloc[0]["p_draw"],
                pred_meta.iloc[0]["p_away_win"],
            ])

        # Calibración final
        ensemble_cal = self.calibrator_.transform(ensemble_raw.reshape(1, -1))[0]

        return {
            "home_team"   : home_team,
            "away_team"   : away_team,
            "p_home_win"  : round(float(ensemble_cal[0]), 4),
            "p_draw"      : round(float(ensemble_cal[1]), 4),
            "p_away_win"  : round(float(ensemble_cal[2]), 4),
            "p_home_pct"  : f"{ensemble_cal[0]:.1%}",
            "p_draw_pct"  : f"{ensemble_cal[1]:.1%}",
            "p_away_pct"  : f"{ensemble_cal[2]:.1%}",
            "layer1_dc"   : [round(x, 4) for x in layer1[0].tolist()],
            "layer1_elo"  : [round(x, 4) for x in layer1[1].tolist()],
            "layer1_rf"   : [round(x, 4) for x in layer1[2].tolist()],
            "layer1_bay"  : [round(x, 4) for x in layer1[3].tolist()],
            "model"       : "CalibratedEnsemble",
        }

    def save(self, path: Path):
        import pickle, os
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Ensemble final guardado: {path}")

    @staticmethod
    def load(path: Path) -> "CalibratedEnsemble":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento del calibrador
# ─────────────────────────────────────────────────────────────────────────────

def train_calibrator(method: str = "isotonic") -> MulticlassCalibrator:
    """
    Carga las predicciones del meta-modelo en validación y calibra.
    """
    ensure_dirs()
    print("═" * 60)
    print(f"CALIBRACIÓN ({method.upper()}): Entrenamiento")
    print("═" * 60)

    if not VALIDATION_SET.exists():
        print("  Sin validation set. Calibrador sin datos.")
        return MulticlassCalibrator(method=method)

    val = pd.read_parquet(VALIDATION_SET)
    val = val.dropna(subset=["target_result"])
    val = val[val["target_result"].isin(["H", "D", "A"])]

    # Isotonic regression necesita MUCHOS datos por bin de probabilidad para
    # no memorizar ruido. Con 3 calibradores OvR (H/D/A) sobre validation sets
    # de fútbol (que rara vez superan unos pocos miles de partidos), 50 es un
    # umbral absurdamente bajo. Subimos a 1500 — por debajo de eso, Isotonic
    # Regression tiende a sobreajustar y empeorar el Log Loss aunque el ECE
    # mejore (ver resultados.txt: ECE bajó pero Log Loss subió).
    MIN_SAMPLES_FOR_ISOTONIC = 1500

    if method == "isotonic" and len(val) < MIN_SAMPLES_FOR_ISOTONIC:
        print(f"  Validation set pequeño para Isotonic ({len(val):,} < "
              f"{MIN_SAMPLES_FOR_ISOTONIC:,}). Usando temperature scaling "
              f"(1 solo parámetro, mucho menos riesgo de sobreajuste).")
        method = "temperature"

    print(f"  Calibrando sobre {len(val):,} partidos del validation set "
          f"(método: {method})")

    # Intentar cargar predicciones del meta-modelo o del RF como fallback
    meta_path = DATA_MODEL / "meta_xgboost.pkl"
    rf_path   = DATA_MODEL / "random_forest.pkl"

    proba_raw = None
    y_true    = val["target_result"].values

    if meta_path.exists():
        try:
            import pickle
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            preds = meta.predict_proba_df(val)
            proba_raw = preds[["p_home_win", "p_draw", "p_away_win"]].values
            print(f"  Usando predicciones del meta-modelo XGBoost")
        except Exception as e:
            print(f"  Meta-modelo no disponible: {e}")

    if proba_raw is None and rf_path.exists():
        try:
            import pickle
            with open(rf_path, "rb") as f:
                rf = pickle.load(f)
            preds = rf.predict_proba_df(val)
            proba_raw = preds[["p_home_win", "p_draw", "p_away_win"]].values
            print(f"  Usando predicciones del Random Forest como fallback")
        except Exception as e:
            print(f"  RF no disponible: {e}")

    if proba_raw is None:
        print("  Sin modelos entrenados. Generando calibrador con prior uniforme.")
        proba_raw = np.full((len(val), 3), 1.0 / 3)

    # Entrenar calibrador
    calibrator = MulticlassCalibrator(method=method)
    calibrator.fit(proba_raw, y_true)

    # Evaluar
    proba_cal = calibrator.transform(proba_raw)
    metrics   = calibrator.evaluate(proba_raw, proba_cal, y_true)
    print(f"\n  Resultados de calibración:")
    print(f"    Log Loss antes : {metrics['before']['log_loss']}")
    print(f"    Log Loss después: {metrics['after']['log_loss']}")
    print(f"    Δ Log Loss      : {metrics['delta_ll']:+.4f}")
    print(f"    ECE antes      : {metrics['before']['ece_avg']}")
    print(f"    ECE después    : {metrics['after']['ece_avg']}")
    print(f"    Δ ECE          : {metrics['delta_ece']:+.5f}")

    # ── GUARDIA DE CALIDAD ────────────────────────────────────────────────
    # Si el calibrador EMPEORA el Log Loss, no sirve para producción —
    # significa que sobreajustó el validation set y va a distorsionar las
    # probabilidades en datos nuevos (ej. WC2026). En ese caso, probamos
    # con temperature scaling como fallback más conservador. Si tampoco
    # mejora, NO guardamos ningún calibrador (dejamos pasar las
    # probabilidades crudas del meta-modelo sin tocar).
    if metrics["delta_ll"] < 0:
        print(f"\n  [AVISO] El calibrador {method} EMPEORA el Log Loss "
              f"({metrics['delta_ll']:+.4f}). Esto indica sobreajuste.")

        if method != "temperature":
            print(f"  → Reintentando con temperature scaling (más robusto)...")
            calibrator_temp = MulticlassCalibrator(method="temperature")
            calibrator_temp.fit(proba_raw, y_true)
            proba_cal_temp = calibrator_temp.transform(proba_raw)
            metrics_temp = calibrator_temp.evaluate(proba_raw, proba_cal_temp, y_true)
            print(f"    Log Loss con temperature: {metrics_temp['after']['log_loss']} "
                  f"(Δ={metrics_temp['delta_ll']:+.4f})")

            if metrics_temp["delta_ll"] >= 0:
                print(f"  [OK] Temperature scaling sí mejora. Usando este en su lugar.")
                calibrator = calibrator_temp
                metrics = metrics_temp
            else:
                print(f"  [ERROR] Ni siquiera temperature scaling mejora el Log Loss.")
                print(f"    NO se guardará ningún calibrador — el pipeline usará")
                print(f"    las probabilidades crudas del meta-modelo sin calibrar.")
                return None
        else:
            print(f"  [ERROR] Temperature scaling tampoco mejora el Log Loss.")
            print(f"    NO se guardará ningún calibrador.")
            return None

    # Guardar (solo si pasó la guardia de calidad)
    path = DATA_MODEL / "calibrator.pkl"
    calibrator.save(path)
    print(f"\n  [OK] Calibrador final: {calibrator.method} "
          f"(Log Loss final: {metrics['after']['log_loss']}, "
          f"Δ={metrics['delta_ll']:+.4f})")

    return calibrator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="isotonic",
                        choices=["isotonic", "platt", "temperature"],
                        help="Método de calibración")
    args = parser.parse_args()
    cal = train_calibrator(method=args.method)
    print("\nCalibración lista.")
