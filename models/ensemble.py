"""
ensemble.py — Orquestador del Sistema Completo
===============================================
Punto de entrada principal del sistema de predicción.

Coordina las 3 capas del modelo:
  Capa 1: Dixon-Coles + Elo/Logistic + Random Forest + Bayesian MAP
  Capa 2: XGBoost Meta-Modelo (aprende a combinar capa 1)
  Capa 3: Calibración isotónica

Funciones principales:
  train_all()                → entrena todos los modelos en orden
  predict_match(h, a, row)   → predicción 1X2 + O/U + BTTS
  predict_batch(df)          → predicción masiva para un DataFrame
  predict_wc2026(fixtures)   → predicción completa del Mundial

Uso rápido:
  python models/ensemble.py --home "Argentina" --away "France"
  python models/ensemble.py --train
"""

import sys
import warnings
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TRAINING_SET, VALIDATION_SET, DATA_MODEL,
                    WC2026_INPUT, ensure_dirs)


# ─────────────────────────────────────────────────────────────────────────────
# Carga lazy de modelos (singleton por proceso)
# ─────────────────────────────────────────────────────────────────────────────

_MODELS = {}


def _load_model(name: str, path: Path):
    """Carga un modelo desde disco y lo cachea en _MODELS."""
    if name in _MODELS:
        return _MODELS[name]
    import pickle
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    _MODELS[name] = model
    return model


def load_all_models() -> Dict:
    """Carga todos los modelos entrenados."""
    models = {
        "dc"    : _load_model("dc",    DATA_MODEL / "dixon_coles.pkl"),
        "elo"   : _load_model("elo",   DATA_MODEL / "elo_logistic.pkl"),
        "rf"    : _load_model("rf",    DATA_MODEL / "random_forest.pkl"),
        "bay"   : _load_model("bay",   DATA_MODEL / "bayesian_map.pkl"),
        "meta"  : _load_model("meta",  DATA_MODEL / "meta_xgboost.pkl"),
        "cal"   : _load_model("cal",   DATA_MODEL / "calibrator.pkl"),
    }
    available = [k for k, v in models.items() if v is not None]
    print(f"  Modelos cargados: {available}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Predicción de un partido
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(home_team: str, away_team: str,
                  row: Optional[pd.Series] = None,
                  models: Optional[Dict] = None,
                  verbose: bool = True) -> Dict:
    """
    Predicción completa de un partido con el ensemble de 3 capas.

    home_team, away_team: nombres exactos de los equipos.
    row: pd.Series con features del partido (delta_elo, etc.). Puede ser None.
    models: dict de modelos cargados (para reutilizar sin recargar).

    Returns: diccionario con todas las probabilidades.
    """
    if models is None:
        models = load_all_models()

    # ── Capa 1: predicciones individuales ────────────────────────────────────
    layer1 = {}

    # Dixon-Coles → goles esperados, O/U, BTTS, marcador
    dc_pred = None
    if models.get("dc"):
        try:
            dc_pred = models["dc"].predict_match(home_team, away_team)
            layer1["dc"] = [dc_pred["p_home_win"], dc_pred["p_draw"],
                            dc_pred["p_away_win"]]
        except Exception:
            layer1["dc"] = [0.333, 0.333, 0.334]

    # Bayesian MAP → Poisson con prior regularizador
    bay_pred = None
    if models.get("bay"):
        try:
            bay_pred = models["bay"].predict_match(home_team, away_team)
            layer1["bay"] = [bay_pred["p_home_win"], bay_pred["p_draw"],
                             bay_pred["p_away_win"]]
        except Exception:
            layer1["bay"] = [0.333, 0.333, 0.334]

    # Elo Logistic + RF → requieren features tabulares
    elo_pred = rf_pred = None
    if row is not None:
        df_row = pd.DataFrame([row])
        df_row["home_team"] = home_team
        df_row["away_team"] = away_team

        if models.get("elo") and models["elo"].is_fitted:
            try:
                p = models["elo"].predict_proba_df(df_row).iloc[0]
                layer1["elo"] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
                elo_pred = p.to_dict()
            except Exception:
                layer1["elo"] = [0.333, 0.333, 0.334]

        if models.get("rf") and models["rf"].is_fitted:
            try:
                p = models["rf"].predict_proba_df(df_row).iloc[0]
                layer1["rf"] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
                rf_pred = p.to_dict()
            except Exception:
                layer1["rf"] = [0.333, 0.333, 0.334]

    if "elo" not in layer1:
        layer1["elo"] = [0.333, 0.333, 0.334]
    if "rf" not in layer1:
        layer1["rf"] = [0.333, 0.333, 0.334]

    # ── Capa 2: meta-modelo XGBoost ──────────────────────────────────────────
    meta_input = {
        "home_team" : home_team,
        "away_team" : away_team,
        "p_home_dc" : layer1["dc"][0],   "p_draw_dc" : layer1["dc"][1],
        "p_away_dc" : layer1["dc"][2],
        "p_home_elo": layer1["elo"][0],  "p_draw_elo": layer1["elo"][1],
        "p_away_elo": layer1["elo"][2],
        "p_home_rf" : layer1["rf"][0],   "p_draw_rf" : layer1["rf"][1],
        "p_away_rf" : layer1["rf"][2],
        "p_home_bay": layer1["bay"][0],  "p_draw_bay": layer1["bay"][1],
        "p_away_bay": layer1["bay"][2],
    }

    # Añadir features originales si hay row
    if row is not None:
        for feat in ["delta_elo", "delta_form", "delta_xg",
                     "delta_fifa_rank", "delta_sv_log", "delta_rest",
                     "expected_home_elo"]:
            val = row.get(feat, np.nan) if isinstance(row, dict) else getattr(row, feat, np.nan)
            meta_input[feat] = float(val) if not pd.isna(val) else 0.0

    df_meta = pd.DataFrame([meta_input])

    ensemble_raw = np.array([
        np.mean([layer1[k][0] for k in layer1]),
        np.mean([layer1[k][1] for k in layer1]),
        np.mean([layer1[k][2] for k in layer1]),
    ])

    if models.get("meta") and models["meta"].is_fitted:
        try:
            p = models["meta"].predict_proba_df(df_meta).iloc[0]
            ensemble_raw = np.array([p["p_home_win"], p["p_draw"], p["p_away_win"]])
        except Exception:
            pass   # fallback al promedio simple

    # ── Capa 3: calibración ──────────────────────────────────────────────────
    ensemble_final = ensemble_raw.copy()
    if models.get("cal") and models["cal"].is_fitted:
        try:
            ensemble_final = models["cal"].transform(
                ensemble_raw.reshape(1, -1)
            )[0]
        except Exception:
            pass

    # ── Output final ─────────────────────────────────────────────────────────
    p_h, p_d, p_a = ensemble_final

    result = {
        "home_team"          : home_team,
        "away_team"          : away_team,
        "p_home_win"         : round(float(p_h), 4),
        "p_draw"             : round(float(p_d), 4),
        "p_away_win"         : round(float(p_a), 4),
        # Goles esperados (de Dixon-Coles o Bayesian)
        "lambda_home"        : dc_pred.get("lambda_home") if dc_pred else (
                               bay_pred.get("lambda_home") if bay_pred else None),
        "lambda_away"        : dc_pred.get("lambda_away") if dc_pred else (
                               bay_pred.get("lambda_away") if bay_pred else None),
        # Mercados adicionales (de Dixon-Coles)
        "p_over25"           : dc_pred.get("p_over25") if dc_pred else None,
        "p_over15"           : dc_pred.get("p_over15") if dc_pred else None,
        "p_under25"          : dc_pred.get("p_under25") if dc_pred else None,
        "p_btts"             : dc_pred.get("p_btts") if dc_pred else None,
        "most_likely_score"  : dc_pred.get("most_likely_score") if dc_pred else "1-1",
        # Capa 1 detalle
        "layer1_dc"          : layer1["dc"],
        "layer1_elo"         : layer1["elo"],
        "layer1_rf"          : layer1["rf"],
        "layer1_bay"         : layer1["bay"],
        "ensemble_pre_cal"   : [round(x, 4) for x in ensemble_raw.tolist()],
        "model"              : "CalibratedEnsemble3Layer",
    }

    if verbose:
        _print_prediction(result)

    return result


def _print_prediction(pred: Dict):
    """Imprime la predicción en formato bonito."""
    h = pred["home_team"]
    a = pred["away_team"]
    ph = pred["p_home_win"]
    pd_ = pred["p_draw"]
    pa = pred["p_away_win"]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  {h:^20s}  vs  {a:^20s}  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Victoria {h[:18]:18s}: {ph:.1%}  ({ph*100:.1f}%)  ║")
    print(f"║  Empate                             : {pd_:.1%}  ({pd_*100:.1f}%)  ║")
    print(f"║  Victoria {a[:18]:18s}: {pa:.1%}  ({pa*100:.1f}%)  ║")

    if pred.get("lambda_home"):
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Goles esperados: {h[:10]:10s} {pred['lambda_home']:.2f}  |  "
              f"{a[:10]:10s} {pred['lambda_away']:.2f}  ║")
        print(f"║  Marcador más probable: {pred['most_likely_score']:5s}                           ║")
        if pred.get("p_over25") is not None:
            print(f"║  Over 2.5: {pred['p_over25']:.1%}  |  BTTS: {pred['p_btts']:.1%}                 ║")

    # Ganador
    print("╠══════════════════════════════════════════════════════════╣")
    winner = h if ph > max(pd_, pa) else (a if pa > pd_ else "Empate")
    max_p  = max(ph, pd_, pa)
    print(f"║  FAVORITO: {winner:25s}  ({max_p:.1%})        ║")
    print("╚══════════════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
# Predicción en batch
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(fixtures_df: pd.DataFrame,
                  models: Optional[Dict] = None,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Predice un DataFrame de partidos.
    Columnas requeridas: home_team, away_team
    Columnas opcionales: cualquier feature de delta_elo, etc.

    Returns: DataFrame con predicciones por fila.
    """
    if models is None:
        models = load_all_models()

    results = []
    for _, row in fixtures_df.iterrows():
        home = row.get("home_team", row.iloc[0])
        away = row.get("away_team", row.iloc[1])
        try:
            pred = predict_match(home, away, row=row,
                                 models=models, verbose=verbose)
        except Exception as e:
            pred = {
                "home_team": home, "away_team": away,
                "p_home_win": 0.333, "p_draw": 0.333, "p_away_win": 0.334,
                "error": str(e)
            }
        results.append(pred)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento completo del pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_all(skip_oof: bool = False):
    """
    Entrena todos los modelos en el orden correcto:
      1. Dixon-Coles (Poisson bivariada)
      2. Elo + Logistic Regression
      3. Random Forest calibrado
      4. Bayesian MAP
      5. Meta-modelo XGBoost (requiere los 4 anteriores)
      6. Calibrador isotónico

    skip_oof: si True, asume que el archivo OOF ya existe.
    """
    ensure_dirs()
    print("═" * 60)
    print("ENTRENAMIENTO COMPLETO DEL ENSEMBLE")
    print("═" * 60)

    if not TRAINING_SET.exists():
        print(f"ERROR: No se encontró el training set en {TRAINING_SET}")
        print("Ejecutar primero: python features/build_master_features.py")
        return

    # ── Capa 1 ────────────────────────────────────────────────────────────────
    print("\n[1/6] Dixon-Coles...")
    from models.dixon_coles import train_dixon_coles
    train_dixon_coles()

    print("\n[2/6] Elo + Logistic Regression...")
    from models.elo_logistic import train_elo_logistic
    train_elo_logistic()

    print("\n[3/6] Random Forest...")
    from models.random_forest import train_random_forest
    train_random_forest()

    print("\n[4/6] Bayesian MAP...")
    from models.bayesian_model import train_bayesian
    train_bayesian(use_mcmc=False)

    # ── Capa 2 ────────────────────────────────────────────────────────────────
    print("\n[5/6] Meta-Modelo XGBoost (OOF stacking)...")
    from models.meta_model import train_meta_model
    train_meta_model(n_oof_folds=5)

    # ── Capa 3 ────────────────────────────────────────────────────────────────
    print("\n[6/6] Calibración isotónica...")
    from models.calibration import train_calibrator
    train_calibrator(method="isotonic")

    print("\n" + "═" * 60)
    print("[OK] ENTRENAMIENTO COMPLETO")
    print("═" * 60)
    print("\nUso:")
    print('  python models/ensemble.py --home "Argentina" --away "France"')
    print("  python models/ensemble.py --wc2026")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de predicción de fútbol — Ensemble 3 Capas"
    )
    parser.add_argument("--train", action="store_true",
                        help="Entrenar todos los modelos")
    parser.add_argument("--home", type=str, default=None,
                        help="Equipo local")
    parser.add_argument("--away", type=str, default=None,
                        help="Equipo visitante")
    parser.add_argument("--wc2026", action="store_true",
                        help="Predecir todos los fixtures del Mundial 2026")
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados en JSON")
    args = parser.parse_args()

    if args.train:
        train_all()

    elif args.home and args.away:
        models = load_all_models()
        result = predict_match(args.home, args.away, models=models, verbose=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nGuardado en: {args.output}")

    elif args.wc2026:
        if not WC2026_INPUT.exists():
            print(f"ERROR: No se encontró {WC2026_INPUT}")
            print("Ejecutar primero: python features/build_master_features.py")
        else:
            fixtures = pd.read_parquet(WC2026_INPUT)
            models   = load_all_models()
            preds    = predict_batch(fixtures, models=models, verbose=True)
            out_path = DATA_MODEL / "wc2026_predictions.parquet"
            preds.to_parquet(out_path, index=False)
            print(f"\n[OK] {len(preds):,} predicciones guardadas en {out_path}")
            if args.output:
                preds.to_json(args.output, orient="records", indent=2)

    else:
        parser.print_help()
