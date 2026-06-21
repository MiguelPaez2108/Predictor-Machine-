"""
run_final.py — Paso final: entrenar Capa 2+3 y correr simulación Monte Carlo
============================================================================
Ejecuta en orden:
  1. Meta-modelo XGBoost (Capa 2)
  2. Calibrador isotónico (Capa 3)
  3. Simulación Monte Carlo del Mundial 2026 (N=10_000 sims por defecto)
  4. Imprime tabla de favoritos al título

Uso:
  python run_final.py                 # 10 000 simulaciones
  python run_final.py --sims 50000   # 50 000 simulaciones (más preciso, más lento)
  python run_final.py --quick        # 1 000 simulaciones (prueba rápida)
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_MODEL, ensure_dirs


def step1_meta_model():
    print("\n" + "═" * 60)
    print("  PASO 1/3 — Meta-Modelo XGBoost (Capa 2)")
    print("═" * 60)
    meta_path = DATA_MODEL / "meta_xgboost.pkl"
    if meta_path.exists():
        print(f"  ✓ meta_xgboost.pkl ya existe → saltando entrenamiento")
        return True
    try:
        from models.meta_model import train_meta_model
        meta = train_meta_model(n_oof_folds=5)
        return meta is not None and meta.is_fitted
    except Exception as e:
        print(f"  ERROR en meta-modelo: {e}")
        import traceback; traceback.print_exc()
        return False


def step2_calibrator():
    print("\n" + "═" * 60)
    print("  PASO 2/3 — Calibrador Isotónico (Capa 3)")
    print("═" * 60)
    cal_path = DATA_MODEL / "calibrator.pkl"
    if cal_path.exists():
        print(f"  ✓ calibrator.pkl ya existe → re-entrenando para asegurar consistencia")
    try:
        from models.calibration import train_calibrator
        cal = train_calibrator(method="isotonic")
        return cal is not None and cal.is_fitted
    except Exception as e:
        print(f"  ERROR en calibrador: {e}")
        import traceback; traceback.print_exc()
        return False


def step3_simulation(n_sims: int):
    print("\n" + "═" * 60)
    print(f"  PASO 3/3 — Simulación Monte Carlo ({n_sims:,} simulaciones)")
    print("═" * 60)
    try:
        from simulation.tournament import (
            simulate_wc2026, print_champion_table,
            print_group_predictions, save_results
        )
        results = simulate_wc2026(n_sims=n_sims, verbose=True)
        if results is None:
            print("  ERROR: La simulación no retornó resultados.")
            return False

        print_champion_table(results, top_n=20)
        print_group_predictions(results)
        save_results(results)
        return True
    except Exception as e:
        print(f"  ERROR en simulación: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline final WC Predictor")
    parser.add_argument("--sims",  type=int, default=10_000,
                        help="Número de simulaciones Monte Carlo (default: 10 000)")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido: 1 000 simulaciones")
    parser.add_argument("--skip-meta",  action="store_true",
                        help="Saltar entrenamiento del meta-modelo")
    parser.add_argument("--skip-cal",   action="store_true",
                        help="Saltar entrenamiento del calibrador")
    parser.add_argument("--only-sim",   action="store_true",
                        help="Solo correr la simulación (ambos modelos deben existir)")
    args = parser.parse_args()

    ensure_dirs()
    n = 1_000 if args.quick else args.sims

    ok_meta = ok_cal = True

    if not args.only_sim:
        if not args.skip_meta:
            ok_meta = step1_meta_model()
        else:
            print("\n  [SKIP] Meta-modelo omitido por --skip-meta")

        if not args.skip_cal:
            ok_cal = step2_calibrator()
        else:
            print("\n  [SKIP] Calibrador omitido por --skip-cal")

    if ok_meta and ok_cal:
        step3_simulation(n_sims=n)
    else:
        print("\n  ⚠ Se encontraron errores en pasos anteriores. Revisa los logs.")
        print("  Puedes correr la simulación directamente con --only-sim si los modelos base están OK.")
