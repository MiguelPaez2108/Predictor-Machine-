"""
run_final.py — Orquestador completo del WC2026 Predictor
=========================================================
Ejecuta todo el pipeline de predicción en secuencia:
  1. Genera features en vivo (wc2026_live_snapshot.parquet)
  2. Limpia caches del simulador (fuerza recarga de datos)
  3. Corre simulación Monte Carlo con llave real
  4. Imprime y guarda resultados

Uso:
  python run_final.py                    # 10,000 sims (default)
  python run_final.py --sims 50000       # 50,000 sims
  python run_final.py --quick            # 1,000 sims (rápido)
  python run_final.py --no-real-bracket  # sin llave fija (simula todo)
"""

import sys
import time
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_FEATURES, DATA_MODEL, ensure_dirs


def step_1_build_live_features():
    """Genera el snapshot de features en vivo desde Sofascore."""
    print("\n" + "=" * 60)
    print("  PASO 1: Generar features en vivo (Sofascore)")
    print("=" * 60)

    from features.build_wc2026_live_features import build_wc2026_live_features

    snapshot = build_wc2026_live_features()
    out_path = DATA_FEATURES / "wc2026_live_snapshot.parquet"
    snapshot.to_parquet(out_path, index=False)
    print(f"  [OK] {len(snapshot)} equipos → {out_path}")
    return snapshot


def step_2_clear_caches():
    """Limpia los caches del simulador para forzar recarga."""
    print("\n" + "=" * 60)
    print("  PASO 2: Limpiar caches del simulador")
    print("=" * 60)

    import simulation.match_simulator as ms

    ms._LOADED_MODELS = None
    ms._LAMBDA_CACHE.clear()
    ms._MATCH_PROB_CACHE.clear()
    ms._LAMBDA_CACHE_CAL.clear()
    ms._STATIC_DATA = None

    print("  [OK] Caches limpiados: modelos, lambdas, probabilidades, datos estáticos")


def step_3_simulate(n_sims: int, seed: int, use_real_bracket: bool):
    """Corre la simulación Monte Carlo."""
    print("\n" + "=" * 60)
    print(f"  PASO 3: Simulación Monte Carlo ({n_sims:,} runs)")
    print("=" * 60)

    from simulation.tournament import (
        simulate_wc2026,
        print_champion_table,
        print_group_predictions,
        save_results,
    )

    results = simulate_wc2026(
        n_sims=n_sims,
        seed=seed,
        verbose=True,
        use_real_bracket=use_real_bracket,
    )

    if results is None:
        print("  ERROR: No hay modelos entrenados.")
        return None

    # Imprimir resultados
    print_champion_table(results, top_n=48)
    print_group_predictions(results)

    # Guardar
    save_results(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="WC2026 Predictor — Pipeline completo"
    )
    parser.add_argument("--sims", type=int, default=10_000,
                        help="Número de simulaciones (default: 10,000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido: 1,000 simulaciones")
    parser.add_argument("--no-real-bracket", action="store_true",
                        dest="no_real_bracket",
                        help="No usar llave real (simular grupos + eliminatoria)")
    parser.add_argument("--skip-features", action="store_true",
                        dest="skip_features",
                        help="Saltar la generación de features en vivo")
    args = parser.parse_args()

    n_sims = 1_000 if args.quick else args.sims
    use_real_bracket = not args.no_real_bracket

    t_total = time.time()

    print("\n" + "█" * 60)
    print("  WC2026 PREDICTOR — PIPELINE COMPLETO")
    print("█" * 60)
    print(f"  Simulaciones: {n_sims:,}")
    print(f"  Llave real:   {'SÍ' if use_real_bracket else 'NO'}")
    print(f"  Seed:         {args.seed}")

    # Paso 1: Features en vivo
    if not args.skip_features:
        step_1_build_live_features()
    else:
        print("\n  [SKIP] Features en vivo (--skip-features)")

    # Paso 2: Limpiar caches
    step_2_clear_caches()

    # Paso 3: Simulación
    results = step_3_simulate(n_sims, args.seed, use_real_bracket)

    elapsed = time.time() - t_total
    print("\n" + "█" * 60)
    print(f"  PIPELINE COMPLETADO en {elapsed:.1f}s")
    print("█" * 60 + "\n")

    return results


if __name__ == "__main__":
    main()
