"""
run_all.py — ejecuta toda la ingesta en orden
Uso: python ingestion/run_all.py
"""

import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
INGESTION_DIR = Path(__file__).parent

scripts = [
    ("01 - Resultados historicos (martj42)",   "01_download_martj42.py"),
    ("02 - StatsBomb eventos + 360 degrees",   "02_download_statsbomb.py"),
    ("03 - Elo historico de selecciones",       "03_download_elo.py"),
    ("04 - Rankings FIFA",                      "04_download_fifa_rankings.py"),
    ("05 - Valores de mercado (Transfermarkt)", "05_download_transfermarkt.py"),
    # 06 clima eliminado: en campo neutral afecta a ambos equipos por igual
]

print("=" * 60)
print("  WC PREDICTOR — PIPELINE DE INGESTA COMPLETO")
print("=" * 60)

failed = []
for label, script_name in scripts:
    script_path = INGESTION_DIR / script_name
    print(f"\n▶  {label}")
    print("-" * 60)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=False,   # muestra output en tiempo real
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"[OK]  Completado en {elapsed:.1f}s")
        else:
            print(f"[ERROR]  Terminó con código {result.returncode} en {elapsed:.1f}s")
            failed.append(label)
    except Exception as e:
        print(f"[ERROR]  ERROR al lanzar {script_name}: {e}")
        failed.append(label)

print("\n" + "=" * 60)
if failed:
    print(f"[AVISO]  {len(failed)} pasos fallaron:")
    for f in failed:
        print(f"   - {f}")
else:
    print("[OK]  Ingesta completa. Todos los datos descargados.")
print("=" * 60)
print("\nSiguiente paso: python features/build_master_features.py")
