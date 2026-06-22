"""
download_afcon_only.py
Descarga SOLO los datos de AFCON 2023 (competition_id=1267, season_id=107)
sin re-descargar los demás torneos que ya están en disco.
"""

import sys
import importlib.util
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cargamos el módulo 02_download_statsbomb dinámicamente (nombre empieza con número)
spec = importlib.util.spec_from_file_location(
    "dl_sb",
    Path(__file__).parent / "02_download_statsbomb.py"
)
dl_sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl_sb)

from config import DATA_SB

if __name__ == "__main__":
    print("=== Descargando AFCON 2023 (competition_id=1267, season_id=107) ===")
    idx = dl_sb.download_competition(1267, 107, DATA_SB / "afcon_2023")
    print(f"\n[OK] Descarga completada: {len(idx)} partidos, {idx['n_events'].sum():,} eventos")
