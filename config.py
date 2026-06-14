"""
config.py — rutas y parámetros globales del proyecto
Todos los scripts importan desde aquí. Cambias una sola vez, aplica en todo.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Raíz del proyecto ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Datos ──────────────────────────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_SB         = DATA_RAW / "statsbomb"
DATA_FEATURES   = ROOT / "data" / "features"
DATA_MODEL      = ROOT / "data" / "model"

# ── Archivos raw ───────────────────────────────────────────────────────────────
INTL_RESULTS    = DATA_RAW / "international_results.parquet"
ELO_HISTORICAL  = DATA_RAW / "elo_historical.parquet"
FIFA_RANKINGS   = DATA_RAW / "fifa_rankings.parquet"
SQUAD_VALUES    = DATA_RAW / "squad_values.parquet"

# ── Archivos de features ───────────────────────────────────────────────────────
TEAM_SNAPSHOT   = DATA_FEATURES / "team_snapshot.parquet"
XG_DERIVED      = DATA_FEATURES / "xg_derived.parquet"
FORM_ROLLING    = DATA_FEATURES / "form_rolling.parquet"
H2H_HISTORY     = DATA_FEATURES / "h2h_history.parquet"
CONTEXT_FEAT    = DATA_FEATURES / "context_features.parquet"

# ── Archivos de modelo ─────────────────────────────────────────────────────────
TRAINING_SET    = DATA_MODEL / "training_set.parquet"
VALIDATION_SET  = DATA_MODEL / "validation_set.parquet"
WC2026_INPUT    = DATA_MODEL / "wc2026_prediction_input.parquet"

# ── StatsBomb competition IDs ──────────────────────────────────────────────────
# IDs verificados contra StatsBomb Open Data (github.com/statsbomb/open-data)
# Formato: (competition_id, season_id, output_dir)
STATSBOMB_COMPS = [
    (43,  106, DATA_SB / "wc_2022"),          # FIFA World Cup 2022
    (43,    3, DATA_SB / "wc_2018"),          # FIFA World Cup 2018
    (55,  282, DATA_SB / "euro_2024"),        # UEFA Euro 2024
    (55,   43, DATA_SB / "euro_2020"),        # UEFA Euro 2020
    (223, 282, DATA_SB / "copa_america_2024"),# Copa América 2024
    (1267,107, DATA_SB / "afcon_2023"),       # Africa Cup of Nations 2023 ✓ Open Data
]

# ── Supabase ───────────────────────────────────────────────────────────────────
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")

# ── Parámetros del modelo ──────────────────────────────────────────────────────
ELO_K = {"world_cup": 60, "qualifier": 40, "friendly": 20, "default": 30}
ELO_HOME_ADVANTAGE  = 100

FORM_WINDOW         = 10
FORM_DECAY          = 0.85

DIXON_COLES_RHO     = -0.13

MONTE_CARLO_RUNS    = 100_000

# ── Splits temporales ──────────────────────────────────────────────────────────
TRAIN_END_DATE      = "2021-07-11"
VALIDATION_END_DATE = "2024-07-14"

# ── Utilidad ───────────────────────────────────────────────────────────────────
def ensure_dirs():
    dirs = [DATA_RAW, DATA_SB, DATA_FEATURES, DATA_MODEL,
            ROOT / "mlflow" / "mlruns"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print(f"OK — ROOT: {ROOT}")
