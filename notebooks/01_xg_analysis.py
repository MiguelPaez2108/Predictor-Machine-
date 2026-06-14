"""
01_xg_analysis.py — Visualizaciones de xG y Tiros con mplsoccer
================================================================
Genera heatmaps, shot maps y gráficos de calibración para analizar
los datos de StatsBomb y validar nuestro modelo.

Basado en la chuleta de statsbombpy (Datogami).

Uso:
  python notebooks/01_xg_analysis.py --team Argentina
  python notebooks/01_xg_analysis.py --tournament "FIFA World Cup" --year 2022
  python notebooks/01_xg_analysis.py --all-teams
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_SB, DATA_FEATURES, DATA_MODEL, ROOT, ensure_dirs

# Intentar importar mplsoccer
try:
    from mplsoccer import Pitch, VerticalPitch, FontManager
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False
    print("ADVERTENCIA: mplsoccer no disponible. pip install mplsoccer")


# ─────────────────────────────────────────────────────────────────────────────
# Paleta de colores del proyecto
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg"         : "#0D1117",
    "bg_secondary": "#161B22",
    "pitch"      : "#1A2332",
    "lines"      : "#2A3A4A",
    "goal"       : "#06D6A0",       # gol → verde
    "no_goal"    : "#E84855",       # no gol → rojo
    "xg_low"     : "#3A86FF",       # xG bajo → azul
    "xg_high"    : "#FF6B2B",       # xG alto → naranja
    "text"       : "#E6EDF3",
    "accent"     : "#58A6FF",
    "team_a"     : "#06D6A0",
    "team_b"     : "#FF6B2B",
}

FIGURES_DIR = ROOT / "notebooks" / "figures"


def setup_style():
    """Configura el estilo global de matplotlib."""
    plt.rcParams.update({
        "figure.facecolor"  : COLORS["bg"],
        "axes.facecolor"    : COLORS["bg_secondary"],
        "axes.edgecolor"    : COLORS["lines"],
        "text.color"        : COLORS["text"],
        "axes.labelcolor"   : COLORS["text"],
        "xtick.color"       : COLORS["text"],
        "ytick.color"       : COLORS["text"],
        "grid.color"        : COLORS["lines"],
        "grid.alpha"        : 0.3,
        "font.family"       : "monospace",
        "figure.dpi"        : 120,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos de eventos StatsBomb
# ─────────────────────────────────────────────────────────────────────────────

def load_team_shots(team_name: str,
                    tournament: str = None,
                    year: int = None) -> pd.DataFrame:
    """
    Carga todos los tiros de un equipo desde los parquets de eventos StatsBomb.
    """
    all_shots = []

    for folder in DATA_SB.iterdir():
        if not folder.is_dir():
            continue

        index_path = folder / "index.parquet"
        if not index_path.exists():
            continue

        index = pd.read_parquet(index_path)

        for _, row in index.iterrows():
            if team_name not in [row.get("home_team", ""), row.get("away_team", "")]:
                continue

            mid      = int(row["match_id"])
            evt_path = folder / f"events_{mid}.parquet"
            if not evt_path.exists():
                continue

            evts = pd.read_parquet(evt_path)
            shots = evts[
                (evts["type"] == "Shot") &
                (evts["team"] == team_name)
            ].copy()

            if shots.empty:
                continue

            # Extraer coordenadas
            if "location" in shots.columns:
                shots["x"] = shots["location"].apply(
                    lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
                shots["y"] = shots["location"].apply(
                    lambda l: l[1] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
            if "shot_end_location" in shots.columns:
                shots["x_end"] = shots["shot_end_location"].apply(
                    lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
                shots["y_end"] = shots["shot_end_location"].apply(
                    lambda l: l[1] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )

            shots["tournament_folder"] = folder.name
            shots["match_id"]          = mid
            all_shots.append(shots)

    if not all_shots:
        return pd.DataFrame()

    result = pd.concat(all_shots, ignore_index=True)

    if tournament:
        result = result[result["tournament_folder"].str.contains(
            tournament.lower().replace(" ", "_"), case=False, na=False
        )]

    return result


def load_team_passes(team_name: str) -> pd.DataFrame:
    """Carga todos los pases de un equipo."""
    all_passes = []

    for folder in DATA_SB.iterdir():
        if not folder.is_dir():
            continue
        index_path = folder / "index.parquet"
        if not index_path.exists():
            continue
        index = pd.read_parquet(index_path)
        for _, row in index.iterrows():
            if team_name not in [row.get("home_team", ""), row.get("away_team", "")]:
                continue
            mid      = int(row["match_id"])
            evt_path = folder / f"events_{mid}.parquet"
            if not evt_path.exists():
                continue
            evts   = pd.read_parquet(evt_path)
            passes = evts[
                (evts["type"] == "Pass") & (evts["team"] == team_name)
            ].copy()
            if passes.empty:
                continue
            if "location" in passes.columns:
                passes["x"] = passes["location"].apply(
                    lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
                passes["y"] = passes["location"].apply(
                    lambda l: l[1] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
            if "pass_end_location" in passes.columns:
                passes["x_end"] = passes["pass_end_location"].apply(
                    lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
                passes["y_end"] = passes["pass_end_location"].apply(
                    lambda l: l[1] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
                )
            all_passes.append(passes)

    return pd.concat(all_passes, ignore_index=True) if all_passes else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Shot Map con xG (burbuja proporcional a xG)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shot_map(team_name: str, shots: pd.DataFrame,
                  title: str = None, save: bool = True):
    """
    Shot map: cada disparo como círculo cuyo tamaño ∝ xG.
    Verde = gol, Rojo = no gol.
    Usa VerticalPitch con half=True (solo campo atacante).
    """
    if not MPLSOCCER_AVAILABLE:
        print("mplsoccer no disponible")
        return

    if shots.empty or "x" not in shots.columns:
        print(f"  Sin datos de tiros para {team_name}")
        return

    shots = shots.dropna(subset=["x", "y"])

    xg_col     = "shot_statsbomb_xg"
    goals      = shots[shots.get("shot_outcome", pd.Series()) == "Goal"] \
        if "shot_outcome" in shots.columns else pd.DataFrame()
    non_goals  = shots[shots.get("shot_outcome", pd.Series()) != "Goal"] \
        if "shot_outcome" in shots.columns else shots

    setup_style()
    pitch = VerticalPitch(
        pitch_type  = "statsbomb",
        half        = True,
        pitch_color = COLORS["pitch"],
        line_color  = COLORS["lines"],
        linewidth   = 1.5,
    )
    fig, ax = pitch.draw(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS["bg"])

    # No goles
    if not non_goals.empty:
        xg_size = 200 + non_goals[xg_col] * 1500 \
            if xg_col in non_goals.columns else 200
        pitch.scatter(
            non_goals["x"], non_goals["y"],
            ax      = ax,
            s       = xg_size,
            c       = COLORS["no_goal"],
            alpha   = 0.6,
            edgecolors = "#FFFFFF",
            linewidths = 0.5,
            zorder  = 4,
            label   = "No gol"
        )

    # Goles
    if not goals.empty:
        xg_size = 250 + goals[xg_col] * 1500 \
            if xg_col in goals.columns else 250
        pitch.scatter(
            goals["x"], goals["y"],
            ax         = ax,
            s          = xg_size,
            c          = COLORS["goal"],
            alpha       = 0.9,
            edgecolors = "#FFFFFF",
            linewidths = 1.0,
            zorder     = 5,
            marker     = "*",
            label      = "Gol"
        )

    # Estadísticas en el título
    total_xg = shots[xg_col].sum() if xg_col in shots.columns else 0
    n_goals  = len(goals)
    n_shots  = len(shots)

    title_text = title or f"{team_name} — Shot Map"
    ax.set_title(
        f"{title_text}\n"
        f"Tiros: {n_shots}  |  Goles: {n_goals}  |  xG total: {total_xg:.2f}  |  "
        f"xG/tiro: {total_xg/n_shots:.3f}" if n_shots > 0 else title_text,
        color    = COLORS["text"],
        fontsize = 12,
        pad      = 10
    )

    # Leyenda
    legend_elements = [
        mpatches.Patch(color=COLORS["goal"],    label=f"Gol ({n_goals})"),
        mpatches.Patch(color=COLORS["no_goal"], label=f"No gol ({n_shots - n_goals})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              facecolor=COLORS["bg_secondary"], edgecolor=COLORS["lines"],
              labelcolor=COLORS["text"], fontsize=9)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = FIGURES_DIR / f"shotmap_{team_name.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"  Guardado: {fname}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Heatmap de densidad de acciones
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_heatmap(team_name: str, events_df: pd.DataFrame,
                        action_type: str = "Pressure",
                        title: str = None, save: bool = True):
    """
    Heatmap KDE de la densidad de un tipo de acción.
    Útil para ver presión defensiva alta, zonas de pase, etc.
    """
    if not MPLSOCCER_AVAILABLE:
        return

    actions = events_df[
        (events_df["team"] == team_name) &
        (events_df["type"] == action_type)
    ].copy()

    if actions.empty or "location" not in actions.columns:
        print(f"  Sin datos de {action_type} para {team_name}")
        return

    actions["x"] = actions["location"].apply(
        lambda l: l[0] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
    )
    actions["y"] = actions["location"].apply(
        lambda l: l[1] if isinstance(l, (list, tuple, np.ndarray)) else np.nan
    )
    actions = actions.dropna(subset=["x", "y"])

    setup_style()
    pitch = Pitch(
        pitch_type  = "statsbomb",
        pitch_color = COLORS["pitch"],
        line_color  = COLORS["lines"],
        linewidth   = 1.5,
    )
    fig, ax = pitch.draw(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS["bg"])

    pitch.kdeplot(
        actions["x"], actions["y"],
        ax         = ax,
        cmap       = "YlOrRd",
        fill       = True,
        levels     = 100,
        alpha      = 0.75,
        bw_adjust  = 0.7,
        zorder     = 2,
    )

    ax.set_title(
        title or f"{team_name} — Densidad de {action_type} (n={len(actions)})",
        color    = COLORS["text"],
        fontsize = 12,
        pad      = 10
    )

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = FIGURES_DIR / f"heatmap_{team_name.replace(' ', '_')}_{action_type.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"  Guardado: {fname}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Comparativa xG por equipo (barras horizontales)
# ─────────────────────────────────────────────────────────────────────────────

def plot_xg_comparison(tournament: str = "FIFA World Cup", year: int = 2022,
                       top_n: int = 20, save: bool = True):
    """
    Barras de xG total y goles reales por equipo en un torneo.
    Compara xG esperado vs goles anotados (conversión).
    """
    xg_path = DATA_FEATURES / "xg_derived.parquet"
    if not xg_path.exists():
        print("  Sin xg_derived.parquet. Ejecutar build_xg_features.py primero.")
        return

    df = pd.read_parquet(xg_path)

    if "tournament" in df.columns:
        df = df[df["tournament"].str.contains(tournament, case=False, na=False)]
    if "year" in df.columns and year:
        df = df[df["year"] == year]

    if df.empty:
        print(f"  Sin datos para {tournament} {year}")
        return

    team_agg = (
        df.groupby("team")
        .agg(xg=("xg", "sum"), goals=("goals", "sum"),
             xga=("xga", "sum"), goals_against=("goals_against", "sum"))
        .reset_index()
        .sort_values("xg", ascending=False)
        .head(top_n)
    )

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Gráfico 1: xG vs Goles ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(COLORS["bg_secondary"])
    y_pos = np.arange(len(team_agg))

    bars_xg = ax.barh(y_pos + 0.2, team_agg["xg"],    height=0.4,
                      color=COLORS["xg_low"],  alpha=0.8, label="xG")
    bars_g  = ax.barh(y_pos - 0.2, team_agg["goals"],  height=0.4,
                      color=COLORS["team_a"],  alpha=0.8, label="Goles reales")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(team_agg["team"], fontsize=9)
    ax.set_xlabel("Goles / xG", color=COLORS["text"])
    ax.set_title(f"xG vs Goles — {tournament} {year}",
                 color=COLORS["text"], fontsize=11, pad=8)
    ax.legend(facecolor=COLORS["bg_secondary"], labelcolor=COLORS["text"])
    ax.spines[["top", "right"]].set_visible(False)

    # ── Gráfico 2: Conversión (goles/xG) ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg_secondary"])

    conversion = (team_agg["goals"] / team_agg["xg"].replace(0, np.nan)).fillna(0)
    colors_conv = [COLORS["goal"] if c >= 1 else COLORS["no_goal"]
                   for c in conversion]

    ax2.barh(y_pos, conversion, color=colors_conv, alpha=0.8)
    ax2.axvline(1.0, color=COLORS["text"], linestyle="--",
                alpha=0.5, linewidth=1, label="Conversión perfecta (1.0)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(team_agg["team"], fontsize=9)
    ax2.set_xlabel("Ratio de conversión (Goles / xG)", color=COLORS["text"])
    ax2.set_title(f"Conversión — {tournament} {year}",
                  color=COLORS["text"], fontsize=11, pad=8)
    ax2.legend(facecolor=COLORS["bg_secondary"], labelcolor=COLORS["text"])
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Análisis xG — {tournament} {year}",
                 color=COLORS["accent"], fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = FIGURES_DIR / f"xg_comparison_{tournament.replace(' ', '_')}_{year}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"  Guardado: {fname}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Curva de calibración del modelo
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(save: bool = True):
    """
    Reliability diagram: probabilidad predicha vs frecuencia real.
    Compara el modelo antes y después de calibrar.
    """
    val_path = DATA_MODEL / "validation_set.parquet"
    if not val_path.exists():
        print("  Sin validation set. Ejecutar build_master_features.py primero.")
        return

    val = pd.read_parquet(val_path)
    val = val.dropna(subset=["target_result"])

    # Intentar cargar predicciones de modelos entrenados
    import pickle

    rf_path  = DATA_MODEL / "random_forest.pkl"
    cal_path = DATA_MODEL / "calibrator.pkl"

    if not rf_path.exists():
        print("  Sin modelos entrenados. Ejecutar models/ensemble.py --train primero.")
        return

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    preds_raw = rf_model.predict_proba_df(val)

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    classes   = [("H", "p_home_win", "Victoria Local"),
                 ("D", "p_draw",     "Empate"),
                 ("A", "p_away_win", "Victoria Visitante")]

    for idx, (cls, col, label) in enumerate(classes):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg_secondary"])

        y_true = (val["target_result"] == cls).astype(float).values
        y_pred = preds_raw[col].values

        # Calcular bins de calibración
        n_bins = 10
        bins   = np.linspace(0, 1, n_bins + 1)
        bin_centers, freq_real = [], []

        for b in range(n_bins):
            mask = (y_pred >= bins[b]) & (y_pred < bins[b + 1])
            if mask.sum() >= 5:
                bin_centers.append(y_pred[mask].mean())
                freq_real.append(y_true[mask].mean())

        # Línea perfecta
        ax.plot([0, 1], [0, 1], "--", color=COLORS["lines"],
                alpha=0.7, linewidth=1, label="Calibración perfecta")

        # Curva del modelo
        if bin_centers:
            ax.plot(bin_centers, freq_real, "o-",
                    color=COLORS["accent"], linewidth=2,
                    markersize=6, label="Modelo RF")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Probabilidad predicha", color=COLORS["text"])
        ax.set_ylabel("Frecuencia real",        color=COLORS["text"])
        ax.set_title(label, color=COLORS["text"], fontsize=10)
        ax.legend(facecolor=COLORS["bg_secondary"], labelcolor=COLORS["text"],
                  fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Curvas de Calibración — Random Forest",
                 color=COLORS["accent"], fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = FIGURES_DIR / "calibration_curve.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"  Guardado: {fname}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualizaciones xG y táctica con mplsoccer"
    )
    parser.add_argument("--team", type=str, default=None,
                        help="Equipo a analizar (ej: Argentina)")
    parser.add_argument("--tournament", type=str, default="FIFA World Cup",
                        help="Torneo (default: FIFA World Cup)")
    parser.add_argument("--year", type=int, default=2022,
                        help="Año del torneo (default: 2022)")
    parser.add_argument("--action", type=str, default="Pressure",
                        help="Tipo de acción para heatmap (default: Pressure)")
    parser.add_argument("--calibration", action="store_true",
                        help="Generar curvas de calibración del modelo")
    parser.add_argument("--xg-comparison", action="store_true",
                        help="Comparativa xG por equipo en el torneo")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar gráfico en pantalla (default: guardar)")
    args = parser.parse_args()

    ensure_dirs()
    save = not args.show

    if args.team:
        print(f"\n→ Shot map: {args.team}")
        shots = load_team_shots(args.team)
        if not shots.empty:
            plot_shot_map(args.team, shots, save=save)
        else:
            print(f"  Sin datos de tiros para {args.team}")

        print(f"\n→ Heatmap {args.action}: {args.team}")
        # Cargar eventos completos del equipo (más costoso)
        passes = load_team_passes(args.team)

    if args.xg_comparison:
        print(f"\n→ Comparativa xG: {args.tournament} {args.year}")
        plot_xg_comparison(args.tournament, args.year, save=save)

    if args.calibration:
        print(f"\n→ Curvas de calibración del modelo")
        plot_calibration_curve(save=save)

    if not any([args.team, args.xg_comparison, args.calibration]):
        # Por defecto: generar comparativa xG del WC2022
        print("Generando análisis por defecto: comparativa xG WC2022...")
        plot_xg_comparison("FIFA World Cup", 2022, save=save)
        plot_calibration_curve(save=save)


if __name__ == "__main__":
    main()
