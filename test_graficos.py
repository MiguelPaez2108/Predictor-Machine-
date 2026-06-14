"""
test_graficos.py — Prueba real de visualizaciones con datos WC2022
===================================================================
Lee los parquets de eventos StatsBomb del WC2022 que ya están descargados
y genera 4 gráficos directamente. No requiere que el pipeline completo haya corrido.

Genera:
  notebooks/figures/test_shotmap_Argentina.png
  notebooks/figures/test_shotmap_France.png
  notebooks/figures/test_heatmap_pressure.png
  notebooks/figures/test_xg_comparison_wc2022.png
  notebooks/figures/test_xg_timeline.png

Uso:
  python test_graficos.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sin ventana, guarda directo a PNG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# Rutas
ROOT       = Path(__file__).parent
SB_WC2022  = ROOT / "data" / "raw" / "statsbomb" / "wc_2022"
FIGURES    = ROOT / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── Paleta de colores ────────────────────────────────────────────────────────
BG       = "#0D1117"
BG2      = "#161B22"
PITCH_C  = "#1A2332"
LINES_C  = "#2A3A4A"
GOAL_C   = "#06D6A0"
NOGOL_C  = "#E84855"
TEXT_C   = "#E6EDF3"
ACCENT   = "#58A6FF"
ORANGE   = "#FF6B2B"
BLUE     = "#3A86FF"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor"  : BG2,
    "text.color"      : TEXT_C,
    "axes.labelcolor" : TEXT_C,
    "xtick.color"     : TEXT_C,
    "ytick.color"     : TEXT_C,
    "axes.edgecolor"  : LINES_C,
    "font.family"     : "monospace",
    "figure.dpi"      : 130,
})

# ─── Verificar mplsoccer ─────────────────────────────────────────────────────
try:
    from mplsoccer import Pitch, VerticalPitch
    HAS_MPLSOCCER = True
    print("  ✓ mplsoccer disponible")
except ImportError:
    HAS_MPLSOCCER = False
    print("  ✗ mplsoccer no disponible — instalar: pip install mplsoccer")


# ─── Carga de datos ───────────────────────────────────────────────────────────
def load_all_wc2022_events() -> pd.DataFrame:
    """Carga todos los eventos del WC2022 en un DataFrame unificado."""
    index = pd.read_parquet(SB_WC2022 / "index.parquet")
    matches = pd.read_parquet(SB_WC2022 / "matches.parquet")

    all_evts = []
    for _, row in index.iterrows():
        mid  = int(row["match_id"])
        path = SB_WC2022 / f"events_{mid}.parquet"
        if not path.exists():
            continue
        evts = pd.read_parquet(path)
        evts["match_id"]   = mid
        evts["home_team"]  = row["home_team"]
        evts["away_team"]  = row["away_team"]
        all_evts.append(evts)

    df = pd.concat(all_evts, ignore_index=True)
    print(f"  Eventos cargados: {len(df):,} eventos de {len(index)} partidos")
    return df, matches


def extract_xy(loc):
    """Extrae x, y de location (lista, tupla o numpy array [x,y])."""
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 1: Shot Map de un equipo (Argentina vs Francia — Final WC2022)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_shotmap_final(df_all: pd.DataFrame, matches: pd.DataFrame):
    print("\n[1/5] Shot map — Final Argentina vs Francia...")

    # Encontrar la final
    final_row = matches[matches["competition_stage"].str.contains("Final", na=False)]
    if final_row.empty:
        final_row = matches.tail(1)
    mid = int(final_row.iloc[0]["match_id"])

    match_evts = df_all[df_all["match_id"] == mid]
    shots = match_evts[match_evts["type"] == "Shot"].copy()

    shots["x"] = shots["location"].apply(lambda l: extract_xy(l)[0])
    shots["y"] = shots["location"].apply(lambda l: extract_xy(l)[1])
    shots = shots.dropna(subset=["x", "y"])

    teams = shots["team"].unique().tolist()

    if len(teams) == 0:
        print("  Sin tiros en la final — verificar datos")
        return

    if not HAS_MPLSOCCER:
        print("  → Sin mplsoccer, generando versión básica con matplotlib...")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_facecolor(PITCH_C)
        # Campo simplificado
        ax.add_patch(plt.Rectangle((0, 0), 120, 80, fill=False, edgecolor=LINES_C, lw=2))
        ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, edgecolor=LINES_C, lw=1.5))
        ax.set_xlim(0, 120); ax.set_ylim(0, 80)
        ax.set_aspect("equal")
        if len(teams) >= 2:
            colors_t = {teams[0]: GOAL_C, teams[1]: ORANGE}
        elif len(teams) == 1:
            colors_t = {teams[0]: GOAL_C}
        else:
            return
        for team in teams:
            ts = shots[shots["team"] == team]
            goals = ts[ts["shot_outcome"] == "Goal"] if "shot_outcome" in ts.columns else pd.DataFrame()
            non_g = ts[ts["shot_outcome"] != "Goal"] if "shot_outcome" in ts.columns else ts
            c = colors_t.get(team, ACCENT)
            xg_col = "shot_statsbomb_xg"
            if not non_g.empty:
                s = 80 + non_g[xg_col] * 800 if xg_col in non_g.columns else 80
                ax.scatter(non_g["x"], non_g["y"], s=s, c=c, alpha=0.5, edgecolors="white", lw=0.5, zorder=3)
            if not goals.empty:
                s = 200 + goals[xg_col] * 800 if xg_col in goals.columns else 200
                ax.scatter(goals["x"], goals["y"], s=s, c=c, marker="*", alpha=0.95, edgecolors="white", lw=0.8, zorder=5)
        ax.set_title(f"Shot Map — Final WC2022\n{' vs '.join(teams)}", color=TEXT_C, fontsize=13, pad=10)
        legend_el = [mpatches.Patch(color=colors_t.get(t, ACCENT), label=t) for t in teams]
        ax.legend(handles=legend_el, facecolor=BG2, labelcolor=TEXT_C, fontsize=9, loc="upper left")
        fig.patch.set_facecolor(BG)
        plt.tight_layout()
        out = FIGURES / "test_shotmap_final_basico.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  ✓ Guardado: {out}")
        return

    # Con mplsoccer
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    colors_t = {teams[0]: GOAL_C, teams[1]: ORANGE} if len(teams) >= 2 else ({teams[0]: GOAL_C} if len(teams) == 1 else {})
    if not colors_t:
        return


    for ax_idx, team in enumerate(teams[:2]):
        ax = axes[ax_idx]
        ts = shots[shots["team"] == team].copy()
        goals = ts[ts["shot_outcome"] == "Goal"] if "shot_outcome" in ts.columns else pd.DataFrame()
        non_g = ts[ts["shot_outcome"] != "Goal"] if "shot_outcome" in ts.columns else ts

        pitch = VerticalPitch(pitch_type="statsbomb", half=True,
                              pitch_color=PITCH_C, line_color=LINES_C, linewidth=1.5)
        pitch.draw(ax=ax)
        ax.set_facecolor(PITCH_C)

        xg_col = "shot_statsbomb_xg"
        c = colors_t.get(team, ACCENT)

        if not non_g.empty:
            s_arr = 100 + non_g[xg_col].fillna(0) * 1200 if xg_col in non_g.columns else 100
            pitch.scatter(non_g["x"], non_g["y"], ax=ax, s=s_arr, c=c,
                         alpha=0.55, edgecolors="white", linewidths=0.5, zorder=4)

        if not goals.empty:
            s_arr = 250 + goals[xg_col].fillna(0) * 1200 if xg_col in goals.columns else 250
            pitch.scatter(goals["x"], goals["y"], ax=ax, s=s_arr, c=c,
                         alpha=0.95, edgecolors="white", linewidths=1, zorder=5, marker="*")

        xg_total = float(ts[xg_col].sum()) if xg_col in ts.columns else 0
        n_goals  = len(goals)
        ax.set_title(f"{team}\n{len(ts)} tiros  |  {n_goals} goles  |  xG: {xg_total:.2f}",
                     color=c, fontsize=11, pad=8, fontweight="bold")

    fig.suptitle("Shot Map — Final WC2022\nArgentina vs Francia",
                 color=TEXT_C, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES / "test_shotmap_final.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 2: Heatmap de presión de Argentina en el WC2022
# ─────────────────────────────────────────────────────────────────────────────
def grafico_heatmap_presion(df_all: pd.DataFrame, team: str = "Argentina"):
    print(f"\n[2/5] Heatmap de Presión — {team} en WC2022...")

    presiones = df_all[
        (df_all["team"] == team) & (df_all["type"] == "Pressure")
    ].copy()

    if presiones.empty:
        print(f"  Sin datos de presión para {team}")
        return

    presiones["x"] = presiones["location"].apply(lambda l: extract_xy(l)[0])
    presiones["y"] = presiones["location"].apply(lambda l: extract_xy(l)[1])
    presiones = presiones.dropna(subset=["x", "y"])
    print(f"  Acciones de presión: {len(presiones):,}")

    if not HAS_MPLSOCCER:
        # Versión básica: scatter coloreado
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_facecolor(PITCH_C)
        ax.add_patch(plt.Rectangle((0, 0), 120, 80, fill=False, edgecolor=LINES_C, lw=2))
        ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal")
        ax.scatter(presiones["x"], presiones["y"], c=ORANGE, alpha=0.1, s=20, zorder=3)
        ax.set_title(f"Densidad de Presión — {team} WC2022\n(n={len(presiones):,} acciones)",
                     color=TEXT_C, fontsize=12)
        fig.patch.set_facecolor(BG)
        plt.tight_layout()
        out = FIGURES / f"test_heatmap_{team.replace(' ', '_')}_basico.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  ✓ Guardado: {out}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)

    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_C,
                  line_color=LINES_C, linewidth=1.5)
    pitch.draw(ax=ax)

    pitch.kdeplot(presiones["x"], presiones["y"], ax=ax,
                  cmap="YlOrRd", fill=True, levels=80,
                  alpha=0.75, bw_adjust=0.65, zorder=2)

    ax.set_title(f"Mapa de Presión Defensiva — {team} | WC2022\n"
                 f"{len(presiones):,} acciones de presión en {len(df_all['match_id'].unique())} partidos",
                 color=TEXT_C, fontsize=12, pad=10, fontweight="bold")

    # Anotación: zona de mayor presión
    ax.text(0.02, 0.02, "Alta presión = rojo/naranja\nBloque bajo = calor en zona defensiva",
            transform=ax.transAxes, color=TEXT_C, fontsize=8, alpha=0.7,
            verticalalignment="bottom")

    plt.tight_layout()
    out = FIGURES / f"test_heatmap_{team.replace(' ', '_')}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 3: xG por equipo — todos los equipos del WC2022
# ─────────────────────────────────────────────────────────────────────────────
def grafico_xg_por_equipo(df_all: pd.DataFrame):
    print("\n[3/5] xG por equipo — WC2022 completo...")

    shots = df_all[df_all["type"] == "Shot"].copy()
    if shots.empty or "shot_statsbomb_xg" not in shots.columns:
        print("  Sin datos de xG")
        return

    shots["gol"] = (shots.get("shot_outcome", pd.Series("")) == "Goal").astype(int) \
        if "shot_outcome" in shots.columns else 0

    agg = (
        shots.groupby("team")
        .agg(
            xg     = ("shot_statsbomb_xg", "sum"),
            goals  = ("gol", "sum"),
            shots  = ("shot_statsbomb_xg", "count"),
        )
        .reset_index()
        .sort_values("xg", ascending=True)
    )
    agg["xg_per_shot"] = (agg["xg"] / agg["shots"]).round(3)
    agg["conversion"]  = (agg["goals"] / agg["xg"].replace(0, np.nan)).round(2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Análisis xG — FIFA World Cup 2022\nDatos StatsBomb Open Data",
                 color=ACCENT, fontsize=15, fontweight="bold", y=1.01)

    y = np.arange(len(agg))

    # Panel 1: xG total vs Goles
    ax = axes[0]
    ax.set_facecolor(BG2)
    ax.barh(y + 0.2, agg["xg"],   height=0.38, color=BLUE,    alpha=0.85, label="xG esperado")
    ax.barh(y - 0.2, agg["goals"], height=0.38, color=GOAL_C, alpha=0.85, label="Goles reales")
    ax.set_yticks(y); ax.set_yticklabels(agg["team"], fontsize=8)
    ax.set_xlabel("Goles / xG")
    ax.set_title("xG vs Goles reales", color=TEXT_C, fontsize=11)
    ax.legend(facecolor=BG2, labelcolor=TEXT_C, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Calidad de tiros (xG por tiro)
    ax2 = axes[1]
    ax2.set_facecolor(BG2)
    colors2 = [ORANGE if v > agg["xg_per_shot"].median() else BLUE for v in agg["xg_per_shot"]]
    ax2.barh(y, agg["xg_per_shot"], color=colors2, alpha=0.85)
    ax2.axvline(agg["xg_per_shot"].median(), color=TEXT_C, linestyle="--",
                alpha=0.5, lw=1, label=f"Mediana: {agg['xg_per_shot'].median():.3f}")
    ax2.set_yticks(y); ax2.set_yticklabels(agg["team"], fontsize=8)
    ax2.set_xlabel("xG por tiro (calidad de ocasiones)")
    ax2.set_title("Calidad de tiros\n(shot quality index)", color=TEXT_C, fontsize=11)
    ax2.legend(facecolor=BG2, labelcolor=TEXT_C, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: Conversión (goles/xG)
    ax3 = axes[2]
    ax3.set_facecolor(BG2)
    conv_clean = agg["conversion"].fillna(0)
    colors3 = [GOAL_C if c >= 1.0 else (ORANGE if c >= 0.7 else NOGOL_C) for c in conv_clean]
    ax3.barh(y, conv_clean, color=colors3, alpha=0.85)
    ax3.axvline(1.0, color=TEXT_C, linestyle="--", alpha=0.5, lw=1.5, label="Conversión = 1.0")
    ax3.set_yticks(y); ax3.set_yticklabels(agg["team"], fontsize=8)
    ax3.set_xlabel("Ratio Goles / xG")
    ax3.set_title("Conversión de xG\n(verde≥1, naranja≥0.7, rojo<0.7)", color=TEXT_C, fontsize=11)
    ax3.legend(facecolor=BG2, labelcolor=TEXT_C, fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "test_xg_por_equipo_wc2022.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 4: xG acumulado a lo largo del WC2022 (timeline)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_xg_timeline(df_all: pd.DataFrame, matches: pd.DataFrame):
    print("\n[4/5] xG timeline — Top 8 equipos WC2022...")

    shots = df_all[df_all["type"] == "Shot"].copy()
    if shots.empty or "shot_statsbomb_xg" not in shots.columns:
        print("  Sin datos")
        return

    # Añadir fecha del partido
    date_map = {}
    if "match_id" in matches.columns and "match_date" in matches.columns:
        date_map = dict(zip(matches["match_id"].astype(int),
                            pd.to_datetime(matches["match_date"])))
    shots["match_date"] = shots["match_id"].map(date_map)

    # Top 8 equipos por xG total
    top_teams = (
        shots.groupby("team")["shot_statsbomb_xg"].sum()
        .sort_values(ascending=False)
        .head(8).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor(BG2)
    fig.patch.set_facecolor(BG)

    cmap = plt.cm.get_cmap("tab10", len(top_teams))

    for i, team in enumerate(top_teams):
        ts = (
            shots[shots["team"] == team]
            .groupby("match_date")["shot_statsbomb_xg"]
            .sum()
            .sort_index()
            .cumsum()
            .reset_index()
        )
        if ts.empty:
            continue
        ax.plot(ts["match_date"], ts["shot_statsbomb_xg"],
                marker="o", markersize=5, linewidth=2,
                label=team, color=cmap(i), alpha=0.9)

    ax.set_title("xG Acumulado por Partido — Top 8 equipos WC2022",
                 color=TEXT_C, fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Fecha", color=TEXT_C)
    ax.set_ylabel("xG Acumulado", color=TEXT_C)
    ax.legend(facecolor=BG2, labelcolor=TEXT_C, fontsize=9,
              loc="upper left", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = FIGURES / "test_xg_timeline_wc2022.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 5: PPDA (presión) por equipo
# ─────────────────────────────────────────────────────────────────────────────
def grafico_ppda_equipos(df_all: pd.DataFrame, matches: pd.DataFrame):
    print("\n[5/5] PPDA por equipo — WC2022...")

    index = pd.read_parquet(SB_WC2022 / "index.parquet")
    records = []

    for _, row in index.iterrows():
        mid     = int(row["match_id"])
        match_e = df_all[df_all["match_id"] == mid]
        if match_e.empty:
            continue

        teams = [row["home_team"], row["away_team"]]
        for team in teams:
            opp = teams[1] if team == teams[0] else teams[0]

            # Pases completados del rival
            opp_passes = match_e[(match_e["team"] == opp) & (match_e["type"] == "Pass")]
            n_opp_pass = int(opp_passes["pass_outcome"].isna().sum()) \
                if "pass_outcome" in opp_passes.columns else len(opp_passes)

            # Acciones defensivas propias en campo rival (x > 60)
            def_acts = match_e[
                (match_e["team"] == team) &
                (match_e["type"].isin(["Pressure", "Tackle", "Interception", "Ball Recovery"]))
            ]
            if not def_acts.empty and "location" in def_acts.columns:
                xs = def_acts["location"].apply(
                    lambda l: l[0] if isinstance(l, (list, tuple)) else np.nan
                )
                n_def = int((xs > 60).sum())
            else:
                n_def = len(def_acts)

            ppda = round(n_opp_pass / n_def, 2) if n_def > 0 else np.nan
            records.append({"team": team, "ppda": ppda})

    if not records:
        print("  Sin datos de PPDA")
        return

    df_ppda = (
        pd.DataFrame(records)
        .groupby("team")["ppda"]
        .mean()
        .dropna()
        .sort_values()
        .reset_index()
    )
    df_ppda.columns = ["team", "ppda_avg"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(BG2)
    fig.patch.set_facecolor(BG)

    # PPDA bajo = presión alta (mejor defensivamente) → verde
    # PPDA alto = bloque bajo → rojo
    median_ppda = df_ppda["ppda_avg"].median()
    colors = [GOAL_C if v <= median_ppda else NOGOL_C for v in df_ppda["ppda_avg"]]

    bars = ax.barh(df_ppda["team"], df_ppda["ppda_avg"], color=colors, alpha=0.85)
    ax.axvline(median_ppda, color=TEXT_C, linestyle="--", alpha=0.5, lw=1.5,
               label=f"Mediana: {median_ppda:.1f}")

    # Anotaciones de valor
    for bar, val in zip(bars, df_ppda["ppda_avg"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left",
                color=TEXT_C, fontsize=8)

    ax.set_xlabel("PPDA (Passes Per Defensive Action)\n← Presión alta  |  Bloque bajo →",
                  color=TEXT_C)
    ax.set_title("Intensidad de Presión Defensiva — WC2022\n"
                 "PPDA bajo = presión alta (estilo Klopp/Guardiola)",
                 color=TEXT_C, fontsize=12, pad=10, fontweight="bold")
    ax.legend(facecolor=BG2, labelcolor=TEXT_C, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    legend_el = [
        mpatches.Patch(color=GOAL_C,  label="Presión alta (PPDA ≤ mediana)"),
        mpatches.Patch(color=NOGOL_C, label="Bloque bajo (PPDA > mediana)"),
    ]
    ax.legend(handles=legend_el, facecolor=BG2, labelcolor=TEXT_C, fontsize=9)

    plt.tight_layout()
    out = FIGURES / "test_ppda_equipos_wc2022.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 55)
    print("TEST DE GRÁFICOS — WC2022 StatsBomb")
    print("═" * 55)

    if not SB_WC2022.exists() or not (SB_WC2022 / "index.parquet").exists():
        print(f"ERROR: No se encontraron datos en {SB_WC2022}")
        print("Ejecutar primero: python ingestion/02_download_statsbomb.py")
        sys.exit(1)

    print("\nCargando todos los eventos del WC2022...")
    df_all, matches = load_all_wc2022_events()

    grafico_shotmap_final(df_all, matches)
    grafico_heatmap_presion(df_all, "Argentina")
    grafico_xg_por_equipo(df_all)
    grafico_xg_timeline(df_all, matches)
    grafico_ppda_equipos(df_all, matches)

    print(f"\n{'═'*55}")
    print(f"✓ 5 gráficos generados en: {FIGURES}")
    print("Abre la carpeta para ver los PNG:")
    print(f"  {FIGURES}")
    print(f"{'═'*55}")
