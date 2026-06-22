"""
diagnose_favorites.py — Diagnóstico de distorsión en las odds del Mundial
===========================================================================
Investiga por qué equipos como Japón o Uruguay aparecen por encima de
Francia pese a la diferencia de plantilla, Elo "real" y nivel histórico.

Inspecciona, en orden:
  1. Elo dinámico calculado por el sistema (el que REALMENTE usa el modelo)
     vs el Elo "oficial" de eloratings.net que tenés en elo.txt
  2. Parámetros ataque/defensa de Dixon-Coles por equipo
  3. Lambdas (goles esperados) en enfrentamientos directos clave
  4. Probabilidades 1X2 antes y después del meta-modelo y la calibración
  5. Peso real de delta_sv_log (valor de plantilla) en el RF y en el meta-modelo
  6. Distorsión del calibrador isotónico específicamente en la cola alta
     (favoritos con P > 0.45), que es donde más duele este tipo de error

Uso:
  python diagnose_favorites.py
  python diagnose_favorites.py --teams Argentina Brazil France Japan Uruguay Spain Colombia England Portugal Australia
"""

import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_FEATURES, DATA_MODEL


DEFAULT_TEAMS = [
    "Argentina", "Brazil", "Spain", "Colombia", "Japan",
    "Uruguay", "France", "England", "Australia", "Portugal",
]

# Elo "oficial" reportado por eloratings.net (el que vos pegaste en elo.txt)
# Lo usamos solo como referencia de contraste, NO lo usa el modelo.
ELO_OFICIAL_REF = {
    "Spain": 2129, "Argentina": 2128, "France": 2084, "England": 2055,
    "Colombia": 1998, "Brazil": 1978, "Portugal": 1967, "Japan": 1910,
    "Uruguay": 1870, "Australia": 1839,
}


def section(title):
    print(f"\n{'═'*78}\n  {title}\n{'═'*78}")


def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Elo dinámico calculado vs Elo oficial de referencia
# ─────────────────────────────────────────────────────────────────────────────

def diag_elo(teams):
    section("1. ELO DINÁMICO DEL SISTEMA vs ELO OFICIAL (eloratings.net)")

    elo_path = DATA_FEATURES / "elo_current.parquet"
    if not elo_path.exists():
        elo_path = DATA_RAW / "elo_current.parquet"

    if not elo_path.exists():
        print("  [AVISO] No se encontró elo_current.parquet. Ejecutar 03_download_elo.py")
        return

    elo = pd.read_parquet(elo_path)
    col_elo = "elo_current" if "elo_current" in elo.columns else "elo_current"
    elo_map = dict(zip(elo["team"], elo[col_elo]))

    print(f"  {'Equipo':18s} {'Elo SISTEMA (usado por el modelo)':>34s} {'Elo OFICIAL (referencia)':>26s} {'Δ':>8s}")
    print(f"  {'─'*18} {'─'*34} {'─'*26} {'─'*8}")

    rows = []
    for t in teams:
        e_sys = elo_map.get(t, np.nan)
        e_ref = ELO_OFICIAL_REF.get(t, np.nan)
        delta = e_sys - e_ref if not (np.isnan(e_sys) or np.isnan(e_ref)) else np.nan
        rows.append((t, e_sys, e_ref, delta))

    rows.sort(key=lambda r: -r[1] if not np.isnan(r[1]) else 9999)
    for t, e_sys, e_ref, delta in rows:
        flag = "  [AVISO] MUY POR DEBAJO" if (not np.isnan(delta) and delta < -100) else ""
        print(f"  {t:18s} {e_sys:>34.1f} {e_ref:>26.1f} {delta:>+8.1f}{flag}")

    print("\n  → Si el Elo SISTEMA de Francia está muy por debajo del OFICIAL,")
    print("    el problema está en el cálculo de 03_download_elo.py (K-factors,")
    print("    home_advantage en neutral, o exceso de amistosos rotados).")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parámetros Dixon-Coles
# ─────────────────────────────────────────────────────────────────────────────

def diag_dixon_coles(teams):
    section("2. PARÁMETROS DIXON-COLES (ataque / defensa)")

    dc = load_pickle(DATA_MODEL / "dixon_coles.pkl")
    if dc is None or not dc.is_fitted:
        print("  [AVISO] dixon_coles.pkl no encontrado o no entrenado.")
        return None

    print(f"  Intercept global: {dc.intercept_:.4f}  (avg goles esperado: {np.exp(dc.intercept_):.2f})")
    print(f"  Rho: {dc.rho_:.4f}\n")
    print(f"  {'Equipo':18s} {'Ataque':>10s} {'Defensa':>10s} {'Ataque rank':>14s}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*14}")

    all_attack = sorted(dc.attack_.items(), key=lambda x: -x[1])
    rank_map = {t: i+1 for i, (t, _) in enumerate(all_attack)}
    n_teams = len(dc.attack_)

    for t in teams:
        att = dc.attack_.get(t, np.nan)
        deff = dc.defense_.get(t, np.nan)
        rank = rank_map.get(t, "?")
        print(f"  {t:18s} {att:>10.4f} {deff:>10.4f} {f'{rank}/{n_teams}':>14s}")

    print("\n  → Ataque alto = mejor delantera. Defensa NEGATIVA = mejor defensa")
    print("    (el parámetro resta del lambda rival).")
    return dc


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lambdas en enfrentamientos directos clave
# ─────────────────────────────────────────────────────────────────────────────

def diag_h2h_lambdas(dc, focus_team="France"):
    section(f"3. GOLES ESPERADOS (λ) — {focus_team} vs cada rival (campo neutral)")

    if dc is None:
        print("  [AVISO] Sin modelo Dixon-Coles, se omite.")
        return

    rivals = [t for t in DEFAULT_TEAMS if t != focus_team]
    print(f"  {'Rival':18s} {f'λ_{focus_team[:10]}':>14s} {'λ_rival':>10s} {'P(gana ' + focus_team[:8] + ')':>16s} {'P(empate)':>10s} {'P(pierde)':>10s}")
    print(f"  {'─'*18} {'─'*14} {'─'*10} {'─'*16} {'─'*10} {'─'*10}")

    for rival in rivals:
        try:
            pred = dc.predict_match(focus_team, rival)
            print(f"  {rival:18s} {pred['lambda_home']:>14.3f} {pred['lambda_away']:>10.3f} "
                  f"{pred['p_home_win']:>15.1%} {pred['p_draw']:>10.1%} {pred['p_away_win']:>10.1%}")
        except Exception as e:
            print(f"  {rival:18s}  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparar capa 1 -> meta-modelo -> calibrado, para ver dónde se distorsiona
# ─────────────────────────────────────────────────────────────────────────────

def diag_pipeline_layers(teams):
    section("4. PIPELINE COMPLETO: Capa1 → Meta-modelo → Calibrado (vs Brasil, neutral)")

    dc   = load_pickle(DATA_MODEL / "dixon_coles.pkl")
    bay  = load_pickle(DATA_MODEL / "bayesian_map.pkl")
    meta = load_pickle(DATA_MODEL / "meta_xgboost.pkl")
    cal  = load_pickle(DATA_MODEL / "calibrator.pkl")

    if dc is None:
        print("  [AVISO] Sin Dixon-Coles, se omite esta sección.")
        return

    opponent = "Brazil"
    print(f"  Comparando cada equipo vs {opponent} (neutral), métrica = P(victoria del equipo)\n")
    print(f"  {'Equipo':18s} {'P_DC':>8s} {'P_Bay':>8s} {'P_promedio_L1':>15s} {'P_meta(si aplica)':>18s} {'P_calibrado':>14s}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*15} {'─'*18} {'─'*14}")

    for t in teams:
        if t == opponent:
            continue
        try:
            p_dc = dc.predict_match(t, opponent)["p_home_win"]
        except Exception:
            p_dc = np.nan
        try:
            p_bay = bay.predict_match(t, opponent)["p_home_win"] if bay else np.nan
        except Exception:
            p_bay = np.nan

        vals = [v for v in [p_dc, p_bay] if not np.isnan(v)]
        p_l1_avg = np.mean(vals) if vals else np.nan

        p_meta_str = "—"
        p_cal_str = "—"
        if meta is not None and meta.is_fitted:
            try:
                meta_input = {
                    "home_team": t, "away_team": opponent,
                    "p_home_dc": p_dc, "p_draw_dc": dc.predict_match(t, opponent)["p_draw"],
                    "p_away_dc": dc.predict_match(t, opponent)["p_away_win"],
                    "p_home_elo": 0.333, "p_draw_elo": 0.333, "p_away_elo": 0.334,
                    "p_home_rf": 0.333, "p_draw_rf": 0.333, "p_away_rf": 0.334,
                    "p_home_bay": p_bay if not np.isnan(p_bay) else 0.333,
                    "p_draw_bay": 0.333, "p_away_bay": 0.334,
                }
                df_meta = pd.DataFrame([meta_input])
                pred_meta = meta.predict_proba_df(df_meta).iloc[0]
                p_meta = pred_meta["p_home_win"]
                p_meta_str = f"{p_meta:.1%}"

                if cal is not None and cal.is_fitted:
                    arr = np.array([[pred_meta["p_home_win"], pred_meta["p_draw"], pred_meta["p_away_win"]]])
                    cal_out = cal.transform(arr)[0]
                    p_cal_str = f"{cal_out[0]:.1%}"
            except Exception as e:
                p_meta_str = f"err"

        print(f"  {t:18s} {p_dc:>7.1%} {p_bay:>7.1%} {p_l1_avg:>14.1%} {p_meta_str:>18s} {p_cal_str:>14s}")

    print("\n  → Si France cae mucho entre 'P_promedio_L1' y 'P_calibrado' mientras")
    print("    Japan/Uruguay no caen igual, el calibrador isotónico es sospechoso #1.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Peso real de delta_sv_log (valor de plantilla)
# ─────────────────────────────────────────────────────────────────────────────

def diag_feature_importance():
    section("5. PESO REAL DEL VALOR DE PLANTILLA (delta_sv_log) EN LOS MODELOS")

    rf = load_pickle(DATA_MODEL / "random_forest.pkl")
    meta = load_pickle(DATA_MODEL / "meta_xgboost.pkl")

    if rf is not None and hasattr(rf, "feature_importances_") and len(rf.feature_importances_) > 0:
        imp = rf.feature_importances_
        rank = imp.rank(ascending=False)
        if "delta_sv_log" in imp.index:
            print(f"  Random Forest:")
            print(f"    delta_sv_log → importancia {imp['delta_sv_log']:.4f}  "
                  f"(puesto {int(rank['delta_sv_log'])} de {len(imp)})")
        else:
            print("  Random Forest: delta_sv_log no está entre las features disponibles.")
    else:
        print("  Random Forest no encontrado o sin feature_importances_.")

    if meta is not None and meta.is_fitted:
        try:
            fi = meta.feature_importance()
            fi_dict = dict(zip(fi["feature"], fi[0] if 0 in fi.columns else fi.iloc[:, 1]))
            if "delta_sv_log" in fi_dict:
                rank_meta = fi.reset_index()
                pos = fi[fi["feature"] == "delta_sv_log"].index
                print(f"\n  Meta-modelo XGBoost:")
                print(f"    delta_sv_log → importancia {fi_dict['delta_sv_log']:.4f}")
            else:
                print("\n  Meta-modelo XGBoost: delta_sv_log NO aparece en la lista de importancia")
                print("    (probablemente ni se está usando como input al meta-modelo).")
        except Exception as e:
            print(f"  Meta-modelo: error al calcular importancia ({e})")

    print("\n  → Si delta_sv_log no aparece en ninguno de los dos, el valor de mercado")
    print("    de Francia (€1.520M, el más alto del torneo) literalmente no influye")
    print("    en la predicción. Todo el peso recae en el Elo dinámico recalculado.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Distorsión del calibrador específicamente en la cola alta (favoritos)
# ─────────────────────────────────────────────────────────────────────────────

def diag_calibration_tail():
    section("6. DISTORSIÓN DEL CALIBRADOR EN LA COLA ALTA (P > 0.45)")

    from config import VALIDATION_SET

    cal = load_pickle(DATA_MODEL / "calibrator.pkl")
    meta = load_pickle(DATA_MODEL / "meta_xgboost.pkl")

    if cal is None or not cal.is_fitted or meta is None or not meta.is_fitted:
        print("  [AVISO] Falta calibrator.pkl o meta_xgboost.pkl entrenados.")
        return

    if not VALIDATION_SET.exists():
        print("  [AVISO] No hay validation_set.parquet.")
        return

    val = pd.read_parquet(VALIDATION_SET)
    val = val.dropna(subset=["target_result"])
    val = val[val["target_result"].isin(["H", "D", "A"])]

    if len(val) < 30:
        print(f"  [AVISO] Validation set muy chico ({len(val)} partidos), resultado poco confiable.")

    preds_meta = meta.predict_proba_df(val)
    proba_raw = preds_meta[["p_home_win", "p_draw", "p_away_win"]].values
    proba_cal = cal.transform(proba_raw)

    # Cola alta: predicciones donde el meta-modelo daba > 0.45 de probabilidad
    # al resultado que efectivamente ocurrió que sea H o A (favoritos claros)
    mask_fav_home = proba_raw[:, 0] > 0.45
    mask_fav_away = proba_raw[:, 2] > 0.45

    if mask_fav_home.sum() > 0:
        avg_raw_h = proba_raw[mask_fav_home, 0].mean()
        avg_cal_h = proba_cal[mask_fav_home, 0].mean()
        print(f"  Favoritos LOCALES claros (P_raw_home > 45%): n={mask_fav_home.sum()}")
        print(f"    P promedio ANTES de calibrar : {avg_raw_h:.1%}")
        print(f"    P promedio DESPUÉS de calibrar: {avg_cal_h:.1%}")
        print(f"    Δ = {avg_cal_h - avg_raw_h:+.1%}  "
              f"{'← el calibrador BAJA a los favoritos' if avg_cal_h < avg_raw_h else ''}")

    if mask_fav_away.sum() > 0:
        avg_raw_a = proba_raw[mask_fav_away, 2].mean()
        avg_cal_a = proba_cal[mask_fav_away, 2].mean()
        print(f"\n  Favoritos VISITANTES claros (P_raw_away > 45%): n={mask_fav_away.sum()}")
        print(f"    P promedio ANTES de calibrar : {avg_raw_a:.1%}")
        print(f"    P promedio DESPUÉS de calibrar: {avg_cal_a:.1%}")
        print(f"    Δ = {avg_cal_a - avg_raw_a:+.1%}")

    print(f"\n  Tamaño del validation set usado para calibrar: {len(val)} partidos")
    print("  → Con <1000 muestras, la isotonic regression en la cola alta (favoritos")
    print("    extremos) es la parte MENOS confiable del calibrador: pocos puntos,")
    print("    mucho ruido. Esto puede estar empujando a Francia hacia abajo de forma")
    print("    no uniforme frente a selecciones con menos partidos como favorito claro.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnóstico de odds del Mundial 2026")
    parser.add_argument("--teams", nargs="+", default=DEFAULT_TEAMS,
                        help="Equipos a comparar")
    parser.add_argument("--focus", default="France",
                        help="Equipo foco para la sección de lambdas H2H")
    args = parser.parse_args()

    print("═" * 78)
    print("  DIAGNÓSTICO — ¿Por qué Francia aparece por detrás de Japón/Uruguay?")
    print("═" * 78)

    diag_elo(args.teams)
    dc = diag_dixon_coles(args.teams)
    diag_h2h_lambdas(dc, focus_team=args.focus)
    diag_pipeline_layers(args.teams)
    diag_feature_importance()
    diag_calibration_tail()

    print("\n" + "═" * 78)
    print("  RESUMEN DE HIPÓTESIS A REVISAR, EN ORDEN DE PROBABILIDAD:")
    print("═" * 78)
    print("""
  1. ELO DINÁMICO MAL CALIBRADO PARA POTENCIAS QUE ROTAN MUCHO
     Francia juega muchos amistosos con plantilla rotada (K=20) donde
     puede empatar/perder sin que eso refleje su nivel real. CONMEBOL
     (Uruguay, Colombia, Brasil, Argentina) juega casi todo en
     eliminatorias (K=40), más consistentes con su nivel.
     → Fix: subir el K de amistosos a un valor menor aún (K=10-15) o
       directamente excluir amistosos del cálculo de Elo para selecciones top.

  2. CALIBRADOR ISOTÓNICO SOBREAJUSTADO CON POCOS DATOS
     El log que pegaste muestra Log Loss empeorando tras calibrar
     (1.1405 → 1.1514) con sólo 3,228 partidos de validación.
     → Fix: usar method="temperature" en vez de "isotonic" (un solo
       parámetro, mucho más robusto con pocos datos), o aumentar el
       validation set fusionando con datos de test.

  3. VALOR DE PLANTILLA SIN PESO REAL EN EL MODELO
     delta_sv_log no aparece en el top-10 de ningún modelo del log.
     → Fix: si querés que el valor de mercado tenga más peso, hay que
       dárselo explícitamente (ej. feature adicional al meta-modelo,
       o un 4to modelo base que prediga directo desde sv_log + edad).

  4. SORTEO/BRACKET DESFAVORABLE PARA FRANCIA EN LA SIMULACIÓN
     Si el cruce de octavos/cuartos de Francia es sistemáticamente
     más duro que el de Brasil/Colombia, eso resta puntos de campeón
     incluso con el mismo nivel base.
     → Verificar en wc2026_simulation.json los rivales de R32/R16
       más frecuentes para Francia vs Brasil.
""")