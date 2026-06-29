"""
predict_match.py — Predictor Interactivo de Partidos (Terminal)
================================================================
Muestra los partidos del Mundial 2026, permite seleccionar uno
(o ingresar un partido personalizado) y muestra:
  - Ganador con porcentaje
  - Marcadores más probables con porcentaje
  - Over/Under, BTTS, goles esperados

Soporta registro de resultados reales y tabla de posiciones en vivo.

Uso:
  python predict_match.py                  → menú interactivo
  python predict_match.py --home Argentina --away France
  python predict_match.py --grupo A
  python predict_match.py --resultado "Argentina 3 Algeria 0"
"""

import sys
import json
import warnings
import pickle
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_MODEL
from simulation.knockout_stage import (
    WC2026_R32_REAL_BRACKET,
    WC2026_R32_REAL_BRACKET_LEFT,
    WC2026_R32_REAL_BRACKET_RIGHT,
)

# ── Constantes ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "data" / "resultados_reales.json"
MAX_GOALS = 9
_FACTORIALS = np.array([float(math.factorial(k)) for k in range(MAX_GOALS + 1)])

# ── Grupos del Mundial 2026 ───────────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Mapa inverso: equipo → grupo
TEAM_GROUP = {}
for g, teams in GROUPS.items():
    for t in teams:
        TEAM_GROUP[t] = g

ALL_TEAMS = sorted(set(t for teams in GROUPS.values() for t in teams))


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE RESULTADOS REALES
# ═══════════════════════════════════════════════════════════════════════════════

def load_results() -> List[Dict]:
    """Carga los resultados reales desde disco."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results: List[Dict]):
    """Guarda los resultados reales a disco."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def add_result(home: str, away: str, home_goals: int, away_goals: int,
               date: str = None) -> Dict:
    """Registra un resultado real."""
    results = load_results()

    # Verificar que los equipos existen
    if home not in ALL_TEAMS:
        return {"error": f"Equipo '{home}' no encontrado"}
    if away not in ALL_TEAMS:
        return {"error": f"Equipo '{away}' no encontrado"}

    # Verificar que son del mismo grupo
    g_home = TEAM_GROUP.get(home)
    g_away = TEAM_GROUP.get(away)
    if g_home != g_away:
        print(color(f"  [AVISO] Nota: {home} (Grupo {g_home}) vs {away} (Grupo {g_away}) — no son del mismo grupo", "yellow"))

    # Verificar duplicados
    for r in results:
        if (r["home_team"] == home and r["away_team"] == away) or \
           (r["home_team"] == away and r["away_team"] == home):
            # Actualizar resultado existente
            r["home_team"] = home
            r["away_team"] = away
            r["home_goals"] = home_goals
            r["away_goals"] = away_goals
            r["date"] = date or datetime.now().strftime("%Y-%m-%d")
            r["updated_at"] = datetime.now().isoformat()
            save_results(results)
            return {"status": "updated", "match": r}

    # Nuevo resultado
    match = {
        "home_team":  home,
        "away_team":  away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "group":      g_home if g_home == g_away else "?",
        "date":       date or datetime.now().strftime("%Y-%m-%d"),
        "created_at": datetime.now().isoformat(),
    }
    results.append(match)
    save_results(results)
    return {"status": "created", "match": match}


def get_group_results(group: str) -> List[Dict]:
    """Obtiene los resultados reales de un grupo."""
    results = load_results()
    teams = set(GROUPS.get(group, []))
    return [r for r in results
            if r["home_team"] in teams and r["away_team"] in teams]


def is_match_played(home: str, away: str) -> Optional[Dict]:
    """Verifica si un partido ya se jugó y devuelve el resultado."""
    results = load_results()
    for r in results:
        if (r["home_team"] == home and r["away_team"] == away) or \
           (r["home_team"] == away and r["away_team"] == home):
            return r
    return None


def compute_standings(group: str, dc_model=None, sim_data: Dict = None) -> List[Dict]:
    """
    Calcula la tabla de posiciones de un grupo.
    Usa resultados reales para partidos jugados y predicciones para los restantes.
    """
    teams = GROUPS.get(group, [])
    if not teams:
        return []

    # Inicializar tabla
    table = {}
    for t in teams:
        table[t] = {
            "team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0, "gd": 0, "pts": 0,
            "results": [],  # "R" = real, "P" = predicted
        }

    # Generar todos los partidos
    all_matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            all_matches.append((teams[i], teams[j]))

    results = load_results()

    for t1, t2 in all_matches:
        # Buscar resultado real
        real = None
        for r in results:
            if (r["home_team"] == t1 and r["away_team"] == t2) or \
               (r["home_team"] == t2 and r["away_team"] == t1):
                real = r
                break

        if real:
            # Usar resultado real
            home = real["home_team"]
            away = real["away_team"]
            hg = real["home_goals"]
            ag = real["away_goals"]
            source = "R"
        else:
            # Usar predicción (marcador más probable del DC)
            home, away = t1, t2
            hg, ag = _predict_score(home, away, dc_model, sim_data)
            source = "P"

        # Actualizar tabla
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg

        if hg > ag:
            table[home]["won"] += 1
            table[home]["pts"] += 3
            table[away]["lost"] += 1
        elif hg < ag:
            table[away]["won"] += 1
            table[away]["pts"] += 3
            table[home]["lost"] += 1
        else:
            table[home]["drawn"] += 1
            table[away]["drawn"] += 1
            table[home]["pts"] += 1
            table[away]["pts"] += 1

        table[home]["results"].append(source)
        table[away]["results"].append(source)

    # Calcular GD
    for t in table:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]

    # Ordenar: pts → gd → gf
    standings = sorted(table.values(),
                       key=lambda x: (x["pts"], x["gd"], x["gf"]),
                       reverse=True)
    return standings


def _predict_score(home: str, away: str, dc_model=None, sim_data: Dict = None) -> Tuple[int, int]:
    """Predice el marcador más probable para un partido no jugado."""
    # Intentar Dixon-Coles
    if dc_model and dc_model.is_fitted:
        try:
            pred = dc_model.predict_match(home, away)
            lh = pred["lambda_home"]
            la = pred["lambda_away"]
            matrix = build_score_matrix(lh, la, dc_model.rho_)
            idx = np.unravel_index(matrix.argmax(), matrix.shape)
            return int(idx[0]), int(idx[1])
        except Exception:
            pass

    # Fallback: datos de simulación
    if sim_data:
        gm = sim_data.get("group_matches", {})
        key = f"{home}|{away}"
        m = gm.get(key)
        if m and "most_likely_score" in m:
            parts = m["most_likely_score"].split("-")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])

    return 1, 0  # fallback genérico


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_dc_model():
    """Carga el modelo Dixon-Coles entrenado."""
    path = DATA_MODEL / "dixon_coles.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_simulation_data() -> Dict:
    """Carga los resultados de simulación Monte Carlo."""
    path = DATA_MODEL / "wc2026_simulation.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def poisson_pmf_vec(lam: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1, dtype=np.float64)
    log_pmf = k * np.log(max(lam, 1e-10)) - lam - np.log(_FACTORIALS)
    return np.exp(log_pmf)


def build_score_matrix(lh: float, la: float, rho: float = -0.13) -> np.ndarray:
    ph = poisson_pmf_vec(lh)
    pa = poisson_pmf_vec(la)
    table = np.outer(ph, pa)
    corr = np.array([
        [1.0 - lh * la * rho, 1.0 + lh * rho],
        [1.0 + la * rho,      1.0 - rho      ],
    ])
    table[:2, :2] *= np.maximum(corr, 1e-9)
    table = np.maximum(table, 0.0)
    s = table.sum()
    if s > 0:
        table /= s
    return table


def get_top_scores(matrix: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
    n = matrix.shape[0]
    scores = []
    for i in range(n):
        for j in range(n):
            scores.append((f"{i}-{j}", float(matrix[i, j])))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def get_outcome_probs(matrix: np.ndarray) -> Tuple[float, float, float]:
    n = matrix.shape[0]
    idx = np.arange(n)
    p_h = float(matrix[idx[:, None] > idx[None, :]].sum())
    p_d = float(np.diag(matrix).sum())
    p_a = float(matrix[idx[:, None] < idx[None, :]].sum())
    total = p_h + p_d + p_a
    if total < 1e-9:
        return 1/3, 1/3, 1/3
    return p_h / total, p_d / total, p_a / total


def get_market_probs(matrix: np.ndarray) -> Dict:
    n = matrix.shape[0]
    total_goals = np.array([[i + j for j in range(n)] for i in range(n)])
    return {
        "over_15":  float(matrix[total_goals > 1.5].sum()),
        "over_25":  float(matrix[total_goals > 2.5].sum()),
        "over_35":  float(matrix[total_goals > 3.5].sum()),
        "under_25": float(matrix[total_goals <= 2.5].sum()),
        "btts":     float(matrix[1:, 1:].sum()),
    }


def predict_full(home: str, away: str, dc_model, sim_data: Dict) -> Dict:
    """Predicción completa de un partido."""
    result = {"home": home, "away": away, "source": "dixon_coles"}

    # Verificar si ya se jugó
    played = is_match_played(home, away)
    if played:
        result["played"] = played

    # Dixon-Coles
    if dc_model and dc_model.is_fitted:
        try:
            pred = dc_model.predict_match(home, away)
            lh = pred["lambda_home"]
            la = pred["lambda_away"]
            matrix = build_score_matrix(lh, la, dc_model.rho_)
            p_h, p_d, p_a = get_outcome_probs(matrix)
            result.update({
                "lambda_home": lh, "lambda_away": la,
                "p_home_win": p_h, "p_draw": p_d, "p_away_win": p_a,
                "top_scores": get_top_scores(matrix, 10),
                "markets": get_market_probs(matrix),
                "score_matrix": matrix,
            })
        except Exception:
            result["source"] = "simulation_only"

    # Simulación Monte Carlo
    key = f"{home}|{away}"
    key_rev = f"{away}|{home}"
    sim_match = sim_data.get("group_matches", {}).get(key)
    sim_match_rev = sim_data.get("group_matches", {}).get(key_rev)

    if sim_match:
        result["sim"] = sim_match
    elif sim_match_rev:
        result["sim"] = {
            "home_team": sim_match_rev["away_team"],
            "away_team": sim_match_rev["home_team"],
            "p_home_win": sim_match_rev["p_away_win"],
            "p_draw": sim_match_rev["p_draw"],
            "p_away_win": sim_match_rev["p_home_win"],
            "avg_home_goals": sim_match_rev["avg_away_goals"],
            "avg_away_goals": sim_match_rev["avg_home_goals"],
            "most_likely_score": "-".join(reversed(
                sim_match_rev["most_likely_score"].split("-"))),
        }

    # Fallback a simulación
    if "p_home_win" not in result and sim_match:
        result.update({
            "p_home_win": sim_match["p_home_win"],
            "p_draw": sim_match["p_draw"],
            "p_away_win": sim_match["p_away_win"],
            "lambda_home": sim_match.get("avg_home_goals", 1.3),
            "lambda_away": sim_match.get("avg_away_goals", 1.0),
            "source": "monte_carlo_10K",
        })
        lh = result["lambda_home"]
        la = result["lambda_away"]
        matrix = build_score_matrix(lh, la)
        result["top_scores"] = get_top_scores(matrix, 10)
        result["markets"] = get_market_probs(matrix)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN EN TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def bar(pct: float, width: int = 30, char: str = "█") -> str:
    filled = int(pct * width)
    return char * filled + "░" * (width - filled)


def color(text: str, code: str) -> str:
    codes = {
        "green":    "\033[92m",
        "red":      "\033[91m",
        "yellow":   "\033[93m",
        "blue":     "\033[94m",
        "cyan":     "\033[96m",
        "magenta":  "\033[95m",
        "white":    "\033[97m",
        "bold":     "\033[1m",
        "dim":      "\033[2m",
        "reset":    "\033[0m",
    }
    return codes.get(code, "") + text + codes["reset"]


def print_header():
    print()
    print(color("╔══════════════════════════════════════════════════════════════╗", "cyan"))
    print(color("║", "cyan") + color("       PREDICTOR MUNDIAL FIFA 2026                       ", "bold") + color("║", "cyan"))
    print(color("║", "cyan") + color("     Sistema de predicción con ensemble 3 capas             ", "dim") + color("║", "cyan"))
    print(color("║", "cyan") + color("     Dixon-Coles + Elo + RF + Bayesian + XGBoost            ", "dim") + color("║", "cyan"))
    print(color("╚══════════════════════════════════════════════════════════════╝", "cyan"))
    print()


def print_groups():
    print(color("  ┌─────────────────────────────────────────────────────────┐", "cyan"))
    print(color("  │            GRUPOS DEL MUNDIAL FIFA 2026                │", "cyan"))
    print(color("  └─────────────────────────────────────────────────────────┘", "cyan"))
    print()

    group_letters = list(GROUPS.keys())
    for row in range(4):
        groups_row = group_letters[row * 3: row * 3 + 3]
        lines = []
        for g in groups_row:
            teams = GROUPS[g]
            header = color(f"  Grupo {g}", "bold")
            lines.append((header, teams))

        print("  ", end="")
        for header, _ in lines:
            print(f"{header:42s}", end="")
        print()
        print("  ", end="")
        for _ in lines:
            print(f"  {'─' * 20:22s}", end="")
        print()

        for i in range(4):
            print("  ", end="")
            for _, teams in lines:
                t = teams[i] if i < len(teams) else ""
                print(f"  {t:20s}", end="")
            print()
        print()


def print_group_matches(group: str, sim_data: Dict):
    """Muestra los partidos de un grupo con resultados reales + predicciones."""
    teams = GROUPS.get(group, [])
    if not teams:
        print(color(f"  Grupo '{group}' no encontrado.", "red"))
        return

    print()
    print(color(f"  ┌───────────────────────────────────────────────────────────┐", "cyan"))
    print(color(f"  │              PARTIDOS DEL GRUPO {group}                          │", "cyan"))
    print(color(f"  └───────────────────────────────────────────────────────────┘", "cyan"))
    print()

    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            matches.append((teams[i], teams[j]))

    gm = sim_data.get("group_matches", {})
    results = load_results()

    for idx, (t1, t2) in enumerate(matches, 1):
        # Verificar si ya se jugó
        real = None
        for r in results:
            if (r["home_team"] == t1 and r["away_team"] == t2) or \
               (r["home_team"] == t2 and r["away_team"] == t1):
                real = r
                break

        if real:
            # PARTIDO JUGADO — mostrar resultado real
            home = real["home_team"]
            away = real["away_team"]
            hg = real["home_goals"]
            ag = real["away_goals"]

            if hg > ag:
                winner_str = color(f"[OK] {home}", "green")
            elif ag > hg:
                winner_str = color(f"[OK] {away}", "green")
            else:
                winner_str = color("[OK] Empate", "yellow")

            print(f"  {color(str(idx), 'bold'):>6s}. {home:20s} vs {away:20s}")
            print(f"         {color(f'RESULTADO FINAL: {hg}-{ag}', 'bold')}"
                  f"  │  {winner_str}"
                  f"  {color('(jugado [OK])', 'green')}")
        else:
            # PARTIDO PENDIENTE — mostrar predicción
            key = f"{t1}|{t2}"
            key_rev = f"{t2}|{t1}"
            m = gm.get(key) or gm.get(key_rev)

            if m:
                home = m["home_team"]
                away = m["away_team"]
                ph = m["p_home_win"]
                pd_ = m["p_draw"]
                pa = m["p_away_win"]
                score = m.get("most_likely_score", "?-?")

                if ph > max(pd_, pa):
                    fav = home
                    fav_p = ph
                    fav_c = "green"
                elif pa > max(ph, pd_):
                    fav = away
                    fav_p = pa
                    fav_c = "green"
                else:
                    fav = "Empate"
                    fav_p = pd_
                    fav_c = "yellow"

                print(f"  {color(str(idx), 'bold'):>6s}. {home:20s} vs {away:20s}")
                print(f"         {color(f'{ph:.0%}', 'green')} / "
                      f"{color(f'{pd_:.0%}', 'yellow')} / "
                      f"{color(f'{pa:.0%}', 'green')}"
                      f"  │  Marcador: {color(score, 'bold')}"
                      f"  │  Favorito: {color(f'{fav} ({fav_p:.0%})', fav_c)}"
                      f"  {color('(pendiente)', 'dim')}")
            else:
                print(f"  {color(str(idx), 'bold'):>6s}. {t1:20s} vs {t2:20s}")
                print(f"         {color('(pendiente — sin datos)', 'dim')}")
        print()


def print_standings(group: str, dc_model=None, sim_data: Dict = None):
    """Imprime la tabla de posiciones de un grupo."""
    standings = compute_standings(group, dc_model, sim_data)
    if not standings:
        return

    n_real = sum(1 for s in standings if "R" in s["results"])

    print()
    print(color(f"  ┌───────────────────────────────────────────────────────────────────┐", "cyan"))
    print(color(f"  │              TABLA DE POSICIONES — GRUPO {group}                        │", "cyan"))
    print(color(f"  └───────────────────────────────────────────────────────────────────┘", "cyan"))
    print()

    if n_real > 0:
        print(color(f"  ℹ  Incluye resultados reales + predicciones para partidos pendientes", "dim"))
    else:
        print(color(f"  ℹ  Basada 100% en predicciones (sin resultados reales aún)", "dim"))
    print()

    header = f"  {'#':>2s}  {'Equipo':22s} {'PJ':>3s} {'PG':>3s} {'PE':>3s} {'PP':>3s} {'GF':>3s} {'GA':>3s} {'GD':>4s} {'Pts':>4s}  {'Estado'}"
    print(color(header, "bold"))
    print(f"  {'─'*2}  {'─'*22} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*3} {'─'*4} {'─'*4}  {'─'*10}")

    for i, s in enumerate(standings):
        team = s["team"]
        # Indicador de partidos reales vs predichos
        real_count = s["results"].count("R")
        pred_count = s["results"].count("P")
        if real_count > 0 and pred_count > 0:
            status = f"{real_count}R + {pred_count}P"
        elif real_count > 0:
            status = f"{real_count}R jugados"
        else:
            status = f"{pred_count}P predichos"

        # Color según posición
        if i < 2:
            tc = "green"   # Clasifican directo
            pos_mark = "^"
        elif i == 2:
            tc = "yellow"  # Posible mejor tercero
            pos_mark = "o"
        else:
            tc = "red"     # Eliminado
            pos_mark = "v"

        gd_str = f"+{s['gd']}" if s["gd"] > 0 else str(s["gd"])

        line = (f"  {color(pos_mark, tc)} {i+1}  "
                f"{color(team, tc):>32s} "
                f"{s['played']:>3d} {s['won']:>3d} {s['drawn']:>3d} {s['lost']:>3d} "
                f"{s['gf']:>3d} {s['ga']:>3d} {gd_str:>4s} "
                f"{color(str(s['pts']), 'bold'):>13s}  "
                f"{color(status, 'dim')}")
        print(line)

    print()
    print(color("  ^ = Clasifica  o = 3ro (posible)  v = Eliminado", "dim"))
    print(color("  R = Resultado real  P = Predicción del modelo", "dim"))
    print()


def print_prediction(pred: Dict):
    """Imprime la predicción completa de un partido."""
    home = pred["home"]
    away = pred["away"]
    ph = pred.get("p_home_win", 0.333)
    pd_ = pred.get("p_draw", 0.334)
    pa = pred.get("p_away_win", 0.333)
    lh = pred.get("lambda_home")
    la = pred.get("lambda_away")
    played = pred.get("played")

    if ph > max(pd_, pa):
        winner, winner_prob, winner_c = home, ph, "green"
    elif pa > max(ph, pd_):
        winner, winner_prob, winner_c = away, pa, "green"
    else:
        winner, winner_prob, winner_c = "EMPATE", pd_, "yellow"

    w = 62

    print()
    print(color("╔" + "═" * w + "╗", "cyan"))
    title = f"  {home}  vs  {away}  "
    print(color("║", "cyan") + color(f"{title:^{w}s}", "bold") + color("║", "cyan"))
    print(color("╠" + "═" * w + "╣", "cyan"))

    # ── RESULTADO REAL si ya se jugó ──────────────────────────────────
    if played:
        hg = played["home_goals"]
        ag = played["away_goals"]
        h_name = played["home_team"]
        a_name = played["away_team"]

        if hg > ag:
            result_str = f"[OK] Ganó {h_name}"
            rc = "green"
        elif ag > hg:
            result_str = f"[OK] Ganó {a_name}"
            rc = "green"
        else:
            result_str = "[OK] Empate"
            rc = "yellow"

        score_line = f"RESULTADO REAL:  {h_name} {hg} - {ag} {a_name}"
        print(color("║", "cyan") + color(f"{score_line:^{w}s}", "bold") + color("║", "cyan"))
        print(color("║", "cyan") + color(f"{result_str:^{w}s}", rc) + color("║", "cyan"))
        print(color("║", "cyan") + color(f"{'(Fecha: ' + played.get('date', '?') + ')':^{w}s}", "dim") + color("║", "cyan"))
        print(color("╠" + "═" * w + "╣", "cyan"))
        pred_label = "PREDICCIÓN DEL MODELO (pre-partido)"
        print(color("║", "cyan") + color(f"{pred_label:^{w}s}", "dim") + color("║", "cyan"))
        print(color("╠" + "═" * w + "╣", "cyan"))

    # ── Probabilidades 1X2 ────────────────────────────────────────────
    print(color("║", "cyan") + color(f"{'PROBABILIDADES DE RESULTADO':^{w}s}", "bold") + color("║", "cyan"))
    print(color("║", "cyan") + " " * w + color("║", "cyan"))

    bar_h = bar(ph, 35)
    line = f"  {home[:18]:18s}  {bar_h} {ph:6.1%}"
    print(color("║", "cyan") + color(f"{line:<{w}s}", "green") + color("║", "cyan"))

    bar_d = bar(pd_, 35)
    line = f"  {'Empate':18s}  {bar_d} {pd_:6.1%}"
    print(color("║", "cyan") + color(f"{line:<{w}s}", "yellow") + color("║", "cyan"))

    bar_a = bar(pa, 35)
    line = f"  {away[:18]:18s}  {bar_a} {pa:6.1%}"
    print(color("║", "cyan") + color(f"{line:<{w}s}", "green") + color("║", "cyan"))

    print(color("║", "cyan") + " " * w + color("║", "cyan"))

    # ── Goles esperados ───────────────────────────────────────────────
    if lh is not None and la is not None:
        print(color("╠" + "═" * w + "╣", "cyan"))
        print(color("║", "cyan") + color(f"{'GOLES ESPERADOS (λ)':^{w}s}", "bold") + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))
        bar_lh = bar(lh / 4.0, 25, "▓")
        bar_la = bar(la / 4.0, 25, "▓")
        line = f"  {home[:18]:18s}  {bar_lh} λ = {lh:.2f}"
        print(color("║", "cyan") + color(f"{line:<{w}s}", "blue") + color("║", "cyan"))
        line = f"  {away[:18]:18s}  {bar_la} λ = {la:.2f}"
        print(color("║", "cyan") + color(f"{line:<{w}s}", "magenta") + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))

    # ── Marcadores más probables ──────────────────────────────────────
    top_scores = pred.get("top_scores", [])
    if top_scores:
        print(color("╠" + "═" * w + "╣", "cyan"))
        print(color("║", "cyan") + color(f"{'MARCADORES MÁS PROBABLES':^{w}s}", "bold") + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))

        for i, (score, prob) in enumerate(top_scores):
            h_goals, a_goals = score.split("-")
            rank = f"#{i+1}"

            if int(h_goals) > int(a_goals):
                sc, indicator = "green", f"→ {home[:12]}"
            elif int(h_goals) < int(a_goals):
                sc, indicator = "red", f"→ {away[:12]}"
            else:
                sc, indicator = "yellow", "→ Empate"

            # Marcar si coincide con resultado real
            hit = ""
            if played:
                real_score = f"{played['home_goals']}-{played['away_goals']}"
                if home == played["home_team"] and score == real_score:
                    hit = " ← [OK] ACERTÓ"
                elif home != played["home_team"]:
                    rev_score = f"{played['away_goals']}-{played['home_goals']}"
                    if score == rev_score:
                        hit = " ← [OK] ACERTÓ"

            score_bar = bar(prob / max(top_scores[0][1], 0.001), 18)
            line = f"  {rank:4s} {color(f'{score:5s}', sc):>20s}  {score_bar} {prob:6.2%}  {color(indicator, sc)}"
            raw_line = f"  {rank:4s} {score:5s}  {bar(prob / max(top_scores[0][1], 0.001), 18)} {prob:6.2%}  {indicator}{hit}"
            pad = w - len(raw_line)
            suffix = color(hit, "green") if hit else ""
            print(color("║", "cyan") + line + suffix + " " * max(0, pad) + color("║", "cyan"))

        print(color("║", "cyan") + " " * w + color("║", "cyan"))

    # ── Mercados adicionales ──────────────────────────────────────────
    markets = pred.get("markets")
    if markets:
        print(color("╠" + "═" * w + "╣", "cyan"))
        print(color("║", "cyan") + color(f"{'MERCADOS ADICIONALES':^{w}s}", "bold") + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))
        m_lines = [
            f"  Over 1.5 goles:  {markets['over_15']:6.1%}   │  Under 2.5 goles: {markets['under_25']:6.1%}",
            f"  Over 2.5 goles:  {markets['over_25']:6.1%}   │  Over 3.5 goles:  {markets['over_35']:6.1%}",
            f"  Ambos marcan:    {markets['btts']:6.1%}   │",
        ]
        for ml in m_lines:
            print(color("║", "cyan") + f"{ml:<{w}s}" + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))

    # ── Monte Carlo ───────────────────────────────────────────────────
    sim = pred.get("sim")
    if sim:
        print(color("╠" + "═" * w + "╣", "cyan"))
        print(color("║", "cyan") + color(f"{'MONTE CARLO (10,000 simulaciones)':^{w}s}", "bold") + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))
        sh = sim.get("p_home_win", 0)
        sd = sim.get("p_draw", 0)
        sa = sim.get("p_away_win", 0)
        sg = sim.get("most_likely_score", "?")
        sg_p = sim.get("most_likely_score_prob", 0)
        avg_h = sim.get("avg_home_goals", 0)
        avg_a = sim.get("avg_away_goals", 0)
        line1 = f"  {sim.get('home_team','')}: {sh:.1%}  |  Empate: {sd:.1%}  |  {sim.get('away_team','')}: {sa:.1%}"
        line2 = f"  Promedio goles: {avg_h:.2f} - {avg_a:.2f}  |  Score MC: {sg} ({sg_p:.1%})"
        print(color("║", "cyan") + f"{line1:<{w}s}" + color("║", "cyan"))
        print(color("║", "cyan") + f"{line2:<{w}s}" + color("║", "cyan"))
        print(color("║", "cyan") + " " * w + color("║", "cyan"))

    # ── Favorito ──────────────────────────────────────────────────────
    print(color("╠" + "═" * w + "╣", "cyan"))
    winner_line = f"[CAMPEON] FAVORITO: {winner}  ({winner_prob:.1%})"
    print(color("║", "cyan") + color(f"{winner_line:^{w}s}", winner_c) + color("║", "cyan"))
    print(color("╚" + "═" * w + "╝", "cyan"))
    print(color(f"  Fuente: {pred.get('source', 'N/A')}", "dim"))
    print()


def print_score_heatmap(pred: Dict):
    matrix = pred.get("score_matrix")
    if matrix is None:
        return

    home = pred["home"]
    away = pred["away"]

    print(color("  ┌─────────────────────────────────────────────────────┐", "cyan"))
    print(color("  │            HEATMAP DE MARCADORES                   │", "cyan"))
    print(color("  └─────────────────────────────────────────────────────┘", "cyan"))
    print()

    n = min(7, matrix.shape[0])
    header = f"  {away[:10]:>10s} →   "
    for j in range(n):
        header += f"  {j:2d}  "
    print(color(header, "dim"))
    print(f"  {home[:10]:>10s} ↓   " + "─" * (n * 5 + 1))

    max_val = matrix[:n, :n].max()
    for i in range(n):
        line = f"  {'':>10s} {i:2d} │ "
        for j in range(n):
            val = matrix[i, j]
            pct = f"{val:.1%}" if val >= 0.01 else f"{val:.2%}"
            if val >= max_val * 0.8:
                line += color(f"{pct:>5s}", "green")
            elif val >= max_val * 0.5:
                line += color(f"{pct:>5s}", "yellow")
            elif val >= max_val * 0.2:
                line += color(f"{pct:>5s}", "dim")
            else:
                line += f"{pct:>5s}"
        print(line)
    print()


def print_all_results():
    """Muestra todos los resultados reales registrados."""
    results = load_results()
    if not results:
        print(color("  No hay resultados registrados aún.", "yellow"))
        return

    print()
    print(color("  ┌───────────────────────────────────────────────────────────┐", "cyan"))
    print(color("  │              RESULTADOS REALES REGISTRADOS               │", "cyan"))
    print(color("  └───────────────────────────────────────────────────────────┘", "cyan"))
    print()

    # Agrupar por grupo
    by_group = {}
    for r in results:
        g = r.get("group", "?")
        by_group.setdefault(g, []).append(r)

    for g in sorted(by_group.keys()):
        print(color(f"  Grupo {g}:", "bold"))
        for r in by_group[g]:
            hg = r["home_goals"]
            ag = r["away_goals"]
            home = r["home_team"]
            away = r["away_team"]
            date = r.get("date", "?")

            if hg > ag:
                rc = "green"
            elif ag > hg:
                rc = "red"
            else:
                rc = "yellow"

            print(f"    {date}  {home:20s} {color(f'{hg}-{ag}', rc):>14s} {away:20s}")
        print()


def find_team(query: str) -> Optional[str]:
    query_lower = query.strip().lower()
    for t in ALL_TEAMS:
        if t.lower() == query_lower:
            return t
    matches = [t for t in ALL_TEAMS if query_lower in t.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(color(f"  Múltiples coincidencias: {', '.join(matches)}", "yellow"))
        print(color(f"  Sé más específico.", "yellow"))
        return None
    else:
        print(color(f"  Equipo '{query}' no encontrado.", "red"))
        print(color(f"  Equipos disponibles:", "dim"))
        for i, t in enumerate(ALL_TEAMS):
            print(f"    {t}", end="")
            if (i + 1) % 4 == 0:
                print()
        print()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MENÚ INTERACTIVO
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_menu():
    print_header()
    print(color("  Cargando modelos...", "dim"))

    dc_model = load_dc_model()
    sim_data = load_simulation_data()

    models_loaded = []
    if dc_model and dc_model.is_fitted:
        models_loaded.append("Dixon-Coles")
    if sim_data.get("group_matches"):
        models_loaded.append(f"Monte Carlo ({sim_data.get('meta',{}).get('n_sims', '?')} sims)")

    if models_loaded:
        print(color(f"  [OK] Modelos cargados: {', '.join(models_loaded)}", "green"))
    else:
        print(color("  [ERROR] No se encontraron modelos entrenados.", "red"))
        print(color("    Ejecutar primero: python run_final.py", "yellow"))
        return

    # Mostrar resultados reales cargados
    results = load_results()
    if results:
        print(color(f"  [OK] Resultados reales: {len(results)} partidos registrados", "green"))

    champ = sim_data.get("champion_probs", [])

    while True:
        print()
        print(color("  ┌─────────────────────────────────────────────────────┐", "cyan"))
        print(color("  │                  MENÚ PRINCIPAL                     │", "cyan"))
        print(color("  └─────────────────────────────────────────────────────┘", "cyan"))
        print()
        print("  1. Ver todos los grupos")
        print("  2. Ver partidos de un grupo (resultados + predicciones)")
        print("  3. Predecir un partido del Mundial")
        print("  4. Predecir un partido personalizado")
        print("  5. Ver favoritos al título")
        print("  6. Predecir TODOS los partidos de un grupo")
        print(color("  ─────────────────────────────────────────────────", "dim"))
        print(color("  7. Registrar resultado real", "yellow"))
        print(color("  8. Ver resultados registrados", "yellow"))
        print(color("  9. Ver tabla de posiciones de un grupo", "yellow"))
        print(color("  ─────────────────────────────────────────────────", "dim"))
        print(color(" 10. Sincronizar resultados desde API", "green"))
        print()
        print(color("  0. Salir", "dim"))
        print()

        try:
            choice = input(color("  → Elige una opción: ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "0":
            print(color("\n  ¡Hasta luego!\n", "cyan"))
            break

        elif choice == "1":
            print_groups()

        elif choice == "2":
            g = input(color("  → Grupo (A-L): ", "bold")).strip().upper()
            if g in GROUPS:
                print_group_matches(g, sim_data)
            else:
                print(color(f"  Grupo '{g}' no válido. Usa A-L.", "red"))

        elif choice == "3":
            print()
            for g, teams in GROUPS.items():
                print(f"  Grupo {color(g, 'bold')}: {', '.join(teams)}")
            print()

            g = input(color("  → Grupo (A-L): ", "bold")).strip().upper()
            if g not in GROUPS:
                print(color(f"  Grupo '{g}' no válido.", "red"))
                continue

            teams = GROUPS[g]
            matches = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    matches.append((teams[i], teams[j]))

            print()
            for idx, (t1, t2) in enumerate(matches, 1):
                played = is_match_played(t1, t2)
                if played:
                    status = color(f" [OK] ({played['home_goals']}-{played['away_goals']})", "green")
                else:
                    status = color(" (pendiente)", "dim")
                print(f"  {idx}. {t1} vs {t2}{status}")
            print()

            try:
                sel = int(input(color("  → Número de partido: ", "bold")).strip())
                if 1 <= sel <= len(matches):
                    home, away = matches[sel - 1]
                    pred = predict_full(home, away, dc_model, sim_data)
                    print_prediction(pred)
                    print_score_heatmap(pred)
                else:
                    print(color("  Número no válido.", "red"))
            except ValueError:
                print(color("  Entrada no válida.", "red"))

        elif choice == "4":
            print()
            print(color("  Equipos disponibles:", "dim"))
            for i, t in enumerate(ALL_TEAMS):
                print(f"    {t:25s}", end="")
                if (i + 1) % 3 == 0:
                    print()
            print("\n")

            h_input = input(color("  → Equipo local: ", "bold")).strip()
            home = find_team(h_input)
            if not home:
                continue

            a_input = input(color("  → Equipo visitante: ", "bold")).strip()
            away = find_team(a_input)
            if not away:
                continue

            if home == away:
                print(color("  ¡Un equipo no puede jugar contra sí mismo!", "red"))
                continue

            print(color(f"\n  Prediciendo: {home} vs {away}...\n", "dim"))
            pred = predict_full(home, away, dc_model, sim_data)
            print_prediction(pred)
            print_score_heatmap(pred)

        elif choice == "5":
            if not champ:
                print(color("  Sin datos de simulación.", "red"))
                continue

            print()
            print(color("  ┌─────────────────────────────────────────────────────────┐", "cyan"))
            print(color("  │            [CAMPEON]  FAVORITOS AL TÍTULO  [CAMPEON]                  │", "cyan"))
            print(color("  └─────────────────────────────────────────────────────────┘", "cyan"))
            print()
            print(f"  {'#':>3s}  {'Equipo':20s} {'Grupo':6s} {'Campeón':>8s} {'Final':>8s} {'SF':>8s} {'QF':>8s} {'R16':>8s} {'Avanza':>8s}")
            print(f"  {'─'*3}  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

            for i, c in enumerate(champ[:20]):
                team = c["team"]
                grp = c.get("group", "?")
                p_ch = c.get("p_champion", 0)
                p_fi = c.get("p_final", 0)
                p_sf = c.get("p_sf", 0)
                p_qf = c.get("p_qf", 0)
                p_r16 = c.get("p_r16", 0)
                p_ga = c.get("p_group_advance", 0)

                tc = "green" if i < 3 else ("yellow" if i < 8 else "white")
                medal = "1." if i == 0 else ("2." if i == 1 else ("3." if i == 2 else "  "))
                line = f"  {medal}{i+1:>2d}  {color(team, tc):>30s} {grp:^6s} {p_ch:>7.1%} {p_fi:>7.1%} {p_sf:>7.1%} {p_qf:>7.1%} {p_r16:>7.1%} {p_ga:>7.1%}"
                print(line)

            print()
            n_sims = sim_data.get("meta", {}).get("n_sims", "?")
            elapsed = sim_data.get("meta", {}).get("elapsed_seconds", 0)
            print(color(f"  Basado en {n_sims:,} simulaciones Monte Carlo ({elapsed:.0f}s)", "dim"))
            print()

        elif choice == "6":
            g = input(color("  → Grupo (A-L): ", "bold")).strip().upper()
            if g not in GROUPS:
                print(color(f"  Grupo '{g}' no válido.", "red"))
                continue

            teams = GROUPS[g]
            matches = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    matches.append((teams[i], teams[j]))

            print()
            print(color(f"  ═══ PREDICCIONES COMPLETAS — GRUPO {g} ═══", "bold"))
            for home, away in matches:
                pred = predict_full(home, away, dc_model, sim_data)
                print_prediction(pred)

        elif choice == "7":
            # ── REGISTRAR RESULTADO REAL ──────────────────────────────
            print()
            print(color("  ═══ REGISTRAR RESULTADO REAL ═══", "bold"))
            print()
            print(color("  Formato: escribe el equipo local, visitante y goles", "dim"))
            print(color("  También puedes escribir todo junto: Argentina 3 Algeria 0", "dim"))
            print()

            raw = input(color("  → Resultado (ej: Argentina 3 Algeria 0): ", "bold")).strip()

            # Intentar parsear formato "Team1 X Team2 Y"
            parsed = _parse_result_input(raw)
            if parsed:
                home, hg, away, ag = parsed
                home_team = find_team(home)
                if not home_team:
                    continue
                away_team = find_team(away)
                if not away_team:
                    continue

                # Confirmar
                print()
                print(f"  {home_team} {hg} - {ag} {away_team}")
                confirm = input(color("  ¿Correcto? (s/n): ", "bold")).strip().lower()
                if confirm in ("s", "si", "sí", "y", "yes", ""):
                    res = add_result(home_team, away_team, hg, ag)
                    if "error" in res:
                        print(color(f"  [ERROR] Error: {res['error']}", "red"))
                    else:
                        status = res["status"]
                        m = res["match"]
                        if status == "updated":
                            print(color(f"  [OK] Resultado ACTUALIZADO: {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}", "green"))
                        else:
                            print(color(f"  [OK] Resultado REGISTRADO: {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}", "green"))
                        print(color(f"    Grupo: {m.get('group', '?')}  |  Fecha: {m.get('date', '?')}", "dim"))
                else:
                    print(color("  Cancelado.", "dim"))
            else:
                # Formato paso a paso
                h_input = input(color("  → Equipo local: ", "bold")).strip()
                home = find_team(h_input)
                if not home:
                    continue

                a_input = input(color("  → Equipo visitante: ", "bold")).strip()
                away = find_team(a_input)
                if not away:
                    continue

                try:
                    hg = int(input(color(f"  → Goles de {home}: ", "bold")).strip())
                    ag = int(input(color(f"  → Goles de {away}: ", "bold")).strip())
                except ValueError:
                    print(color("  Entrada no válida.", "red"))
                    continue

                res = add_result(home, away, hg, ag)
                if "error" in res:
                    print(color(f"  [ERROR] Error: {res['error']}", "red"))
                else:
                    m = res["match"]
                    print(color(f"  [OK] Resultado registrado: {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}", "green"))

        elif choice == "8":
            print_all_results()

        elif choice == "9":
            g = input(color("  → Grupo (A-L): ", "bold")).strip().upper()
            if g in GROUPS:
                print_standings(g, dc_model, sim_data)
            else:
                print(color(f"  Grupo '{g}' no válido.", "red"))

        elif choice == "10":
            # ── SINCRONIZAR DESDE API ─────────────────────────────
            try:
                from ingestion.sync_results import sync, print_quick_standings
                new, updated, _ = sync()
                if new or updated:
                    # Recargar resultados después del sync
                    results = load_results()
                    print(color(f"  [OK] {len(results)} resultados totales en memoria", "green"))
                    ver = input(color("  → ¿Ver tabla de un grupo? (A-L / Enter para saltar): ", "bold")).strip().upper()
                    if ver and ver in GROUPS:
                        print_standings(ver, dc_model, sim_data)
            except ImportError as e:
                print(color(f"  [ERROR] Error importando sync_results: {e}", "red"))
            except Exception as e:
                print(color(f"  [ERROR] Error en sincronización: {e}", "red"))

        else:
            print(color("  Opción no válida.", "red"))


def _parse_result_input(raw: str) -> Optional[Tuple[str, int, str, int]]:
    """
    Intenta parsear formatos como:
      'Argentina 3 Algeria 0'
      'Argentina 3-0 Algeria'
      'Argentina 3 - 0 Algeria'
    """
    import re

    # Formato: Team1 X - Y Team2  o  Team1 X Team2 Y
    # Intentar: Team1 <num> - <num> Team2
    m = re.match(r'^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+)$', raw.strip())
    if m:
        return m.group(1).strip(), int(m.group(2)), m.group(4).strip(), int(m.group(3))

    # Formato: Team1 X Team2 Y
    # Necesitamos encontrar los números
    parts = raw.strip().split()
    if len(parts) >= 4:
        # Buscar patrón: palabras... número palabras... número
        nums = [(i, parts[i]) for i in range(len(parts)) if parts[i].isdigit()]
        if len(nums) >= 2:
            idx1, n1 = nums[0]
            idx2, n2 = nums[1]
            team1 = " ".join(parts[:idx1])
            team2 = " ".join(parts[idx1+1:idx2])
            if team1 and team2:
                return team1, int(n1), team2, int(n2)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=" Predictor de partidos del Mundial FIFA 2026"
    )
    parser.add_argument("--home", type=str, default=None, help="Equipo local")
    parser.add_argument("--away", type=str, default=None, help="Equipo visitante")
    parser.add_argument("--grupo", type=str, default=None, help="Mostrar partidos (A-L)")
    parser.add_argument("--tabla", type=str, default=None, help="Tabla de posiciones (A-L)")
    parser.add_argument("--favoritos", action="store_true", help="Favoritos al título")
    parser.add_argument("--resultado", type=str, default=None,
                        help='Registrar resultado: "Argentina 3 Algeria 0"')
    parser.add_argument("--resultados", action="store_true",
                        help="Mostrar todos los resultados registrados")
    args = parser.parse_args()

    if args.resultado:
        parsed = _parse_result_input(args.resultado)
        if parsed:
            home, hg, away, ag = parsed
            home_t = find_team(home)
            away_t = find_team(away)
            if home_t and away_t:
                res = add_result(home_t, away_t, hg, ag)
                if "error" in res:
                    print(color(f"  [ERROR] {res['error']}", "red"))
                else:
                    m = res["match"]
                    print(color(f"  [OK] {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}", "green"))
        else:
            print(color("  Formato no válido. Usa: --resultado 'Argentina 3 Algeria 0'", "red"))

    elif args.resultados:
        print_all_results()

    elif args.home and args.away:
        print_header()
        print(color("  Cargando modelos...", "dim"))
        dc_model = load_dc_model()
        sim_data = load_simulation_data()
        home = find_team(args.home)
        away = find_team(args.away)
        if home and away:
            pred = predict_full(home, away, dc_model, sim_data)
            print_prediction(pred)
            print_score_heatmap(pred)

    elif args.grupo:
        print_header()
        sim_data = load_simulation_data()
        g = args.grupo.upper()
        if g in GROUPS:
            print_group_matches(g, sim_data)
        else:
            print(color(f"  Grupo '{g}' no válido. Usa A-L.", "red"))

    elif args.tabla:
        print_header()
        dc_model = load_dc_model()
        sim_data = load_simulation_data()
        g = args.tabla.upper()
        if g in GROUPS:
            print_standings(g, dc_model, sim_data)
        else:
            print(color(f"  Grupo '{g}' no válido. Usa A-L.", "red"))

    elif args.favoritos:
        print_header()
        sim_data = load_simulation_data()
        champ = sim_data.get("champion_probs", [])
        if champ:
            for i, c in enumerate(champ[:20]):
                print(f"  {i+1:>2d}. {c['team']:20s}  {c.get('p_champion',0):.2%}")

    else:
        interactive_menu()
