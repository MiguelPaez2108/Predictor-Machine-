"""
odds_comparison.py — Comparativa predicciones del modelo vs odds del mercado
=============================================================================
Conecta con The Odds API (https://the-odds-api.com) para obtener cuotas
reales de casas de apuestas y comparar con las probabilidades del modelo.

También integra predicción de jugadores con más probabilidad de marcar
usando stats de Sofascore (xG, goles, minutos jugados en el torneo).

API Key: aa55ec5be4748f07a76766ac6f1efc19

Uso:
  python odds_comparison.py                             # menú interactivo
  python odds_comparison.py --match "Argentina vs Cape Verde"
  python odds_comparison.py --all-r32                  # todos los cruces R32
  python odds_comparison.py --scorers "Brasil vs Japón" # goleadores probables
  python odds_comparison.py --value                    # apuestas con valor positivo
"""

import sys
import json
import math
import time
import warnings
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_MODEL, DATA_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

ODDS_API_KEY    = "aa55ec5be4748f07a76766ac6f1efc19"
ODDS_API_BASE   = "https://api.the-odds-api.com/v4"
SPORT_KEY       = "soccer_fifa_world_cup"

# Casas de apuestas preferidas (de mayor a menor reputación)
PREFERRED_BOOKS = [
    "betfair_ex_eu", "pinnacle", "bet365", "draftkings",
    "fanduel", "bovada", "unibet_eu", "williamhill",
    "betcris", "mybookieag"
]

# ─────────────────────────────────────────────────────────────────────────────
# Cruces reales del R32
# ─────────────────────────────────────────────────────────────────────────────

WC2026_R32 = [
    # Lado izquierdo
    ("Germany",       "Paraguay"),
    ("France",        "Sweden"),
    ("South Africa",  "Canada"),
    ("Netherlands",   "Morocco"),
    ("Portugal",      "Croatia"),
    ("Spain",         "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium",       "Senegal"),
    # Lado derecho
    ("Brazil",        "Japan"),
    ("Ivory Coast",   "Norway"),
    ("Mexico",        "Ecuador"),
    ("England",       "DR Congo"),
    ("Argentina",     "Cape Verde"),
    ("Australia",     "Egypt"),
    ("Switzerland",   "Algeria"),
    ("Colombia",      "Ghana"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Colores terminal
# ─────────────────────────────────────────────────────────────────────────────

def c(text, code):
    codes = {
        "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
        "blue": "\033[94m", "cyan": "\033[96m", "magenta": "\033[95m",
        "white": "\033[97m", "bold": "\033[1m", "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return codes.get(code, "") + text + codes["reset"]


def prob_to_odds(p: float) -> float:
    """Probabilidad → cuota decimal."""
    if p <= 0 or p >= 1:
        return 0.0
    return round(1.0 / p, 2)


def odds_to_prob(decimal_odds: float) -> float:
    """Cuota decimal → probabilidad implícita."""
    if decimal_odds <= 1.0:
        return 0.0
    return round(1.0 / decimal_odds, 4)


def kelly_fraction(model_prob: float, market_odds: float, fraction: float = 0.25) -> float:
    """
    Criterio de Kelly fraccionado para dimensionar la apuesta.
    fraction=0.25 → Kelly al 25% (más conservador).
    """
    if market_odds <= 1 or model_prob <= 0:
        return 0.0
    b = market_odds - 1
    q = 1 - model_prob
    k = (b * model_prob - q) / b
    return round(max(0.0, k * fraction), 4)


# ─────────────────────────────────────────────────────────────────────────────
# The Odds API — obtener cuotas
# ─────────────────────────────────────────────────────────────────────────────

def fetch_odds(sport: str = SPORT_KEY,
               markets: str = "h2h",
               regions: str = "eu,us") -> List[Dict]:
    """
    Obtiene las cuotas disponibles para el WC2026 desde The Odds API.
    Devuelve lista de partidos con cuotas de múltiples casas.
    """
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey":   ODDS_API_KEY,
        "regions":  regions,
        "markets":  markets,
        "oddsFormat": "decimal",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")

        if resp.status_code == 401:
            print(c("  [ERROR] API key inválida o sin cuota.", "red"))
            return []
        elif resp.status_code == 422:
            print(c("  [INFO] No hay eventos disponibles ahora para WC2026.", "yellow"))
            return []
        elif resp.status_code != 200:
            print(c(f"  [ERROR] HTTP {resp.status_code}: {resp.text[:200]}", "red"))
            return []

        data = resp.json()
        print(c(f"  [API] {len(data)} eventos | Requests restantes: {remaining} | Usados: {used}", "dim"))
        return data

    except requests.RequestException as e:
        print(c(f"  [ERROR] Conexión: {e}", "red"))
        return []


def find_match_in_odds(home: str, away: str,
                       odds_data: List[Dict]) -> Optional[Dict]:
    """Busca un partido específico en los datos de cuotas."""
    home_lower = home.lower()
    away_lower = away.lower()

    for event in odds_data:
        evt_home = event.get("home_team", "").lower()
        evt_away = event.get("away_team", "").lower()

        # Match directo o invertido
        if ((home_lower in evt_home or evt_home in home_lower) and
            (away_lower in evt_away or evt_away in away_lower)):
            return event
        if ((away_lower in evt_home or evt_home in away_lower) and
            (home_lower in evt_away or evt_away in home_lower)):
            # Invertir home/away
            event = dict(event)
            event["home_team"], event["away_team"] = event["away_team"], event["home_team"]
            for bk in event.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "h2h":
                        outcomes = mkt["outcomes"]
                        if len(outcomes) >= 2:
                            outcomes[0], outcomes[1] = outcomes[1], outcomes[0]
            return event

    return None


def extract_best_odds(event: Dict) -> Dict:
    """
    Extrae las mejores cuotas disponibles para H/D/A del evento.
    'Mejor cuota' = mayor decimal (más favorable para el apostador).
    """
    best = {"home": 0.0, "draw": 0.0, "away": 0.0,
            "book_home": "—", "book_draw": "—", "book_away": "—",
            "n_books": 0}

    if not event:
        return best

    home_name = event.get("home_team", "")
    bookmakers = event.get("bookmakers", [])
    best["n_books"] = len(bookmakers)

    for bk in bookmakers:
        bk_name = bk.get("title", bk.get("key", "?"))
        for market in bk.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = float(outcome.get("price", 0))

                if name.lower() == "draw":
                    if price > best["draw"]:
                        best["draw"] = price
                        best["book_draw"] = bk_name
                elif home_name.lower() in name.lower() or name.lower() in home_name.lower():
                    if price > best["home"]:
                        best["home"] = price
                        best["book_home"] = bk_name
                else:
                    if price > best["away"]:
                        best["away"] = price
                        best["book_away"] = bk_name

    return best


def get_average_odds(event: Dict) -> Dict:
    """Calcula las cuotas promedio entre todas las casas."""
    sums = {"home": [], "draw": [], "away": []}
    if not event:
        return {
            "home": 0.0,
            "draw": 0.0,
            "away": 0.0,
            "n_books": 0,
        }
    home_name = event.get("home_team", "")

    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = float(outcome.get("price", 0))
                if name.lower() == "draw":
                    sums["draw"].append(price)
                elif home_name.lower() in name.lower() or name.lower() in home_name.lower():
                    sums["home"].append(price)
                else:
                    sums["away"].append(price)

    return {
        "home": round(np.mean(sums["home"]), 2) if sums["home"] else 0.0,
        "draw": round(np.mean(sums["draw"]), 2) if sums["draw"] else 0.0,
        "away": round(np.mean(sums["away"]), 2) if sums["away"] else 0.0,
        "n_books": len(event.get("bookmakers", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo Dixon-Coles
# ─────────────────────────────────────────────────────────────────────────────

def load_dc_model():
    path = DATA_MODEL / "dixon_coles.pkl"
    if not path.exists():
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sim_data() -> Dict:
    path = DATA_MODEL / "wc2026_simulation.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_probs(home: str, away: str, dc_model=None, sim_data: Dict = None) -> Dict:
    """Obtiene probabilidades del modelo para un partido."""
    result = {"home": 0.333, "draw": 0.334, "away": 0.333,
              "lambda_home": 1.3, "lambda_away": 1.0, "source": "prior"}

    # Primero intentar Monte Carlo (más preciso, incluye toda la capa del ensemble)
    if sim_data:
        gm = sim_data.get("group_matches", {})
        key = f"{home}|{away}"
        key_rev = f"{away}|{home}"
        m = gm.get(key) or gm.get(key_rev)
        if m:
            if key_rev in gm and key not in gm:
                result.update({
                    "home": m["p_away_win"], "draw": m["p_draw"], "away": m["p_home_win"],
                    "lambda_home": m.get("avg_away_goals", 1.0),
                    "lambda_away": m.get("avg_home_goals", 1.3),
                    "source": "monte_carlo",
                })
            else:
                result.update({
                    "home": m["p_home_win"], "draw": m["p_draw"], "away": m["p_away_win"],
                    "lambda_home": m.get("avg_home_goals", 1.3),
                    "lambda_away": m.get("avg_away_goals", 1.0),
                    "source": "monte_carlo",
                })
            return result

    # Fallback a Dixon-Coles
    if dc_model and dc_model.is_fitted:
        try:
            pred = dc_model.predict_match(home, away)
            result.update({
                "home": pred["p_home_win"],
                "draw": pred["p_draw"],
                "away": pred["p_away_win"],
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "source": "dixon_coles",
            })
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Predicción de goleadores
# ─────────────────────────────────────────────────────────────────────────────

def load_player_stats() -> pd.DataFrame:
    """Carga stats de jugadores del WC2026 desde Sofascore."""
    path = DATA_RAW / "sofascore_wc2026" / "match_player_stats.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return df


def load_shots_data() -> pd.DataFrame:
    """Carga datos de tiros del WC2026."""
    path = DATA_RAW / "sofascore_wc2026" / "match_shots.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def predict_scorers(home: str, away: str, top_n: int = 5) -> pd.DataFrame:
    """
    Predice los jugadores con más probabilidad de marcar en el partido.
    Basado en xG acumulado, goles marcados y tiros en el torneo.
    """
    player_df = load_player_stats()
    shots_df  = load_shots_data()

    if player_df.empty:
        return pd.DataFrame()

    # Detectar columna de equipo
    team_col = next((c for c in ["team", "teamName", "squad"] if c in player_df.columns), None)
    if not team_col:
        return pd.DataFrame()

    # Detectar columna de nombre
    name_col = next((c for c in ["name", "playerName", "player"] if c in player_df.columns), None)
    if not name_col:
        return pd.DataFrame()

    records = []
    for team in [home, away]:
        # Filtrar por equipo (búsqueda flexible)
        mask = player_df[team_col].str.contains(
            team.split()[0], case=False, na=False
        )
        team_players = player_df[mask].copy()

        if team_players.empty:
            continue

        # Agrupar por jugador
        agg_dict = {}

        # xG acumulado
        xg_col = next((c for c in ["expectedGoals", "xg", "xG", "expected_goals"]
                      if c in team_players.columns), None)
        if xg_col:
            agg_dict[xg_col] = "sum"

        # Goles marcados
        goals_col = next((c for c in ["goals", "gls", "Gls", "goal"]
                         if c in team_players.columns), None)
        if goals_col:
            agg_dict[goals_col] = "sum"

        # Rating Sofascore
        rating_col = next((c for c in ["rating", "sofascoreRating", "sofascore_rating"]
                          if c in team_players.columns), None)
        if rating_col:
            agg_dict[rating_col] = "mean"

        # Minutos jugados
        mins_col = next((c for c in ["minutesPlayed", "minutes", "mins"]
                        if c in team_players.columns), None)
        if mins_col:
            agg_dict[mins_col] = "sum"

        grouped = team_players.groupby(name_col).agg(agg_dict).reset_index()

        for _, row in grouped.iterrows():
            xg = float(row.get(xg_col, 0) or 0) if xg_col else 0.0
            goals = int(row.get(goals_col, 0) or 0) if goals_col else 0
            rating = float(row.get(rating_col, 6.5) or 6.5) if rating_col else 6.5
            mins = float(row.get(mins_col, 0) or 0) if mins_col else 0.0

            if mins < 30:  # Filtrar jugadores con muy pocos minutos
                continue

            # Puntuación combinada: xG ponderado + goles reales + rating
            scorer_score = (
                xg * 0.5 +
                goals * 0.35 +
                (rating - 6.0) * 0.15 +
                (mins / 90) * 0.05
            )

            # Probabilidad de marcar en el partido (aprox.)
            # Basado en xG promedio por partido
            n_matches = max(1, round(mins / 90))
            xg_per_match = xg / n_matches if n_matches > 0 else 0.0
            p_score = 1 - math.exp(-max(0, xg_per_match)) if xg_per_match > 0 else 0.0

            records.append({
                "team":          team,
                "player":        str(row[name_col]),
                "goals":         goals,
                "xg_total":      round(xg, 3),
                "xg_per_match":  round(xg_per_match, 3),
                "p_score_match": round(p_score, 3),
                "rating":        round(rating, 2),
                "minutes":       int(mins),
                "scorer_score":  round(scorer_score, 3),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("scorer_score", ascending=False)

    # Top N por equipo
    result = pd.concat([
        df[df["team"] == home].head(top_n),
        df[df["team"] == away].head(top_n),
    ]).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Comparativa modelo vs mercado
# ─────────────────────────────────────────────────────────────────────────────

def compare_match(home: str, away: str,
                   dc_model=None, sim_data: Dict = None,
                   odds_data: List[Dict] = None,
                   show_scorers: bool = True) -> Dict:
    """Análisis completo de un partido: modelo vs mercado vs goleadores."""

    model = get_model_probs(home, away, dc_model, sim_data)
    event = find_match_in_odds(home, away, odds_data or [])
    best_odds = extract_best_odds(event)
    avg_odds  = get_average_odds(event)

    result = {
        "home": home, "away": away,
        "model": model,
        "best_odds": best_odds,
        "avg_odds": avg_odds,
        "has_market_data": event is not None,
    }

    # Probabilidades implícitas del mercado (promedio)
    mkt_prob_home = odds_to_prob(avg_odds["home"]) if avg_odds["home"] > 1 else None
    mkt_prob_draw = odds_to_prob(avg_odds["draw"]) if avg_odds["draw"] > 1 else None
    mkt_prob_away = odds_to_prob(avg_odds["away"]) if avg_odds["away"] > 1 else None

    # Ventaja del modelo (edge) = P_modelo - P_mercado
    if mkt_prob_home:
        result["edge_home"] = round(model["home"] - mkt_prob_home, 4)
        result["kelly_home"] = kelly_fraction(model["home"], avg_odds["home"])
    if mkt_prob_draw:
        result["edge_draw"] = round(model["draw"] - mkt_prob_draw, 4)
        result["kelly_draw"] = kelly_fraction(model["draw"], avg_odds["draw"])
    if mkt_prob_away:
        result["edge_away"] = round(model["away"] - mkt_prob_away, 4)
        result["kelly_away"] = kelly_fraction(model["away"], avg_odds["away"])

    # Goleadores probables
    if show_scorers:
        scorers = predict_scorers(home, away)
        result["scorers"] = scorers.to_dict("records") if not scorers.empty else []

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(comp: Dict):
    home = comp["home"]
    away = comp["away"]
    model = comp["model"]
    best = comp["best_odds"]
    avg = comp["avg_odds"]
    w = 68

    print()
    print(c("╔" + "═" * w + "╗", "cyan"))
    title = f"  {home}  vs  {away}  "
    print(c("║", "cyan") + c(f"{title:^{w}s}", "bold") + c("║", "cyan"))
    print(c("║", "cyan") + c(f"{'[ ODDS VS MODELO ]':^{w}s}", "dim") + c("║", "cyan"))
    print(c("╠" + "═" * w + "╣", "cyan"))

    # Probabilidades del modelo
    print(c("║", "cyan") + c(f"{'MODELO (ensemble)':^{w}s}", "bold") + c("║", "cyan"))
    print(c("║", "cyan") + " " * w + c("║", "cyan"))

    ph = model["home"]
    pd_ = model["draw"]
    pa = model["away"]
    lh = model.get("lambda_home", 1.3)
    la = model.get("lambda_away", 1.0)
    src = model.get("source", "?")

    line = (f"  {home[:16]:16s}: {c(f'{ph:.1%}', 'green')} ({prob_to_odds(ph):.2f})"
            f"   Empate: {c(f'{pd_:.1%}', 'yellow')} ({prob_to_odds(pd_):.2f})"
            f"   {away[:16]:16s}: {c(f'{pa:.1%}', 'green')} ({prob_to_odds(pa):.2f})")
    print(c("║", "cyan") + f"  {line}")

    line2 = f"  λ_home={lh:.2f}  λ_away={la:.2f}  (fuente: {src})"
    print(c("║", "cyan") + c(f"{line2:<{w}s}", "dim") + c("║", "cyan"))
    print(c("║", "cyan") + " " * w + c("║", "cyan"))

    # Cuotas del mercado
    print(c("╠" + "═" * w + "╣", "cyan"))

    if comp["has_market_data"]:
        n_books = avg.get("n_books", 0)
        print(c("║", "cyan") + c(f"{'MERCADO (' + str(n_books) + ' casas)':^{w}s}", "bold") + c("║", "cyan"))
        print(c("║", "cyan") + " " * w + c("║", "cyan"))

        # Mejor cuota disponible
        b_h = best.get("home", 0)
        b_d = best.get("draw", 0)
        b_a = best.get("away", 0)
        bk_h = best.get("book_home", "—")
        bk_d = best.get("book_draw", "—")
        bk_a = best.get("book_away", "—")

        hdr = f"  {'Resultado':12s} {'Mejor Cuota':>12s} {'Casa':>15s} {'Cuota Prom':>12s} {'P Implícita':>12s} {'Edge Modelo':>12s}"
        print(c("║", "cyan") + c(f"{hdr:<{w}s}", "dim") + c("║", "cyan"))
        print(c("║", "cyan") + f"  {'─'*64}" + c("║", "cyan"))

        rows = [
            (f"1 {home[:12]}", b_h, bk_h, avg["home"], ph,
             comp.get("edge_home", 0), comp.get("kelly_home", 0)),
            ("X Empate",       b_d, bk_d, avg["draw"], pd_,
             comp.get("edge_draw", 0), comp.get("kelly_draw", 0)),
            (f"2 {away[:12]}", b_a, bk_a, avg["away"], pa,
             comp.get("edge_away", 0), comp.get("kelly_away", 0)),
        ]

        for label, best_q, book, avg_q, model_p, edge, kelly in rows:
            p_impl = odds_to_prob(avg_q) if avg_q > 1 else 0.0
            edge_str = f"{edge:+.1%}" if edge else "N/A"
            kelly_str = f"{kelly:.1%}" if kelly and kelly > 0 else "—"

            # Color según edge
            if edge and edge > 0.03:
                edge_c = "green"
                marker = "★"
            elif edge and edge < -0.03:
                edge_c = "red"
                marker = "▼"
            else:
                edge_c = "yellow"
                marker = "≈"

            line = (f"  {label:12s} "
                    f"{best_q:>10.2f}  {book:>14s}  "
                    f"{avg_q:>10.2f}  "
                    f"{p_impl:>10.1%}  "
                    f"{c(f'{marker} {edge_str}', edge_c):>22s}")
            print(c("║", "cyan") + f"{line}")

            if kelly and kelly > 0:
                kelly_line = f"    {'':12s} Kelly (25%): {kelly_str}"
                print(c("║", "cyan") + c(f"{kelly_line:<{w}s}", "dim") + c("║", "cyan"))

        print(c("║", "cyan") + " " * w + c("║", "cyan"))
    else:
        print(c("║", "cyan") + c(f"{'  Sin datos de mercado disponibles aún':^{w}s}", "yellow") + c("║", "cyan"))
        print(c("║", "cyan") + c(f"{'  (El partido puede no estar abierto en las casas todavía)':^{w}s}", "dim") + c("║", "cyan"))
        print(c("║", "cyan") + " " * w + c("║", "cyan"))

    # Goleadores probables
    scorers = comp.get("scorers", [])
    if scorers:
        print(c("╠" + "═" * w + "╣", "cyan"))
        print(c("║", "cyan") + c(f"{'JUGADORES CON MÁS PROBABILIDAD DE MARCAR':^{w}s}", "bold") + c("║", "cyan"))
        print(c("║", "cyan") + " " * w + c("║", "cyan"))

        hdr = f"  {'Jugador':22s} {'Equipo':14s} {'Goles':>6s} {'xG Total':>9s} {'P(marcar)':>10s} {'Rating':>7s}"
        print(c("║", "cyan") + c(f"{hdr:<{w}s}", "dim") + c("║", "cyan"))
        print(c("║", "cyan") + f"  {'─'*64}" + c("║", "cyan"))

        prev_team = None
        for row in scorers[:10]:
            team = row.get("team", "")
            if team != prev_team:
                print(c("║", "cyan") + c(f"  ── {team} ──", "cyan") + " " * max(0, w - len(f"  ── {team} ──")) + c("║", "cyan"))
                prev_team = team

            player = row.get("player", "?")[:20]
            goals  = row.get("goals", 0)
            xg_tot = row.get("xg_total", 0)
            p_sc   = row.get("p_score_match", 0)
            rating = row.get("rating", 0)

            p_color = "green" if p_sc > 0.3 else ("yellow" if p_sc > 0.15 else "white")

            line = (f"  {player:22s} {team[:14]:14s} "
                    f"{goals:>6d} {xg_tot:>9.2f} "
                    f"{c(f'{p_sc:.1%}', p_color):>19s} "
                    f"{rating:>7.2f}")
            print(c("║", "cyan") + f"{line}")

        print(c("║", "cyan") + " " * w + c("║", "cyan"))
        print(c("║", "cyan") + c("  * P(marcar) estimada basada en xG promedio por partido en el torneo", "dim") + " " * max(0, w - 70) + c("║", "cyan"))
        print(c("║", "cyan") + " " * w + c("║", "cyan"))

    print(c("╚" + "═" * w + "╝", "cyan"))


def print_value_bets(comparisons: List[Dict], min_edge: float = 0.05):
    """Muestra las apuestas con mayor ventaja sobre el mercado."""
    print(c("\n  ╔══════════════════════════════════════════════════════╗", "green"))
    print(c("  ║         ★  APUESTAS CON VALOR POSITIVO  ★          ║", "green"))
    print(c("  ╚══════════════════════════════════════════════════════╝\n", "green"))

    value_bets = []
    for comp in comparisons:
        if not comp["has_market_data"]:
            continue
        for outcome, edge_key, prob_key, odds_key in [
            ("1 Victoria Local", "edge_home", "home", "home"),
            ("X Empate",         "edge_draw", "draw", "draw"),
            ("2 Victoria Vis.",  "edge_away", "away", "away"),
        ]:
            edge = comp.get(edge_key, 0) or 0
            if edge >= min_edge:
                value_bets.append({
                    "match":   f"{comp['home']} vs {comp['away']}",
                    "outcome": outcome,
                    "edge":    edge,
                    "p_model": comp["model"][prob_key.split("_")[0] if "_" in prob_key else prob_key],
                    "odds":    comp["avg_odds"].get(odds_key, 0),
                    "kelly":   comp.get(f"kelly_{odds_key}", 0) or 0,
                })

    if not value_bets:
        print(c(f"  Sin apuestas con edge ≥ {min_edge:.0%} en este momento.", "yellow"))
        return

    value_bets.sort(key=lambda x: -x["edge"])

    print(f"  {'Partido':30s} {'Resultado':18s} {'Edge':>8s} {'P Modelo':>9s} {'Odds':>7s} {'Kelly 25%':>10s}")
    print(f"  {'─'*30} {'─'*18} {'─'*8} {'─'*9} {'─'*7} {'─'*10}")

    for vb in value_bets:
        match_short = vb["match"][:28]
        print(f"  {match_short:30s} {vb['outcome']:18s} "
              f"""{c(f"+{vb['edge']:.1%}", 'green'):>17s} """
              f"{vb['p_model']:>9.1%} "
              f"{vb['odds']:>7.2f} "
              f"{vb['kelly']:>9.1%}")

    print(c("\n  ⚠  Kelly fraccionado al 25% — apostar responsablemente", "dim"))
    print(c("  ⚠  El modelo no garantiza ganancias. Usar solo como referencia analítica.\n", "dim"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comparativa predicciones vs odds + goleadores WC2026"
    )
    parser.add_argument("--match", type=str, default=None,
                        help="Partido: 'Argentina vs Cape Verde'")
    parser.add_argument("--all-r32", action="store_true",
                        help="Analizar todos los cruces de R32")
    parser.add_argument("--scorers", type=str, default=None,
                        help="Solo goleadores: 'Brasil vs Japón'")
    parser.add_argument("--value", action="store_true",
                        help="Mostrar solo apuestas con valor positivo (edge>5%)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Edge mínimo para value bets (default: 0.05)")
    args = parser.parse_args()

    print(c("\n  WC2026 — ODDS & PREDICCIONES", "bold"))
    print(c("  ─────────────────────────────────────────────\n", "cyan"))

    # Cargar modelos
    dc_model = load_dc_model()
    sim_data = load_sim_data()

    # Obtener odds
    print(c("  Obteniendo cuotas del mercado...", "dim"))
    odds_data = fetch_odds()

    if args.scorers:
        parts = re.split(r"\s+vs\s+", args.scorers, flags=re.IGNORECASE)
        if len(parts) == 2:
            home, away = parts[0].strip(), parts[1].strip()
            scorers = predict_scorers(home, away)
            if scorers.empty:
                print(c("  Sin datos de jugadores disponibles (ejecutar 08_scrape_sofascore_wc2026.py)", "yellow"))
            else:
                print(scorers.to_string(index=False))
        return

    if args.match:
        parts = re.split(r"\s+vs\s+", args.match, flags=re.IGNORECASE)
        if len(parts) == 2:
            home, away = parts[0].strip(), parts[1].strip()
            comp = compare_match(home, away, dc_model, sim_data, odds_data)
            print_comparison(comp)
        return

    if args.all_r32 or args.value:
        comps = []
        for home, away in WC2026_R32:
            comp = compare_match(home, away, dc_model, sim_data, odds_data,
                                show_scorers=not args.value)
            comps.append(comp)
            if not args.value:
                print_comparison(comp)
            time.sleep(0.1)

        if args.value:
            print_value_bets(comps, min_edge=args.min_edge)
        return

    # Sin argumentos: menú interactivo
    print("  Cruces del R32:\n")
    for i, (h, a) in enumerate(WC2026_R32, 1):
        side = "IZQ" if i <= 8 else "DER"
        print(f"  {i:>2d}. [{side}] {h:25s} vs {a}")

    print()
    sel = input(c("  → Número de cruce (o 'todos' para todos): ", "bold")).strip()

    if sel.lower() in ("todos", "all"):
        for home, away in WC2026_R32:
            comp = compare_match(home, away, dc_model, sim_data, odds_data)
            print_comparison(comp)
    else:
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(WC2026_R32):
                home, away = WC2026_R32[idx]
                comp = compare_match(home, away, dc_model, sim_data, odds_data)
                print_comparison(comp)
            else:
                print(c("  Número no válido.", "red"))
        except ValueError:
            print(c("  Entrada no válida.", "red"))


import re

if __name__ == "__main__":
    main()
