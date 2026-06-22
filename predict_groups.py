"""
predict_groups.py — Predicciones de Fase de Grupos del Mundial 2026
====================================================================
Muestra los 72 partidos de fase de grupos con:
  - Probabilidades 1X2 (Victoria local / Empate / Victoria visitante)
  - Marcador más probable
  - Goles esperados
  - Favorito destacado

Uso:
  python predict_groups.py              # Todos los grupos
  python predict_groups.py --group A    # Solo grupo A
  python predict_groups.py --group C H  # Solo grupos C y H
"""

import sys
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_MODEL
from simulation.wc2026_fixtures import WC2026_GROUPS


# ─── Colores ANSI para terminal ──────────────────────────────────────────────
class C:
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    BG_BLUE = "\033[44m"
    UNDERLINE = "\033[4m"


def load_simulation_data() -> dict:
    """Carga los datos de la simulación Monte Carlo."""
    sim_path = DATA_MODEL / "wc2026_simulation.json"
    if not sim_path.exists():
        print(f"  ERROR: No se encontró {sim_path}")
        print(f"  Ejecutar primero: python run_final.py")
        sys.exit(1)
    with open(sim_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_winner_label(ph, pd, pa, home, away):
    """Determina el favorito y su confianza."""
    if ph > pa and ph > pd:
        conf = ph
        winner = home
        emoji = "C"
    elif pa > ph and pa > pd:
        conf = pa
        winner = away
        emoji = "F "
    else:
        conf = pd
        winner = "EMPATE"
        emoji = "E"
    
    if conf > 0.55:
        strength = f"{C.GREEN}■■■■■{C.RESET}"
    elif conf > 0.45:
        strength = f"{C.YELLOW}■■■■{C.DIM}■{C.RESET}"
    elif conf > 0.35:
        strength = f"{C.YELLOW}■■■{C.DIM}■■{C.RESET}"
    else:
        strength = f"{C.RED}■■{C.DIM}■■■{C.RESET}"
    
    return winner, conf, emoji, strength


def print_header():
    print()
    print(f"{C.BOLD}{C.CYAN}{'═' * 82}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}    PREDICCIONES — FASE DE GRUPOS — MUNDIAL 2026{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}      Basadas en 10,000 simulaciones Monte Carlo{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 82}{C.RESET}")


def print_group_header(group_letter, teams):
    print(f"\n{C.BOLD}{C.BG_BLUE}{C.WHITE}  GRUPO {group_letter}  {C.RESET}  ", end="")
    print(f"{C.DIM}{' · '.join(teams)}{C.RESET}")
    print(f"  {'─' * 78}")


def print_match(match_data, idx):
    home = match_data["home_team"]
    away = match_data["away_team"]
    ph   = match_data["p_home_win"]
    pd_  = match_data["p_draw"]
    pa   = match_data["p_away_win"]
    score = match_data["most_likely_score"]
    avg_h = match_data["avg_home_goals"]
    avg_a = match_data["avg_away_goals"]
    score_prob = match_data["most_likely_score_prob"]
    
    winner, conf, emoji, strength = get_winner_label(ph, pd_, pa, home, away)
    
    # Barras de probabilidad visual
    bar_len = 30
    h_bar = int(ph * bar_len)
    d_bar = int(pd_ * bar_len)
    a_bar = bar_len - h_bar - d_bar
    prob_bar = f"{C.GREEN}{'█' * h_bar}{C.YELLOW}{'█' * d_bar}{C.RED}{'█' * a_bar}{C.RESET}"
    
    # Línea principal
    print(f"\n  {C.BOLD}{C.WHITE}  {home:>22s}  {C.CYAN}vs{C.RESET}  {C.BOLD}{C.WHITE}{away:<22s}{C.RESET}")
    
    # Probabilidades
    print(f"    {C.GREEN}Victoria {home[:12]:12s}: {ph:>5.1%}{C.RESET}  │  "
          f"{C.YELLOW}Empate: {pd_:>5.1%}{C.RESET}  │  "
          f"{C.RED}Victoria {away[:12]:12s}: {pa:>5.1%}{C.RESET}")
    
    # Barra visual
    print(f"    {prob_bar}")
    
    # Marcador + favorito
    print(f"    {C.BOLD}[TABLA] Marcador más probable: {C.MAGENTA}{score}{C.RESET} "
          f"{C.DIM}({score_prob:.0%}){C.RESET}"
          f"   │   Goles esperados: {C.CYAN}{avg_h:.1f}{C.RESET} – {C.CYAN}{avg_a:.1f}{C.RESET}"
          f"   │   {emoji} {C.BOLD}{winner}{C.RESET} {strength}")
    
    print(f"  {'─' * 78}")


def print_group_standings(sim_data, group_letter):
    """Muestra la tabla de posiciones esperada del grupo."""
    champion_probs = sim_data.get("champion_probs", [])
    group_teams = []
    for t in champion_probs:
        if t.get("group") == group_letter:
            group_teams.append(t)
    
    if not group_teams:
        return
    
    group_teams.sort(key=lambda x: -x.get("p_group_advance", 0))
    
    print(f"\n  {C.BOLD}{C.UNDERLINE}Tabla esperada:{C.RESET}")
    print(f"    {'Equipo':22s}  {'Clasifica':>10s}  {'Octavos':>8s}  {'Campeón':>8s}  {'Pts Prom':>8s}")
    for t in group_teams:
        clasif = t.get("p_group_advance", 0)
        r16 = t.get("p_r16", 0)
        champ = t.get("p_champion", 0)
        pts = t.get("avg_group_pts", 0)
        
        if clasif > 0.85:
            color = C.GREEN
        elif clasif > 0.50:
            color = C.YELLOW
        else:
            color = C.RED
        
        print(f"    {color}{t['team']:22s}  {clasif:>9.1%}  {r16:>7.1%}  {champ:>7.1%}  {pts:>8.1f}{C.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Predicciones de Fase de Grupos — Mundial 2026")
    parser.add_argument("--group", nargs="*", default=None,
                        help="Grupos a mostrar (ej: --group A C H). Sin argumento = todos.")
    args = parser.parse_args()

    sim_data = load_simulation_data()
    group_matches = sim_data.get("group_matches", {})

    # Filtrar grupos
    if args.group:
        show_groups = [g.upper() for g in args.group]
    else:
        show_groups = list("ABCDEFGHIJKL")

    print_header()

    for group_letter in show_groups:
        teams = WC2026_GROUPS.get(group_letter, [])
        if not teams:
            continue

        print_group_header(group_letter, teams)

        # Buscar partidos de este grupo
        idx = 0
        for key, match in group_matches.items():
            if match.get("group") == group_letter:
                idx += 1
                print_match(match, idx)

        # Tabla de posiciones esperada
        print_group_standings(sim_data, group_letter)
    
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 82}{C.RESET}")
    n_sims = sim_data.get("meta", {}).get("n_sims", "?")
    print(f"{C.DIM}  Modelo: Ensemble 3 Capas (Dixon-Coles + Elo + RF + Bayesian + XGBoost + Calibración){C.RESET}")
    print(f"{C.DIM}  Basado en {n_sims:,} simulaciones Monte Carlo con datos actualizados al 19/06/2026{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 82}{C.RESET}\n")


if __name__ == "__main__":
    main()
