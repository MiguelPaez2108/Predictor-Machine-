"""
knockout_stage.py — Simulador de Fase Eliminatoria
===================================================
Simula la llave eliminatoria completa del Mundial 2026 a partir
de los 32 clasificados (24 directos + 8 mejores terceros).

Formato oficial FIFA WC2026:
  Ronda de 32  (R32)  → 16 partidos
  Octavos      (R16)  → 8 partidos
  Cuartos      (QF)   → 4 partidos
  Semifinales  (SF)   → 2 partidos
  3º y 4º      (3P)   → 1 partido
  Final        (F)    → 1 partido

  Total: 32 partidos eliminatorios

Llave definida por FIFA (basada en el sorteo de Washington D.C.):
  Los cruces de R32 siguen la posición en el grupo, no los nombres
  de los equipos → se resuelven dinámicamente tras la fase de grupos.
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.match_simulator import simulate_knockout_match, get_models
from simulation.group_stage import (
    GroupResult, select_best_third_place, get_qualifier
)


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de datos del resultado eliminatorio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnockoutMatch:
    round:    str
    team1:    str
    team2:    str
    goals_1:  int = 0
    goals_2:  int = 0
    winner:   str = ""
    loser:    str = ""
    went_to_et:   bool = False
    went_to_pens: bool = False


@dataclass
class KnockoutResult:
    r32_matches:  List[KnockoutMatch] = field(default_factory=list)
    r16_matches:  List[KnockoutMatch] = field(default_factory=list)
    qf_matches:   List[KnockoutMatch] = field(default_factory=list)
    sf_matches:   List[KnockoutMatch] = field(default_factory=list)
    third_match:  Optional[KnockoutMatch] = None
    final_match:  Optional[KnockoutMatch] = None
    champion:     str = ""
    runner_up:    str = ""
    third_place:  str = ""
    fourth_place: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Llave oficial R32 del Mundial 2026
# ─────────────────────────────────────────────────────────────────────────────
# Fuente: FIFA / sorteo Washington D.C., diciembre 2024
# Formato: (slot_A, slot_B) donde slot = "1A", "2B", "3G", etc.
# Los 8 mejores terceros se distribuyen en la llave según los grupos
# de donde provengan. FIFA establece la distribución exacta de 3ros.

# Llave oficial de R32:
# (Bracket A) Matches 1-8:
WC2026_R32_BRACKET = [
    # Bracket superior
    ("1A", "2D"),   # Partido R32-1
    ("1B", "2E"),   # Partido R32-2
    ("1C", "2F"),   # Partido R32-3
    ("1D", "2A"),   # Partido R32-4
    ("1E", "2H"),   # Partido R32-5
    ("1F", "2I"),   # Partido R32-6
    ("1G", "2J"),   # Partido R32-7
    ("1H", "2G"),   # Partido R32-8
    ("1I", "2L"),   # Partido R32-9
    ("1J", "2K"),   # Partido R32-10
    ("1K", "2C"),   # Partido R32-11
    ("1L", "2B"),   # Partido R32-12
    # 8 mejores terceros (por rendimiento, se asignan a slots libres)
    # FIFA aún no ha publicado la llave exacta de terceros para 2026;
    # usamos distribución basada en precedente de WC2022 (donde 3eros
    # se insertan en los slots predefinidos de la llave). Por ahora:
    ("3ABCD", "1G_WINNER_SIDE"),   # placeholder — se resolverá dinámicamente
    ("3EFGH", "1J_WINNER_SIDE"),
    ("3IJKL", "1K_WINNER_SIDE"),
    ("3ABCL", "1L_WINNER_SIDE"),   # placeholder
]

# Nota: la distribución exacta de los 8 mejores terceros en 2026
# sigue siendo provisional (FIFA la publicará tras la fase de grupos).
# El simulador la gestiona asignando los 8 mejores 3ros a los 4
# cruces del bracket inferior.

# ─────────────────────────────────────────────────────────────────────────────
# Resolver slots de equipos
# ─────────────────────────────────────────────────────────────────────────────

def resolve_qualifiers(all_group_results: Dict[str, GroupResult]) -> Dict[str, str]:
    """
    Construye el mapa {slot → team} para los 32 clasificados.
    Slots: "1A"..."1L" (primeros), "2A"..."2L" (segundos), "3_best_1"..."3_best_8".
    """
    slots: Dict[str, str] = {}

    for group in "ABCDEFGHIJKL":
        gr = all_group_results.get(group)
        if gr is None or not gr.standings:
            slots[f"1{group}"] = f"1st_{group}"
            slots[f"2{group}"] = f"2nd_{group}"
            continue
        slots[f"1{group}"] = gr.standings[0].team
        if len(gr.standings) > 1:
            slots[f"2{group}"] = gr.standings[1].team
        else:
            slots[f"2{group}"] = f"2nd_{group}"

    best_thirds = select_best_third_place(all_group_results)
    for i, team in enumerate(best_thirds):
        slots[f"3_best_{i+1}"] = team

    return slots


def build_r32_matchups(slots: Dict[str, str],
                       best_thirds: List[str]) -> List[Tuple[str, str]]:
    """
    Construye los 16 enfrentamientos de R32 resolviendo los slots de grupos
    y asignando los 8 mejores terceros a los cuatro slots libres.

    Llave oficial de 12 grupos (R32 de 16 partidos):
    Los primeros (1A-1L) y segundos (2A-2L) se cruzan según la llave FIFA.
    Los 8 mejores terceros se reparten en los 4 cruces de terceros.
    """
    # 12 cruces directos 1º vs 2º (llave oficial WC2026)
    direct_matchups = [
        (slots.get("1A", "1A"), slots.get("2D", "2D")),
        (slots.get("1B", "1B"), slots.get("2E", "2E")),
        (slots.get("1C", "1C"), slots.get("2F", "2F")),
        (slots.get("1D", "1D"), slots.get("2A", "2A")),
        (slots.get("1E", "1E"), slots.get("2H", "2H")),
        (slots.get("1F", "1F"), slots.get("2I", "2I")),
        (slots.get("1G", "1G"), slots.get("2J", "2J")),
        (slots.get("1H", "1H"), slots.get("2G", "2G")),
        (slots.get("1I", "1I"), slots.get("2L", "2L")),
        (slots.get("1J", "1J"), slots.get("2K", "2K")),
        (slots.get("1K", "1K"), slots.get("2C", "2C")),
        (slots.get("1L", "1L"), slots.get("2B", "2B")),
    ]

    # 4 cruces de mejores terceros (distribuidos en parejas de 2)
    # Distribución simplificada: tercer[0-1] vs tercer[2-3], etc.
    thirds_matchups = []
    for i in range(0, min(len(best_thirds), 8), 2):
        t1 = best_thirds[i] if i < len(best_thirds) else f"3rd_{i+1}"
        t2 = best_thirds[i+1] if i+1 < len(best_thirds) else f"3rd_{i+2}"
        thirds_matchups.append((t1, t2))

    return direct_matchups + thirds_matchups


# ─────────────────────────────────────────────────────────────────────────────
# Simular una ronda eliminatoria
# ─────────────────────────────────────────────────────────────────────────────

def simulate_round(matchups: List[Tuple[str, str]],
                   round_name: str,
                   models: Optional[Dict] = None,
                   rng: Optional[np.random.Generator] = None) -> Tuple[List[KnockoutMatch], List[str]]:
    """
    Simula una ronda completa de partidos eliminatorios.
    Devuelve (lista de partidos, lista de ganadores en orden).
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    matches: List[KnockoutMatch] = []
    winners: List[str] = []

    for t1, t2 in matchups:
        result = simulate_knockout_match(t1, t2, models, rng)
        m = KnockoutMatch(
            round    = round_name,
            team1    = t1,
            team2    = t2,
            goals_1  = result["goals_team1"],
            goals_2  = result["goals_team2"],
            winner   = result["winner"],
            loser    = result["loser"],
            went_to_et   = result["went_to_et"],
            went_to_pens = result["went_to_penalties"],
        )
        matches.append(m)
        winners.append(result["winner"])

    return matches, winners


# ─────────────────────────────────────────────────────────────────────────────
# Simular fase eliminatoria completa
# ─────────────────────────────────────────────────────────────────────────────

def simulate_knockout_stage(all_group_results: Dict[str, GroupResult],
                             models: Optional[Dict] = None,
                             rng: Optional[np.random.Generator] = None) -> KnockoutResult:
    """
    Simula la fase eliminatoria completa desde R32 hasta la Final.
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    ko = KnockoutResult()

    # ── Resolver clasificados ─────────────────────────────────────────────────
    slots       = resolve_qualifiers(all_group_results)
    best_thirds = select_best_third_place(all_group_results)

    # ── Ronda de 32 ──────────────────────────────────────────────────────────
    r32_matchups = build_r32_matchups(slots, best_thirds)
    r32_matches, r32_winners = simulate_round(r32_matchups, "R32", models, rng)
    ko.r32_matches = r32_matches

    # ── Octavos (R16) ─────────────────────────────────────────────────────────
    # Los ganadores de R32 se emparejan en orden: win[0]vs win[1], win[2]vs win[3]...
    r16_matchups = [(r32_winners[i], r32_winners[i+1])
                    for i in range(0, len(r32_winners), 2)]
    r16_matches, r16_winners = simulate_round(r16_matchups, "R16", models, rng)
    ko.r16_matches = r16_matches

    # ── Cuartos (QF) ─────────────────────────────────────────────────────────
    qf_matchups = [(r16_winners[i], r16_winners[i+1])
                   for i in range(0, len(r16_winners), 2)]
    qf_matches, qf_winners = simulate_round(qf_matchups, "QF", models, rng)
    ko.qf_matches = qf_matches

    # ── Semis ─────────────────────────────────────────────────────────────────
    sf_matchups = [(qf_winners[i], qf_winners[i+1])
                   for i in range(0, len(qf_winners), 2)]
    sf_matches, sf_winners = simulate_round(sf_matchups, "SF", models, rng)
    ko.sf_matches = sf_matches

    # Perdedores de semis → partido por 3º y 4º
    sf_losers = [m.loser for m in sf_matches]

    # ── Partido por 3er lugar ─────────────────────────────────────────────────
    if len(sf_losers) >= 2:
        third_res = simulate_knockout_match(sf_losers[0], sf_losers[1], models, rng)
        ko.third_match = KnockoutMatch(
            round    = "3rd_Place",
            team1    = sf_losers[0],
            team2    = sf_losers[1],
            goals_1  = third_res["goals_team1"],
            goals_2  = third_res["goals_team2"],
            winner   = third_res["winner"],
            loser    = third_res["loser"],
            went_to_et   = third_res["went_to_et"],
            went_to_pens = third_res["went_to_penalties"],
        )
        ko.third_place  = third_res["winner"]
        ko.fourth_place = third_res["loser"]

    # ── Final ─────────────────────────────────────────────────────────────────
    if len(sf_winners) >= 2:
        final_res = simulate_knockout_match(sf_winners[0], sf_winners[1], models, rng)
        ko.final_match = KnockoutMatch(
            round    = "Final",
            team1    = sf_winners[0],
            team2    = sf_winners[1],
            goals_1  = final_res["goals_team1"],
            goals_2  = final_res["goals_team2"],
            winner   = final_res["winner"],
            loser    = final_res["loser"],
            went_to_et   = final_res["went_to_et"],
            went_to_pens = final_res["went_to_penalties"],
        )
        ko.champion   = final_res["winner"]
        ko.runner_up  = final_res["loser"]

    return ko


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulation.group_stage import simulate_all_groups

    print("=== Test Knockout Stage (1 simulación) ===")
    models = get_models()
    rng    = np.random.default_rng(42)

    print("Simulando fase de grupos...")
    group_results = simulate_all_groups(models, rng)

    print("Simulando fase eliminatoria...")
    ko = simulate_knockout_stage(group_results, models, rng)

    print(f"\n🏆 CAMPEÓN:    {ko.champion}")
    print(f"🥈 FINALISTA:  {ko.runner_up}")
    print(f"🥉 TERCER:     {ko.third_place}")
    print(f"4️⃣  CUARTO:     {ko.fourth_place}")

    print("\n--- Final ---")
    if ko.final_match:
        f = ko.final_match
        print(f"  {f.team1} {f.goals_1}–{f.goals_2} {f.team2}")
        if f.went_to_et:
            print("  (Tiempo extra)")
        if f.went_to_pens:
            print("  (Penaltis)")
