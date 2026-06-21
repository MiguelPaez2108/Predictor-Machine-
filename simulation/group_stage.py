"""
group_stage.py — Simulador de Fase de Grupos
=============================================
Simula los 12 grupos del Mundial 2026 (A–L), cada uno con 4 equipos
en round-robin (6 partidos por grupo, 72 en total).

Clasificación por grupo:
  - 3 puntos por victoria, 1 por empate, 0 por derrota
  - Desempate: diferencia de goles → goles a favor → resultado H2H
  - Clasifican: 1º y 2º directo + los 8 mejores 3ros

Funciones públicas:
  simulate_group(group_name, teams, models, rng)  → GroupResult
  simulate_all_groups(models, rng)                → AllGroupsResult
  select_best_third_place(all_results)            → list[str] (8 equipos)
"""

import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.match_simulator import simulate_group_match, get_models
from simulation.wc2026_fixtures import WC2026_GROUPS, GROUP_VENUES, WC2026_VENUES


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamGroupStats:
    team:      str
    group:     str
    played:    int = 0
    wins:      int = 0
    draws:     int = 0
    losses:    int = 0
    gf:        int = 0   # goles a favor
    ga:        int = 0   # goles en contra
    pts:       int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class GroupResult:
    group:        str
    standings:    List[TeamGroupStats]  # ordenado 1º→4º
    matches:      List[Dict]            # resultados de todos los partidos


# ─────────────────────────────────────────────────────────────────────────────
# Clasificación dentro de un grupo
# ─────────────────────────────────────────────────────────────────────────────

def _rank_group(stats: Dict[str, TeamGroupStats],
                h2h_results: Dict[Tuple[str, str], Dict]) -> List[TeamGroupStats]:
    """
    Ordena equipos por: 1) Pts  2) GD  3) GF  4) H2H Pts  5) aleatorio.
    Devuelve lista de equipos ordenada del 1º al 4º.
    """
    teams_list = list(stats.values())

    def sort_key(s: TeamGroupStats):
        # H2H entre equipos empatados (simplified: solo pts H2H)
        return (-s.pts, -s.gd, -s.gf)

    teams_list.sort(key=sort_key)
    # Asignar posiciones
    for pos, t in enumerate(teams_list):
        t.position = pos + 1  # type: ignore
    return teams_list


def load_real_results() -> List[Dict]:
    """Carga los resultados reales de los partidos registrados."""
    import json
    root = Path(__file__).parent.parent
    path = root / "data" / "resultados_reales.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Simular un grupo completo
# ─────────────────────────────────────────────────────────────────────────────

def simulate_group(group_name: str, teams: List[str],
                   models: Optional[Dict] = None,
                   rng: Optional[np.random.Generator] = None) -> GroupResult:
    """
    Simula los 6 partidos del grupo y devuelve la tabla de posiciones.
    Usa resultados reales si están registrados.
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    # Inicializar estadísticas
    stats: Dict[str, TeamGroupStats] = {
        t: TeamGroupStats(team=t, group=group_name) for t in teams
    }

    matches: List[Dict] = []
    h2h: Dict[Tuple[str, str], Dict] = {}
    real_results = load_real_results()

    import itertools
    for home, away in itertools.combinations(teams, 2):
        # Buscar si el partido tiene un resultado real registrado
        real = None
        for r in real_results:
            if (r["home_team"] == home and r["away_team"] == away) or \
               (r["home_team"] == away and r["away_team"] == home):
                real = r
                break

        if real:
            # Usar resultado real con la correspondencia correcta de local/visitante
            if real["home_team"] == home:
                hg = real["home_goals"]
                ag = real["away_goals"]
            else:
                hg = real["away_goals"]
                ag = real["home_goals"]
            res_char = "H" if hg > ag else ("A" if hg < ag else "D")
            result = {
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "result": res_char,
                "real": True
            }
        else:
            result = simulate_group_match(home, away, models, rng)
            result["real"] = False

        matches.append({**result, "group": group_name})

        hg, ag = result["home_goals"], result["away_goals"]

        # Actualizar estadísticas
        stats[home].played += 1
        stats[away].played += 1
        stats[home].gf += hg;  stats[home].ga += ag
        stats[away].gf += ag;  stats[away].ga += hg

        if result["result"] == "H":
            stats[home].wins   += 1;  stats[home].pts += 3
            stats[away].losses += 1
        elif result["result"] == "D":
            stats[home].draws += 1;  stats[home].pts += 1
            stats[away].draws += 1;  stats[away].pts += 1
        else:
            stats[away].wins   += 1;  stats[away].pts += 3
            stats[home].losses += 1

        h2h[(home, away)] = result

    standings = _rank_group(stats, h2h)
    return GroupResult(group=group_name, standings=standings, matches=matches)


# ─────────────────────────────────────────────────────────────────────────────
# Simular todos los grupos
# ─────────────────────────────────────────────────────────────────────────────

def simulate_all_groups(models: Optional[Dict] = None,
                        rng: Optional[np.random.Generator] = None) -> Dict[str, GroupResult]:
    """
    Simula los 12 grupos del Mundial 2026.
    Devuelve dict {grupo_letra: GroupResult}.
    """
    if models is None:
        models = get_models()
    if rng is None:
        rng = np.random.default_rng()

    results: Dict[str, GroupResult] = {}
    for group_name, teams in WC2026_GROUPS.items():
        results[group_name] = simulate_group(group_name, teams, models, rng)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Selección de los 8 mejores terceros
# ─────────────────────────────────────────────────────────────────────────────

def select_best_third_place(all_results: Dict[str, GroupResult]) -> List[str]:
    """
    Selecciona los 8 mejores terceros de los 12 grupos.
    Criterio: Pts → GD → GF → aleatorio.
    """
    thirds: List[TeamGroupStats] = []
    for gr in all_results.values():
        if len(gr.standings) >= 3:
            thirds.append(gr.standings[2])

    thirds.sort(key=lambda s: (-s.pts, -s.gd, -s.gf))
    return [t.team for t in thirds[:8]]


# ─────────────────────────────────────────────────────────────────────────────
# Extraer clasificados por posición
# ─────────────────────────────────────────────────────────────────────────────

def get_qualifier(all_results: Dict[str, GroupResult],
                  group: str, position: int) -> str:
    """
    Devuelve el equipo que quedó en 'position' (1-indexed) del grupo dado.
    """
    gr = all_results.get(group)
    if gr is None or len(gr.standings) < position:
        return f"?{group}{position}"
    return gr.standings[position - 1].team


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test Group Stage ===")
    models = get_models()
    print(f"Modelos: {list(models.keys())}")

    rng = np.random.default_rng(42)
    result = simulate_group("A", WC2026_GROUPS["A"], models, rng)

    print(f"\nGrupo {result.group}:")
    print(f"  {'Pos':3s} {'Equipo':25s} {'Pts':4s} {'GD':4s} {'GF':4s}")
    for i, t in enumerate(result.standings):
        print(f"  {i+1:<3d} {t.team:25s} {t.pts:<4d} {t.gd:<4d} {t.gf:<4d}")

    print(f"\nPartidos jugados:")
    for m in result.matches:
        print(f"  {m['home_team']:20s} {m['home_goals']}–{m['away_goals']} {m['away_team']}")
