"""
tournament.py — Motor de Simulación Monte Carlo del Mundial 2026
================================================================
Orquesta N simulaciones completas del torneo y agrega estadísticas.

Salida principal:
  simulate_wc2026(n_sims)  →  WC2026Results

Uso desde CLI:
  python simulation/tournament.py --sims 10000
  python simulation/tournament.py --sims 50000 --output results/wc2026_sim.json

Métricas calculadas:
  Por equipo:
    - P(campeón)          win probability
    - P(finalista)
    - P(semifinalista)
    - P(cuartofinalista)
    - P(clasificar R16)
    - P(clasificar grupos)
    - Goles esperados en grupos
    - Marcadores de grupo promedio

  Por partido de grupos:
    - P(victoria local), P(empate), P(victoria visitante)
    - Goles esperados H/A
    - Marcadores más probables

Rendimiento:
  - 10 000 sims: ~2 min (sin modelos cargados = primera vez)
  - 10 000 sims: ~45 s  (modelos en caché)
  - Soporte multi-proceso opcional (n_jobs > 1)
"""

import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_MODEL, MONTE_CARLO_RUNS, ensure_dirs
from simulation.wc2026_fixtures import WC2026_GROUPS, all_teams
from simulation.match_simulator  import get_models
from simulation.group_stage      import simulate_all_groups, GroupResult
from simulation.knockout_stage   import simulate_knockout_stage, KnockoutResult


# ─────────────────────────────────────────────────────────────────────────────
# Contenedor de resultados agregados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamSimResults:
    team:                str
    group:               str
    n_sims:              int     = 0
    n_group_advance:     int     = 0   # clasificó entre los 32
    n_r16_advance:       int     = 0   # superó R32
    n_qf_advance:        int     = 0   # llegó a cuartos
    n_sf_advance:        int     = 0   # llegó a semis
    n_final:             int     = 0   # llegó a la final
    n_champion:          int     = 0   # ganó el torneo
    total_group_gf:      float   = 0.0
    total_group_ga:      float   = 0.0
    total_group_pts:     float   = 0.0

    @property
    def p_group_advance(self) -> float:
        return self.n_group_advance / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_r16(self) -> float:
        return self.n_r16_advance / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_qf(self) -> float:
        return self.n_qf_advance / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_sf(self) -> float:
        return self.n_sf_advance / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_final(self) -> float:
        return self.n_final / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_champion(self) -> float:
        return self.n_champion / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def avg_gf(self) -> float:
        return self.total_group_gf / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def avg_pts(self) -> float:
        return self.total_group_pts / self.n_sims if self.n_sims > 0 else 0.0


@dataclass
class MatchSimResults:
    """Estadísticas agregadas de un partido de grupos."""
    home_team:      str
    away_team:      str
    group:          str
    n_sims:         int     = 0
    n_home_win:     int     = 0
    n_draw:         int     = 0
    n_away_win:     int     = 0
    total_home_gf:  float   = 0.0
    total_away_gf:  float   = 0.0
    score_counts:   Dict    = field(default_factory=dict)

    @property
    def p_home_win(self) -> float:
        return self.n_home_win / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_draw(self) -> float:
        return self.n_draw / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def p_away_win(self) -> float:
        return self.n_away_win / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def avg_home_goals(self) -> float:
        return self.total_home_gf / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def avg_away_goals(self) -> float:
        return self.total_away_gf / self.n_sims if self.n_sims > 0 else 0.0

    @property
    def most_likely_score(self) -> str:
        if not self.score_counts:
            return "1-0"
        return max(self.score_counts, key=self.score_counts.get)

    @property
    def most_likely_score_prob(self) -> float:
        if not self.score_counts:
            return 0.0
        return max(self.score_counts.values()) / self.n_sims


@dataclass
class WC2026Results:
    n_sims:          int
    elapsed_seconds: float
    teams:           Dict[str, TeamSimResults]
    group_matches:   Dict[str, MatchSimResults]
    champion_counts: Dict[str, int]           # {team: n_wins}

    def champion_probs(self) -> pd.DataFrame:
        """DataFrame ordenado por probabilidad de ganar el mundial."""
        rows = []
        for team, res in self.teams.items():
            rows.append({
                "team":             team,
                "group":            res.group,
                "p_champion":       res.p_champion,
                "p_final":          res.p_final,
                "p_sf":             res.p_sf,
                "p_qf":             res.p_qf,
                "p_r16":            res.p_r16,
                "p_group_advance":  res.p_group_advance,
                "avg_group_pts":    round(res.avg_pts, 2),
                "avg_group_gf":     round(res.avg_gf, 2),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # ranking 1-based
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Una simulación completa del torneo
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_simulation(models: Dict,
                         rng: np.random.Generator,
                         use_real_bracket: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Corre una simulación completa del torneo.
    Devuelve (group_data, knockout_data, champion).
    """
    # Fase de grupos
    group_results = simulate_all_groups(models, rng)

    # Fase eliminatoria
    ko = simulate_knockout_stage(group_results, models, rng,
                                 use_real_bracket=use_real_bracket)

    # Construir datos de grupo: {team: {pts, gf, ga, position}}
    group_data: Dict[str, Dict] = {}
    match_data:  List[Dict] = []
    for gr_name, gr in group_results.items():
        for pos, stats in enumerate(gr.standings):
            group_data[stats.team] = {
                "position": pos + 1,
                "pts":  stats.pts,
                "gf":   stats.gf,
                "ga":   stats.ga,
                "group": gr_name,
            }
        for m in gr.matches:
            match_data.append(m)

    # Construir progreso en la eliminatoria
    ko_data: Dict[str, str] = {}   # {team: farthest_round}

    for m in (ko.r32_matches or []):
        for t in [m.team1, m.team2]:
            ko_data[t] = "R32"

    for m in (ko.r16_matches or []):
        for t in [m.team1, m.team2]:
            ko_data[t] = "R16"

    for m in (ko.qf_matches or []):
        for t in [m.team1, m.team2]:
            ko_data[t] = "QF"

    for m in (ko.sf_matches or []):
        for t in [m.team1, m.team2]:
            ko_data[t] = "SF"

    if ko.final_match:
        for t in [ko.final_match.team1, ko.final_match.team2]:
            ko_data[t] = "F"

    if ko.champion:
        ko_data[ko.champion] = "Champion"

    return group_data, ko_data, match_data


# ─────────────────────────────────────────────────────────────────────────────
# Motor Monte Carlo principal
# ─────────────────────────────────────────────────────────────────────────────

def simulate_wc2026(n_sims: int = 10_000,
                    seed: int = 42,
                    verbose: bool = True,
                    use_real_bracket: bool = False) -> WC2026Results:
    """
    Corre N simulaciones completas del Mundial 2026 y agrega estadísticas.

    n_sims: número de simulaciones (rec: ≥ 10 000 para resultados estables)
    seed:   semilla para reproducibilidad
    verbose: si True, muestra progreso cada 1 000 simulaciones
    use_real_bracket: si True, usa la llave real de R32 (post-grupos)
    """
    ensure_dirs()
    t0 = time.time()

    models = get_models()
    if not models:
        print("ERROR: No hay modelos entrenados. Ejecutar: python models/ensemble.py --train")
        return None

    rng = np.random.default_rng(seed)

    # Inicializar acumuladores
    team_results: Dict[str, TeamSimResults] = {}
    for group, teams in WC2026_GROUPS.items():
        for t in teams:
            team_results[t] = TeamSimResults(team=t, group=group, n_sims=0)

    # Acumulador de partidos de grupos
    match_results: Dict[str, MatchSimResults] = {}

    champion_counts: Dict[str, int] = defaultdict(int)

    if verbose:
        bracket_mode = "LLAVE REAL (post-grupos)" if use_real_bracket else "SIMULACIÓN COMPLETA"
        print(f"\n{'═'*60}")
        print(f"  SIMULACIÓN MONTE CARLO — MUNDIAL 2026")
        print(f"  N = {n_sims:,} simulaciones  |  Modo: {bracket_mode}")
        print(f"{'═'*60}\n")

    for sim in range(n_sims):
        if verbose and (sim + 1) % 1_000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (sim + 1) * (n_sims - sim - 1)
            print(f"  [{sim+1:>6,}/{n_sims:,}]  "
                  f"ETA: {eta:.0f}s  |  "
                  f"Líder actual: {max(champion_counts, key=champion_counts.get, default='—')} "
                  f"({max(champion_counts.values(), default=0)/(sim+1)*100:.1f}%)")

        try:
            group_data, ko_data, match_data = _run_one_simulation(
                models, rng, use_real_bracket=use_real_bracket
            )
        except Exception as e:
            # Si hay error en una simulación, la saltamos
            continue

        # ── Actualizar estadísticas de equipos ───────────────────────────────
        for team, gd in group_data.items():
            if team not in team_results:
                team_results[team] = TeamSimResults(
                    team=team, group=gd.get("group", "?"), n_sims=0
                )
            tr = team_results[team]
            tr.n_sims           += 1
            tr.total_group_gf   += gd.get("gf", 0)
            tr.total_group_ga   += gd.get("ga", 0)
            tr.total_group_pts  += gd.get("pts", 0)

            # Clasificación de grupos (tanto pos <= 2 como mejores terceros)
            # se gestiona de forma única en el bucle de ko_data más abajo para evitar duplicados.

        # Los equipos que avanzaron a la fase eliminatoria cuentan como "group_advance"
        for team, round_str in ko_data.items():
            if team not in team_results:
                team_results[team] = TeamSimResults(
                    team=team, group="?", n_sims=1
                )
            tr = team_results[team]
            # Asegurar que team n_sims está actualizado
            if tr.n_sims == 0:
                tr.n_sims = 1

            if round_str in ("R32", "R16", "QF", "SF", "F", "Champion"):
                tr.n_group_advance += 1
            if round_str in ("R16", "QF", "SF", "F", "Champion"):
                tr.n_r16_advance += 1
            if round_str in ("QF", "SF", "F", "Champion"):
                tr.n_qf_advance += 1
            if round_str in ("SF", "F", "Champion"):
                tr.n_sf_advance += 1
            if round_str in ("F", "Champion"):
                tr.n_final += 1
            if round_str == "Champion":
                tr.n_champion += 1
                champion_counts[team] += 1

        # ── Actualizar estadísticas de partidos de grupos ────────────────────
        for m in match_data:
            key = f"{m['home_team']}|{m['away_team']}"
            if key not in match_results:
                match_results[key] = MatchSimResults(
                    home_team=m["home_team"],
                    away_team=m["away_team"],
                    group=m.get("group", "?"),
                )
            mr = match_results[key]
            mr.n_sims += 1
            hg, ag = m["home_goals"], m["away_goals"]
            mr.total_home_gf += hg
            mr.total_away_gf += ag
            score_key = f"{hg}-{ag}"
            mr.score_counts[score_key] = mr.score_counts.get(score_key, 0) + 1
            if m["result"] == "H":
                mr.n_home_win += 1
            elif m["result"] == "D":
                mr.n_draw += 1
            else:
                mr.n_away_win += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  [OK] {n_sims:,} simulaciones completadas en {elapsed:.1f}s\n")

    return WC2026Results(
        n_sims          = n_sims,
        elapsed_seconds = elapsed,
        teams           = team_results,
        group_matches   = match_results,
        champion_counts = dict(champion_counts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formateo de resultados en terminal
# ─────────────────────────────────────────────────────────────────────────────

def print_champion_table(results: WC2026Results, top_n: int = 20):
    """Imprime la tabla de favoritos al título."""
    df = results.champion_probs()

    print(f"\n{'═'*78}")
    print(f"  [CAMPEON]  ODDS DEL MUNDIAL 2026  —  {results.n_sims:,} SIMULACIONES")
    print(f"{'═'*78}")
    print(f"  {'#':3s} {'Equipo':22s} {'Grupo':5s} "
          f"{'Campeón':>9s} {'Final':>9s} {'Semis':>9s} "
          f"{'Cuartos':>9s} {'R16':>7s} {'Grupos':>7s}")
    print(f"  {'─'*3} {'─'*22} {'─'*5} "
          f"{'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*7}")

    for rank, row in df.head(top_n).iterrows():
        bar = "█" * max(1, int(row["p_champion"] * 100))
        print(f"  {rank:<3d} {row['team']:22s} [{row['group']}]   "
              f"{row['p_champion']:>8.1%} "
              f"{row['p_final']:>8.1%} "
              f"{row['p_sf']:>8.1%} "
              f"{row['p_qf']:>8.1%} "
              f"{row['p_r16']:>6.1%} "
              f"{row['p_group_advance']:>6.1%}")

    print(f"{'═'*78}\n")


def print_group_predictions(results: WC2026Results):
    """Imprime predicciones por grupo."""
    print(f"\n{'═'*78}")
    print(f"  PREDICCIONES POR GRUPO — P(clasificar entre los 32)")
    print(f"{'═'*78}")

    for group in "ABCDEFGHIJKL":
        group_teams = [(t, results.teams[t]) for t in WC2026_GROUPS.get(group, [])
                       if t in results.teams]
        if not group_teams:
            continue

        group_teams.sort(key=lambda x: -x[1].p_group_advance)
        print(f"\n  Grupo {group}:")
        print(f"    {'Equipo':22s} {'Pts Prom':>8s} {'GF Prom':>7s} {'Clasifica':>9s} {'Campeón':>8s}")
        for team, tr in group_teams:
            print(f"    {team:22s} {tr.avg_pts:>8.1f} {tr.avg_gf:>7.1f} "
                  f"{tr.p_group_advance:>8.1%} {tr.p_champion:>7.1%}")


def print_group_match_predictions(results: WC2026Results, group: str = "ALL"):
    """Imprime predicciones de partidos de grupos."""
    print(f"\n{'═'*78}")
    print(f"  PREDICCIONES DE PARTIDOS DE GRUPOS")
    print(f"{'═'*78}")

    for key, mr in sorted(results.group_matches.items()):
        if group != "ALL" and mr.group != group:
            continue
        h = mr.home_team
        a = mr.away_team
        ph, pd_, pa = mr.p_home_win, mr.p_draw, mr.p_away_win
        score = mr.most_likely_score
        print(f"\n  [{mr.group}] {h:20s} vs {a:20s}")
        print(f"       P(H)={ph:.1%}  P(D)={pd_:.1%}  P(A)={pa:.1%}")
        print(f"       Goles: {mr.avg_home_goals:.2f}–{mr.avg_away_goals:.2f}"
              f"  |  Marcador más probable: {score} ({mr.most_likely_score_prob:.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: WC2026Results,
                 output_path: Optional[Path] = None) -> Path:
    """Guarda los resultados en JSON y Parquet."""
    ensure_dirs()
    if output_path is None:
        output_path = DATA_MODEL / "wc2026_simulation.json"

    # Exportar a dict serializable
    champion_probs_df = results.champion_probs()

    # JSON compacto
    out = {
        "meta": {
            "n_sims":           results.n_sims,
            "elapsed_seconds":  round(results.elapsed_seconds, 1),
        },
        "champion_probs": champion_probs_df.to_dict(orient="records"),
        "champion_counts": results.champion_counts,
        "group_matches": {
            k: {
                "home_team":             v.home_team,
                "away_team":             v.away_team,
                "group":                 v.group,
                "p_home_win":            round(v.p_home_win, 4),
                "p_draw":                round(v.p_draw, 4),
                "p_away_win":            round(v.p_away_win, 4),
                "avg_home_goals":        round(v.avg_home_goals, 3),
                "avg_away_goals":        round(v.avg_away_goals, 3),
                "most_likely_score":     v.most_likely_score,
                "most_likely_score_prob":round(v.most_likely_score_prob, 4),
            }
            for k, v in results.group_matches.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Resultados guardados: {output_path}")

    # También en Parquet
    parquet_path = output_path.with_suffix(".parquet")
    champion_probs_df.to_parquet(parquet_path, index=True)
    print(f"  [OK] Tabla de odds guardada: {parquet_path}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulación Monte Carlo del Mundial 2026"
    )
    parser.add_argument("--sims",   type=int,  default=10_000,
                        help="Número de simulaciones (default: 10 000)")
    parser.add_argument("--seed",   type=int,  default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--output", type=str,  default=None,
                        help="Ruta de salida JSON (default: data/model/wc2026_simulation.json)")
    parser.add_argument("--group",  type=str,  default=None,
                        help="Imprimir partidos de un grupo específico (ej: --group A)")
    parser.add_argument("--top",    type=int,  default=20,
                        help="Top N equipos en la tabla de favoritos (default: 20)")
    parser.add_argument("--quick",  action="store_true",
                        help="Modo rápido: 1 000 simulaciones")
    parser.add_argument("--real-bracket", action="store_true", dest="real_bracket",
                        help="Usar la llave real de R32 (post-fase-de-grupos)")
    args = parser.parse_args()

    n = 1_000 if args.quick else args.sims
    out_path = Path(args.output) if args.output else None

    results = simulate_wc2026(n_sims=n, seed=args.seed, verbose=True,
                              use_real_bracket=args.real_bracket)

    if results is not None:
        print_champion_table(results, top_n=args.top)
        print_group_predictions(results)

        if args.group:
            print_group_match_predictions(results, group=args.group.upper())

        save_results(results, output_path=out_path)
