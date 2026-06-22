"""
02_download_statsbomb.py
Descarga eventos + datos 360 de StatsBomb Open Data sin autenticación.
Fuente: statsbombpy (pip install statsbombpy)
Salida: data/raw/statsbomb/{torneo}/events_{match_id}.parquet
        data/raw/statsbomb/{torneo}/frames_{match_id}.parquet (360°)
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STATSBOMB_COMPS, ensure_dirs

def download_competition(comp_id, season_id, out_dir):
    from statsbombpy import sb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCompetición {comp_id} / Temporada {season_id} → {out_dir.name}")

    # Obtener lista de partidos
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    matches.to_parquet(out_dir / "matches.parquet", index=False)
    print(f"  Partidos encontrados: {len(matches)}")

    events_list  = []
    has_360 = False

    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        print(f"  [{mid}] {row['home_team']} vs {row['away_team']}...", end=" ")

        # Eventos del partido
        evts = sb.events(match_id=mid, flatten_attrs=True)
        evts["match_id"] = mid
        evts.to_parquet(out_dir / f"events_{mid}.parquet", index=False)

        events_list.append({
            "match_id":    mid,
            "home_team":   row["home_team"],
            "away_team":   row["away_team"],
            "home_score":  row["home_score"],
            "away_score":  row["away_score"],
            "n_events":    len(evts),
            "has_360":     row.get("match_available_360") is not None,
        })

        # Datos 360° si están disponibles
        if row.get("match_available_360"):
            has_360 = True
            try:
                frames = sb.three_sixty(match_id=mid)
                frames["match_id"] = mid
                frames.to_parquet(out_dir / f"frames_{mid}.parquet", index=False)
                print("[OK] 360°")
            except Exception as e:
                print(f"[AVISO] 360 no disponible: {e}")
        else:
            print("[OK]")

    # Índice de la competición
    idx = pd.DataFrame(events_list)
    idx.to_parquet(out_dir / "index.parquet", index=False)

    print(f"  Total eventos: {idx['n_events'].sum():,}")
    print(f"  Con datos 360°: {idx['has_360'].sum()}/{len(idx)}")
    return idx


def download_all():
    ensure_dirs()
    summary = []

    for comp_id, season_id, out_dir in STATSBOMB_COMPS:
        try:
            idx = download_competition(comp_id, season_id, out_dir)
            summary.append({
                "comp_id":   comp_id,
                "season_id": season_id,
                "folder":    Path(out_dir).name,
                "matches":   len(idx),
                "events":    idx["n_events"].sum(),
                "has_360":   idx["has_360"].sum(),
            })
        except Exception as e:
            print(f"  ERROR en comp {comp_id}/{season_id}: {e}")

    print("\n── RESUMEN ──────────────────────────────────")
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    print(f"\nTotal partidos: {df_sum['matches'].sum()}")
    print(f"Total eventos:  {df_sum['events'].sum():,}")


if __name__ == "__main__":
    download_all()
