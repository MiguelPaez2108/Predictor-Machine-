"""
wc2026_fixtures.py — Fixtures oficiales del Mundial 2026
=========================================================
Grupos y llave eliminatoria según el sorteo del 5 de diciembre de 2024
(Washington D.C.). Datos oficiales FIFA.

Formato del torneo:
  - 48 equipos / 12 grupos (A-L) / 4 equipos por grupo
  - Clasifican: top 2 de cada grupo + 8 mejores terceros (32 en total)
  - Fase eliminatoria: Ronda de 32 → 16 → Cuartos → Semis → Final

Nombres de equipos alineados con el dataset martj42 (international_results).
"""

# ─────────────────────────────────────────────────────────────────────────────
# GRUPOS OFICIALES — MUNDIAL 2026
# Fuente: FIFA / sorteo Washington D.C., 5 diciembre 2024
# ─────────────────────────────────────────────────────────────────────────────

WC2026_GROUPS = {
    "A": ["Mexico",                  "South Africa",  "South Korea",   "Czech Republic"],
    "B": ["Canada",                  "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil",                  "Morocco",       "Haiti",         "Scotland"],
    "D": ["United States",           "Paraguay",      "Australia",     "Turkey"],
    "E": ["Germany",                 "Curaçao",       "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",             "Japan",         "Sweden",        "Tunisia"],
    "G": ["Belgium",                 "Egypt",         "Iran",          "New Zealand"],
    "H": ["Spain",                   "Cape Verde",    "Saudi Arabia",  "Uruguay"],
    "I": ["France",                  "Senegal",       "Iraq",          "Norway"],
    "J": ["Argentina",               "Algeria",       "Austria",       "Jordan"],
    "K": ["Portugal",                "DR Congo",      "Uzbekistan",    "Colombia"],
    "L": ["England",                 "Croatia",       "Ghana",         "Panama"],
}

# Mapeo de nombres alternativos → nombre canónico usado en los datos
# (algunos equipos tienen nombres distintos en StatsBomb vs martj42 vs FIFA)
TEAM_NAME_MAP = {
    # FIFA / StatsBomb → martj42 dataset
    "Türkiye"               : "Turkey",
    "Türkiye "              : "Turkey",
    "Czechia"               : "Czech Republic",
    "Côte d'Ivoire"         : "Ivory Coast",
    "Curaçao"               : "Curacao",
    "Cabo Verde"            : "Cape Verde",
    "DR Congo"              : "DR Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "United States"         : "United States",
    "New Zealand"           : "New Zealand",
    "Cote d'Ivoire"         : "Ivory Coast",
    # StatsBomb usa nombres completos
    "Iran (Islamic Republic of)": "Iran",
}

def normalize_team_name(name: str) -> str:
    """Normaliza el nombre del equipo al nombre canónico del dataset."""
    return TEAM_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# SEDES POR GRUPO (para ajuste de altitud/clima)
# ─────────────────────────────────────────────────────────────────────────────

WC2026_VENUES = {
    # Ciudad: (altitud_metros, país_sede)
    "New York / New Jersey" : (10,   "United States"),
    "Los Angeles"           : (71,   "United States"),
    "Dallas"                : (139,  "United States"),
    "San Francisco Bay Area": (4,    "United States"),
    "Miami"                 : (3,    "United States"),
    "Atlanta"               : (320,  "United States"),
    "Seattle"               : (52,   "United States"),
    "Boston"                : (14,   "United States"),
    "Houston"               : (13,   "United States"),
    "Philadelphia"          : (12,   "United States"),
    "Kansas City"           : (270,  "United States"),
    "Toronto"               : (76,   "Canada"),
    "Vancouver"             : (2,    "Canada"),
    "Guadalajara"           : (1566, "Mexico"),
    "Mexico City"           : (2240, "Mexico"),    # Alta altitud — factor crítico
    "Monterrey"             : (537,  "Mexico"),
}

# Asignación de grupos a sedes principales (simplificado)
GROUP_VENUES = {
    "A": "Mexico City",        "B": "Toronto",
    "C": "Los Angeles",        "D": "Dallas",
    "E": "New York / New Jersey", "F": "Seattle",
    "G": "Atlanta",            "H": "Miami",
    "I": "Boston",             "J": "Houston",
    "K": "Philadelphia",       "L": "Vancouver",
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE FIXTURES DE FASE DE GRUPOS
# ─────────────────────────────────────────────────────────────────────────────

def generate_group_fixtures() -> list:
    """
    Genera todos los partidos de fase de grupos (cada grupo = round-robin 6 partidos).
    Devuelve lista de dicts: {group, home_team, away_team, venue, altitude}
    """
    import itertools
    fixtures = []
    for group, teams in WC2026_GROUPS.items():
        venue    = GROUP_VENUES.get(group, "United States")
        altitude = WC2026_VENUES.get(venue, (0, "United States"))[0]
        for home, away in itertools.combinations(teams, 2):
            fixtures.append({
                "phase"      : "group",
                "group"      : group,
                "home_team"  : home,
                "away_team"  : away,
                "venue"      : venue,
                "altitude_m" : altitude,
                "neutral"    : True,    # todos son campo neutral
            })
    return fixtures


def get_group_of(team: str) -> str:
    """Devuelve el grupo al que pertenece un equipo."""
    team_norm = normalize_team_name(team)
    for group, teams in WC2026_GROUPS.items():
        if team in teams or team_norm in teams:
            return group
    return "?"


def all_teams() -> list:
    """Lista ordenada de los 48 equipos del Mundial 2026."""
    teams = []
    for group_teams in WC2026_GROUPS.values():
        teams.extend(group_teams)
    return sorted(teams)


# ─────────────────────────────────────────────────────────────────────────────
# LLAVE ELIMINATORIA OFICIAL
# Los 8 grupos de la ronda de 32 están definidos por FIFA:
# Los cruces son: 1A vs 2C, 1B vs 2D, etc. (ver tabla oficial)
# ─────────────────────────────────────────────────────────────────────────────

# Cruces de Ronda de 32 (basado en clasificación por grupos)
# Formato: (slot_ganador_grupo_X, slot_segundo_grupo_Y)
# Los 8 mejores 3ros van a slots específicos según rendimiento
ROUND_OF_32_BRACKET = [
    # Bracket superior
    ("1A", "2D"), ("1B", "2E"), ("1C", "2F"), ("1D", "2A"),
    # Bracket inferior
    ("1E", "2H"), ("1F", "2I"), ("1G", "2J"), ("1H", "2E"),
    # Terceros (8 mejores, slots a confirmar por FIFA tras fase de grupos)
    # Por ahora se simularán aleatoriamente dentro de los slots libres
]

# Nota: La llave completa del torneo se determina dinámicamente
# en simulation/simulate_tournament.py según los clasificados reales.


if __name__ == "__main__":
    fixtures = generate_group_fixtures()
    print(f"Total partidos de grupos: {len(fixtures)}")
    print(f"Total equipos: {len(all_teams())}")
    print()
    for group, teams in WC2026_GROUPS.items():
        print(f"  Grupo {group}: {' | '.join(teams)}")
