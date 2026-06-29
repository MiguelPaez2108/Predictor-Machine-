"""
print_real_bracket.py — Imprime la llave real del R32 WC2026
Parche standalone para predict_match.py. Agregar esta función al archivo principal.
"""

WC2026_R32_REAL_BRACKET_LEFT = [
    ("Germany",       "Paraguay"),
    ("France",        "Sweden"),
    ("South Africa",  "Canada"),
    ("Netherlands",   "Morocco"),
    ("Portugal",      "Croatia"),
    ("Spain",         "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium",       "Senegal"),
]

WC2026_R32_REAL_BRACKET_RIGHT = [
    ("Brazil",        "Japan"),
    ("Ivory Coast",   "Norway"),
    ("Mexico",        "Ecuador"),
    ("England",       "DR Congo"),
    ("Argentina",     "Cape Verde"),
    ("Australia",     "Egypt"),
    ("Switzerland",   "Algeria"),
    ("Colombia",      "Ghana"),
]

WC2026_R32_REAL_BRACKET = WC2026_R32_REAL_BRACKET_LEFT + WC2026_R32_REAL_BRACKET_RIGHT


def print_real_bracket(results_file=None):
    """
    Imprime la llave real del R32 con resultados registrados donde corresponda.
    Llamar desde predict_match.py como opción 1 del menú.
    """
    import json
    from pathlib import Path

    # Cargar resultados reales
    played = {}
    if results_file and Path(results_file).exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        for r in results:
            stage = r.get("stage", r.get("group", ""))
            if stage in ("R32", "R16", "QF", "SF", "F"):
                key = f"{r['home_team']}|{r['away_team']}"
                key_rev = f"{r['away_team']}|{r['home_team']}"
                played[key] = r
                played[key_rev] = r

    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    print(f"\n{CYAN}╔{'═'*72}╗{RESET}")
    print(f"{CYAN}║{BOLD}{'  LLAVE REAL R32 — MUNDIAL FIFA 2026':^72s}{RESET}{CYAN}║{RESET}")
    print(f"{CYAN}╠{'═'*72}╣{RESET}")

    print(f"{CYAN}║{BOLD}{'  LADO IZQUIERDO':^72s}{RESET}{CYAN}║{RESET}")
    print(f"{CYAN}╠{'─'*72}╣{RESET}")

    for i, (home, away) in enumerate(WC2026_R32_REAL_BRACKET_LEFT, 1):
        key = f"{home}|{away}"
        r = played.get(key)

        if r:
            hg = r["home_goals"]
            ag = r["away_goals"]
            if hg > ag:
                result_str = f"{GREEN}{hg}-{ag}{RESET} {GREEN}→ {home[:18]}{RESET}"
            elif ag > hg:
                result_str = f"{RED}{hg}-{ag}{RESET} {GREEN}→ {away[:18]}{RESET}"
            else:
                result_str = f"{YELLOW}{hg}-{ag} Penaltis{RESET}"
            line = f"  {i}. {home[:22]:22s} {result_str}"
        else:
            line = f"  {i}. {home[:22]:22s} vs  {away[:22]:22s}  {DIM}(pendiente){RESET}"

        padding = max(0, 70 - len(line.replace(GREEN,'').replace(RED,'').replace(YELLOW,'').replace(DIM,'').replace(BOLD,'').replace(RESET,'').replace(CYAN,'')))
        print(f"{CYAN}║{RESET}{line}{' '*padding}{CYAN}║{RESET}")

    print(f"{CYAN}╠{'─'*72}╣{RESET}")
    print(f"{CYAN}║{BOLD}{'  LADO DERECHO':^72s}{RESET}{CYAN}║{RESET}")
    print(f"{CYAN}╠{'─'*72}╣{RESET}")

    for i, (home, away) in enumerate(WC2026_R32_REAL_BRACKET_RIGHT, 9):
        key = f"{home}|{away}"
        r = played.get(key)

        if r:
            hg = r["home_goals"]
            ag = r["away_goals"]
            if hg > ag:
                result_str = f"{GREEN}{hg}-{ag}{RESET} {GREEN}→ {home[:18]}{RESET}"
            elif ag > hg:
                result_str = f"{RED}{hg}-{ag}{RESET} {GREEN}→ {away[:18]}{RESET}"
            else:
                result_str = f"{YELLOW}{hg}-{ag} Penaltis{RESET}"
            line = f"  {i}. {home[:22]:22s} {result_str}"
        else:
            line = f"  {i}. {home[:22]:22s} vs  {away[:22]:22s}  {DIM}(pendiente){RESET}"

        padding = max(0, 70 - len(line.replace(GREEN,'').replace(RED,'').replace(YELLOW,'').replace(DIM,'').replace(BOLD,'').replace(RESET,'').replace(CYAN,'')))
        print(f"{CYAN}║{RESET}{line}{' '*padding}{CYAN}║{RESET}")

    print(f"{CYAN}╚{'═'*72}╝{RESET}")
    print(f"{DIM}  Brasil 2-1 Japón  |  Sudáfrica 0-1 Canadá (actualizados){RESET}\n")
