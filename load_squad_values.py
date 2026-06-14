"""
Carga EXACTAMENTE los valores del archivo valor.txt buscados uno a uno por el usuario.
Sin añadir ni quitar nada.
"""
import sys, math
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SQUAD_VALUES, ensure_dirs

# EXACTAMENTE lo que está en valor.txt — sin modificaciones
# (nombre_ingles, valor_en_euros)
VALORES_REALES = [
    ("France",              1_520_000_000),  # Francia        - 1,52 mil mill.
    ("England",             1_360_000_000),  # Inglaterra     - 1,36 mil mill.
    ("Spain",               1_220_000_000),  # España         - 1,22 mil mill.
    ("Portugal",            1_010_000_000),  # Portugal       - 1,01 mil mill.
    ("Germany",               947_000_000),  # Alemania       - 947,00 mill.
    ("Brazil",                928_200_000),  # Brasil         - 928,20 mill.
    ("Argentina",             807_500_000),  # Argentina      - 807,50 mill.
    ("Netherlands",           754_200_000),  # Países Bajos   - 754,20 mill.
    ("Norway",                589_900_000),  # Noruega        - 589,90 mill.
    ("Belgium",               547_500_000),  # Bélgica        - 547,50 mill.
    ("Ivory Coast",           522_100_000),  # Costa de Marfil- 522,10 mill.
    ("Senegal",               478_100_000),  # Senegal        - 478,10 mill.
    ("Turkey",                473_700_000),  # Turquía        - 473,70 mill.
    ("Morocco",               447_700_000),  # Marruecos      - 447,70 mill.
    ("Sweden",                406_080_000),  # Suecia         - 406,08 mill.
    ("Croatia",               387_300_000),  # Croacia        - 387,30 mill.
    ("United States",         385_650_000),  # Estados Unidos - 385,65 mill.
    ("Ecuador",               368_700_000),  # Ecuador        - 368,70 mill.
    ("Uruguay",               359_300_000),  # Uruguay        - 359,30 mill.
    ("Switzerland",           332_500_000),  # Suiza          - 332,50 mill.
    ("Colombia",              302_350_000),  # Colombia       - 302,35 mill.
    ("Japan",                 270_850_000),  # Japón          - 270,85 mill.
    ("Algeria",               256_900_000),  # Argelia        - 256,90 mill.
    ("Ghana",                 234_600_000),  # Ghana          - 234,60 mill.
    ("Austria",               245_200_000),  # Austria        - 245,20 mill.
    ("Canada",                198_650_000),  # Canadá         - 198,65 mill.
    ("Mexico",                191_850_000),  # México         - 191,85 mill.
    ("Czech Republic",        188_180_000),  # Rep. Checa     - 188,18 mill.
    ("Scotland",              170_250_000),  # Escocia        - 170,25 mill.
    ("Bosnia and Herzegovina",146_400_000),  # Bosnia         - 146,40 mill.
    ("Congo DR",              143_900_000),  # Rep. del Congo - 143,90 mill.
    ("Paraguay",              153_650_000),  # Paraguay       - 153,65 mill.
    ("South Korea",           139_050_000),  # Corea del Sur  - 139,05 mill.
    ("Egypt",                 116_480_000),  # Egipto         - 116,48 mill.
    ("Uzbekistan",             85_330_000),  # Uzbekistán     - 85,33 mill.
    ("Australia",              77_450_000),  # Australia      - 77,45 mill.
    ("Tunisia",                69_950_000),  # Túnez          - 69,95 mill.
    ("Haiti",                  55_900_000),  # Haití          - 55,90 mill.
    ("Cape Verde",             54_500_000),  # Cabo Verde     - 54,50 mill.
    ("South Africa",           49_250_000),  # Sudáfrica      - 49,25 mill.
    ("Saudi Arabia",           40_680_000),  # Arabia Saudí   - 40,68 mill.
    ("New Zealand",            34_350_000),  # Nueva Zelanda  - 34,35 mill.
    ("Panama",                 34_550_000),  # Panamá         - 34,55 mill.
    ("Iran",                   32_050_000),  # Irán           - 32,05 mill.
    ("Curacao",                25_780_000),  # Curazao        - 25,78 mill.
    ("Iraq",                   21_200_000),  # Irak           - 21,20 mill.
    ("Jordan",                 20_300_000),  # Jordania       - 20,30 mill.
    ("Qatar",                  19_930_000),  # Catar          - 19,93 mill.
]

ensure_dirs()

rows = []
for team, value in VALORES_REALES:
    rows.append({
        "team":            team,
        "squad_value_eur": value,
        "squad_value_log": round(math.log(value), 4),
        "source":          "transfermarkt_manual_june2026",
    })

df = pd.DataFrame(rows)
df.to_parquet(SQUAD_VALUES, index=False)

print(f"Guardado: {SQUAD_VALUES}")
print(f"Equipos: {len(df)}\n")
for i, row in df.iterrows():
    print(f"  {i+1:2}. {row['team']:30} {row['squad_value_eur']/1e6:>8.2f}M EUR")
