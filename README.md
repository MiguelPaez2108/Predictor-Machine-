#  WC 2026 Predictor — Sistema de Predicción del Mundial

Sistema de predicción cuantitativa de alto rendimiento para la **Copa del Mundo FIFA 2026**, construido sobre un ensemble de 3 capas de modelos matemáticos y simulación Monte Carlo con 100 000 iteraciones.

---

## Arquitectura del Sistema

```
Datos Crudos (martj42, Elo, FIFA, StatsBomb)
        ↓
   Feature Engineering
        ↓
  ┌─────────────────────────────────────────┐
  │           CAPA 1 — Modelos Base         │
  │  Dixon-Coles │ Elo+Logistic │ RF │ Bay  │
  └─────────────────────────────────────────┘
        ↓ OOF predictions
  ┌─────────────────────────────────────────┐
  │     CAPA 2 — Meta-Modelo XGBoost        │
  │      (aprende a combinar Capa 1)        │
  └─────────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────────┐
  │      CAPA 3 — Calibración Isotónica     │
  └─────────────────────────────────────────┘
        ↓
  Simulación Monte Carlo (100 000 runs)
        ↓
  Odds de campeón, marcadores, clasificación
```

### Capa 1 — Modelos Base

| Modelo | Descripción | Output |
|--------|-------------|--------|
| **Dixon-Coles** | Poisson bivariada con corrección τ para marcadores bajos | λ_home, λ_away, P(1X2), Over/Under, BTTS |
| **Elo + Logistic** | Regresión logística sobre diferencial Elo y features tabulares | P(H/D/A) |
| **Random Forest** | Ensemble de 500 árboles con calibración isotónica | P(H/D/A) |
| **Bayesian MAP** | Poisson con prior regularizador Gamma | λ_home, λ_away, P(1X2) |

### Capa 2 — Meta-Modelo XGBoost
XGBoost entrenado con predicciones pseudo-OOF de los 4 modelos base + features originales (delta_elo, delta_xg, delta_form...) para aprender el peso óptimo de cada modelo.

### Capa 3 — Calibración Isotónica
Calibra las probabilidades finales para que sean bien calibradas empíricamente (Brier Score optimizado).

---

## Estructura del Proyecto

```
wc_predictor/
├── config.py                    # Rutas y parámetros globales
├── requirements.txt             # Dependencias
│
├── ingestion/                   # Descarga y normalización de datos
│   ├── international_results.py
│   ├── elo_ratings.py
│   ├── fifa_rankings.py
│   ├── squad_values.py
│   └── statsbomb_loader.py
│
├── features/                    # Ingeniería de features
│   ├── build_master_features.py # Orquestador principal
│   ├── build_elo_features.py
│   ├── build_form_features.py
│   ├── build_h2h_features.py
│   ├── build_xg_features.py
│   └── build_context_features.py
│
├── models/                      # Modelos matemáticos
│   ├── ensemble.py              # Orquestador del ensemble
│   ├── dixon_coles.py           # Poisson bivariada DC
│   ├── elo_logistic.py          # Elo + Logistic Regression
│   ├── random_forest.py         # Random Forest calibrado
│   ├── bayesian_model.py        # Bayesian Poisson MAP
│   ├── meta_model.py            # XGBoost meta-learner
│   └── calibration.py           # Calibración isotónica
│
├── simulation/                  # Simulación Monte Carlo
│   ├── tournament.py            # Motor MC principal
│   ├── group_stage.py           # Fase de grupos (12 grupos)
│   ├── knockout_stage.py        # Fase eliminatoria (R32→Final)
│   ├── match_simulator.py       # Simulador estocástico de partidos
│   └── wc2026_fixtures.py       # Grupos y llave oficial FIFA 2026
│
├── evaluation/                  # Evaluación y métricas
│
└── data/                        # Datos (no incluidos en el repo)
    ├── raw/                     # Datos crudos
    ├── features/                # Features procesadas
    └── model/                   # Modelos entrenados (.pkl)
```

---

## Instalación y Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno (opcional)

```bash
# .env
SUPABASE_URL=tu_url
SUPABASE_KEY=tu_key
```

### 3. Descargar datos

```bash
# Resultados internacionales (martj42/international-football-results)
python ingestion/international_results.py

# Ratings Elo históricos
python ingestion/elo_ratings.py

# Rankings FIFA
python ingestion/fifa_rankings.py

# Valores de plantilla (Transfermarkt)
python ingestion/squad_values.py

# Datos StatsBomb Open Data (xG, pases, etc.)
python ingestion/statsbomb_loader.py
```

### 4. Construir features

```bash
python features/build_master_features.py
```

### 5. Entrenar el ensemble completo

```bash
python models/ensemble.py --train
```

### 6. Predecir un partido

```bash
python models/ensemble.py --home "Argentina" --away "France"
```

### 7. Simular el Mundial 2026 (Monte Carlo)

```bash
# Simulación rápida (1 000 runs)
python simulation/tournament.py --quick

# Simulación completa (100 000 runs)
python simulation/tournament.py --sims 100000

# Ver predicciones de un grupo específico
python simulation/tournament.py --quick --group C

# Guardar resultados
python simulation/tournament.py --sims 50000 --output results/wc2026.json
```

---

## Features del Modelo

| Feature | Descripción | Fuente |
|---------|-------------|--------|
| `delta_elo` | Diferencia de rating Elo | ClubElo / calculado |
| `expected_home_elo` | Probabilidad de victoria Elo pura | Fórmula Elo |
| `delta_form` | Diferencia de forma (últimos 10 partidos, decay 0.85) | martj42 |
| `delta_xg` | Diferencia de xG promedio | StatsBomb |
| `delta_xga` | Diferencia de xG concedidos | StatsBomb |
| `delta_fifa_rank` | Diferencia de ranking FIFA | FIFA |
| `delta_sv_log` | Diferencia de valor de plantilla (log) | Transfermarkt |
| `delta_rest` | Diferencia de días de descanso | martj42 |
| `h2h_win_rate` | Tasa de victorias H2H | martj42 |
| `h2h_gd_avg` | Diferencia de goles H2H media | martj42 |

---

## Grupos del Mundial 2026

```
Grupo A: México, Sudáfrica, Corea del Sur, Rep. Checa
Grupo B: Canadá, Bosnia y Herzegovina, Qatar, Suiza
Grupo C: Brasil, Marruecos, Haití, Escocia
Grupo D: Estados Unidos, Paraguay, Australia, Turquía
Grupo E: Alemania, Curazao, Costa de Marfil, Ecuador
Grupo F: Países Bajos, Japón, Suecia, Túnez
Grupo G: Bélgica, Egipto, Irán, Nueva Zelanda
Grupo H: España, Cabo Verde, Arabia Saudita, Uruguay
Grupo I: Francia, Senegal, Irak, Noruega
Grupo J: Argentina, Argelia, Austria, Jordania
Grupo K: Portugal, DR Congo, Uzbekistán, Colombia
Grupo L: Inglaterra, Croacia, Ghana, Panamá
```

---

## Métricas del Modelo (Validation Set 2021-2024)

| Modelo | Log Loss | Brier Score | Accuracy |
|--------|----------|-------------|----------|
| Random Forest | 0.8767 | 0.1715 | 60.5% |
| Elo + Logistic | ~0.88 | ~0.172 | ~59% |
| Dixon-Coles | ~0.90 | ~0.175 | ~58% |
| **Ensemble Calibrado** | **~0.86** | **~0.169** | **~61%** |

> Nota: En fútbol, una accuracy del 55-62% es considerada de alta calidad para predicción de resultados 3-way (H/D/A).

---

## Referencias

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics.
- Maher, M.J. (1982). *Modelling association football scores*. Statistica Neerlandica.
- StatsBomb Open Data: [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
- martj42 International Football Results: [github.com/martj42/international-football-results](https://github.com/martj42/international-football-results)

---

## Autor

**Miguel Paez** — Sistema de predicción cuantitativa FIFA World Cup 2026
