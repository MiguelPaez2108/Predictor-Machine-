## **Datogami - Guía statsbombpy** 

`pip install statsbombpy mplsoccer` 

## **1. Estructura de datos** 

StatsBomborganizasusdatos en cuatro niveles jerarquicos. Siempre se accede en este orden: primero se obtienen las competiciones disponibles, luego los partidos de esa competicion, y finalmente los eventos de cada partido. 

## **competitions()** 

Devuelve todas las competiciones y temporadas con datos disponibles. Es el punto de partida: aqui se obtienen los IDs necesarios para las siguientes llamadas. 

`from statsbombpy import sb` 

`comps = sb.competitions()` 

`print(comps[['competition_id', 'competition_name', 'season_id', 'season_name']])` 

_**`competition_id`**_ 

_**`season_id`**_ 

_**`competition_name`**_ 

int.Identificadorunicode la int.Identificador de la temporada competicion Referencia util: FIFA World Cup 2022 = competition_id `43` /season_id `106` 

str. Nombre de la competicion 

_**`season_name`**_ 

str.Nombredelatemporada (ej. "2022") 

## **matches(competition_id, season_id)** 

Devuelve todos los partidos de una competicion y temporada concretas. El campo match_id es la clave para descargar los eventos de cada partido. 

`matches = sb.matches(competition_id=43, season_id=106) # Encontrar la final por fecha matches.sort_values('match_date').tail(1)` 

_**`match_id`**_ 

int.Clavepara llamar a events() 

_**`competition_stage`**_ str. Grupo, Octavos, Final... 

_**`match_date`**_ date. Fecha del partido 

_**`stadium`**_ str.Nombredel estadio 

_**`home_team / away_team`**_ str. Nombres de los equipos _**`referee`**_ str.Nombredel arbitro 

_**`home_score / away_score`**_ int.Resultado _**`match_week`**_ int. Jornada 

## **events(match_id)** 

El nucleo de StatsBomb. Devuelve todas las acciones de un partido: unos 3.000 eventos por partido, cada uno con coordenadas de campo, tiempo, jugador, equipo y docenas de atributos especificos segun el tipo de accion. El campo de juego mide 120 x 80 unidades, con el origen en la esquina inferior izquierda del equipo local. 

## `events = sb.events(match_id=3869685)` 

`# Tipos de eventos disponibles events['type'].value_counts()` 

`# Extraer coordenadas x, y (location es una lista [x, y]) events['x'] = events['location'].apply(lambda l: l[0]) events['y'] = events['location'].apply(lambda l: l[1])` 

## **Tipos de eventos principales** 

_**`Pass`**_ 

Pasescompletados e incompletos 

_**`Pressure`**_ 

Presion al portador 

_**`Shot`**_ 

Tiros.Incluye xG y resultado 

_**`Dribble`**_ 

Regates intentados 

_**`Carry`**_ 

Conduccion del balon 

_**`Duel`**_ 

Duelosaereos y en disputa 

_**`Ball Recovery`**_ Recuperaciones 

_**`Clearance`**_ Despejes defensivos 

_**`Goal Keeper`**_ Acciones del portero 

_**`Interception`**_ Interceptaciones 

## _**`Block`**_ 

Bloqueos de tiro o pase 

_**`Foul Committed`**_ Faltascometidas 

## **lineups(match_id)** 

Devuelve un diccionario con los jugadores de cada equipo, su dorsal, posiciones y tarjetas recibidas. 

`lineups = sb.lineups(match_id=3869685)` 

`# Devuelve dict: {nombre_equipo: DataFrame} arg = lineups['Argentina']` 

_**`player_name`**_ str. Nombre completo 

_**`jersey_number positions cards`**_ int.Dorsal lista.Posiciones en el partido lista.Tarjetas recibidas 

## **frames(match_id)** 

Datos de seguimiento 360 grados. Devuelve la posicion de todos los jugadores en el campo para cada evento del partido. Solo disponible en algunas competiciones: La Liga 2020/21, Women's Euro 2022 y una seleccion de partidos del Mundial 2022. 

`# Comprobar disponibilidad antes de llamar` 

`matches['match_available_360']` 

`frames = sb.frames(match_id=3869685)` 

## **2. Filtros utiles** a, 

## **Filtrar por jugador** 

Los nombres de jugadores en StatsBomb incluyen tildes y nombres completos. Conviene consultarlos antes de filtrar. 

`# Ver todos los jugadores del partido events['player'].dropna().unique() # Filtrar por jugador messi = events[events['player'] == 'Lionel Andres Messi Cuccittini']` 

## **Filtrar por tipo de evento y atributos** 

`shots = events[events['type'] == 'Shot'] passes = events[events['type'] == 'Pass'] carries = events[events['type'] == 'Carry'] # Solo goles goals = shots[shots['shot_outcome'] == 'Goal'] # Pases completados (outcome NaN significa completado) completed = passes[passes['pass_outcome'].isna()] # Primer tiempo / acciones bajo presion first_half = events[events['period'] == 1] under_press = events[events['under_pressure'] == True]` 

## **Estadisticas rapidas por jugador** 

`# xG total por jugador en el partido xg_ranking = ( events[events['type'] == 'Shot'] .groupby('player')['shot_statsbomb_xg'].sum() .sort_values(ascending=False)` 

`)` 

`# Numero de pases por jugador` 

## `pases = passes.groupby('player')['id'].count()` 

## **Iterar todos los partidos de un torneo** 

Para analizar una competicion completa se itera sobre todos los match_id y se concatenan los DataFrames. Guardar en Parquet evita volver a descargar. 

`import pandas as pd all_events = [] for mid in matches['match_id']: ev = sb.events(match_id=mid) all_events.append(ev)` 

`df_wc = pd.concat(all_events, ignore_index=True)` 

`# Guardar para reutilizar sin volver a descargar df_wc.to_parquet('wc2022.parquet')` 

`df_wc = pd.read_parquet('wc2022.parquet') # carga rapida` 

## **3. Columnas clave** 

Cada evento tiene columnas universales (presentes siempre) y columnas especificas del tipo de accion (solo se rellenan cuando el evento es de ese tipo; el resto aparece como NaN). 

## **Columnas universales** 

## _**`type`**_ 

str.Tipo de evento: Pass, Shot, Carry... 

## _**`location`**_ 

list.[x,y]enel campo (120x80) 

## _**`minute / second`**_ 

int. Tiempo del evento 

## _**`under_pressure`**_ 

bool. El jugador esta siendo presionado 

## _**`period`**_ 

int.1=primer tiempo, 2=segundo, 3/4=prorroga, 5=penaltis 

## _**`play_pattern`**_ 

str. Origendelaposesion: From GK, Free Kick... 

## _**`player / team`**_ 

str.Jugadoryequipoque ejecuta la accion 

## _**`duration`**_ 

float.Duracion de la accion en segundos 

## **Columnas de Shot** 

## _**`shot_statsbomb_xg`**_ 

float0-1.Probabilidaddegol segun el modelo de SB 

## _**`shot_outcome`**_ 

str.Goal / Saved / Off T / Blocked / Wayward / Post 

## _**`shot_body_part`**_ 

str.RightFoot / LeftFoot / Head 

## _**`shot_technique`**_ 

str.Normal / Volley / Half Volley / Lob / Backheel 

## _**`shot_first_time`**_ 

bool. El tiro es a primera 

## _**`shot_type`**_ 

str.OpenPlay / Free Kick / Penalty / Corner 

## _**`shot_one_on_one`**_ 

bool.Manoamanoconel portero 

## _**`shot_end_location`**_ 

list.[x,y,z]posiciondondetermina el balon 

## _**`shot_freeze_frame`**_ 

list.Posiciondetodoslosjugadores (requiere 360) 

## **Columnas de Pass** 

## _**`pass_end_location`**_ 

list. [x, y] destino del pase 

## _**`pass_angle`**_ 

float.Angulodelpase en radianes 

## _**`pass_height`**_ 

str.Ground / Low / High 

## _**`pass_recipient`**_ 

str.Nombredeljugador que recibe 

## _**`pass_outcome`**_ 

str.NaN=completado. Incomplete / Out si falla 

## _**`pass_goal_assist`**_ 

bool.Elpasegeneraungol directamente 

## _**`pass_length`**_ 

float.Longituddelpase en metros 

## _**`pass_type`**_ 

str.FreeKick / Corner / Throw-in / Kick Off 

## _**`pass_shot_assist`**_ 

bool. El paseprecede untiro 

_**`dribble_outcome`**_ str.Complete / Incomplete 

## **Otras columnas de accion** 

## _**`carry_end_location`**_ 

list. [x, y] donde termina la conduccion 

## _**`duel_type`**_ 

str.Tackle / Aerial Lost 

## _**`duel_outcome`**_ 

str.Won / Lost / Success / Fail 

## _**`dribble_nutmeg`**_ 

bool. El regate es un caño 

## _**`goalkeeper_type`**_ 

str.ShotSaved / Punch / Claim... 

## **4. Visualizacion con mplsoccer** 

mplsoccer eslalibreria de referenciaparadibujar campos de futbol y trazar eventos encima. Se integra directamente con los datos de StatsBomb y con matplotlib. 

## **Heatmap con KDE** 

Muestra la densidad de acciones de un jugador sobre el campo mediante una estimacion de densidad de kernel gaussiana. bw_adjust controla el suavizado: valores altos producen manchas mas difusas. 

## `from mplsoccer import Pitch` 

`pitch = Pitch(pitch_type='statsbomb', pitch_color='#0D1117', line_color='#2A3145') fig, ax = pitch.draw(figsize=(10, 7))` 

`pitch.kdeplot(df['x'], df['y'], ax=ax, cmap='YlOrRd', fill=True, levels=200, alpha=0.8, bw_adjust=0.6)` 

## **Shot map con xG** 

Representa cada disparo como una burbuja cuyo tamano es proporcional al xG. VerticalPitch con half=True muestra solo el medio campo atacante. 

`from mplsoccer import VerticalPitch` 

`pitch = VerticalPitch(pitch_type='statsbomb', half=True) fig, ax = pitch.draw()` 

`pitch.scatter(shots['x'], shots['y'],` 

`ax=ax,` 

`s = 300 + shots['shot_statsbomb_xg'] * 1500,` 

`c = shots['shot_outcome'].map({'Goal': '#06D6A0'}).fillna('#E84855'), alpha=0.85, zorder=4)` 

## **Tipos de campo disponibles** 

Siempre usa statsbomb cuando trabajes con datos de StatsBomb para que las coordenadas encajen exactamente. 

|**_'statsbomb'_**|**_'opta'_**|**_'tracab'_**|**_'wyscout'_**|**_'uefa'_**|
|---|---|---|---|---|
|120x80.Recomendado con|100x100|metros reales|100x100|105x68|
|datos SB|||||



## **5. ScraperFC para el Mundial 2026** 

Cuando StatsBombno tengadatos de unacompeticion (como el Mundial 2026 en tiempo real), ScraperFC permite hacer scraping desde FBref, Understat y otras fuentes sin necesidad de API de pago. 

`# pip install scraperfc` 

`from scraperfc import FBref` 

`fbref = FBref()` 

`# Stats de jugadores del torneo players = fbref.scrape_player_season_stats( competition='FIFA World Cup', season='2025-2026', stat_category='summary' # shooting, passing, defense...` 

`)` 

Datos: StatsBomb Open Data github.com/statsbomb/statsbombpy 

# Sigue Datogami Para Más! 

