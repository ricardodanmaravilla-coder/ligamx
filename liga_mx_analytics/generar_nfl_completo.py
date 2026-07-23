import requests
import pandas as pd
import os

API_KEY = "1abc53997c1b26e3b447796665e36e44"
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
NFL_LEAGUE_ID = 1

# Ampliamos el rango desde 2021 hasta 2025 (2025 incluye principios de 2026)
TEMPORADAS = [2021, 2022, 2023, 2024, 2025]
archivo_csv = 'data/historico_nfl.csv'
todos_los_partidos = []

for temporada in TEMPORADAS:
    print(f"Descargando y extrayendo temporada NFL {temporada}...")
    res = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={"league": NFL_LEAGUE_ID, "season": temporada})
    
    if res.status_code == 200:
        data = res.json().get("response", [])
        print(f" -> Procesando {len(data)} registros...")
        
        for g in data:
            # Extracción correcta de la fecha anidada
            game_obj = g.get("game", {})
            date_dict = game_obj.get("date", {})
            fecha = date_dict.get("date", "2021-01-01")
            
            # Extraer equipos
            teams = g.get("teams", {})
            local = teams.get("home", {}).get("name", "Desconocido")
            visita = teams.get("away", {}).get("name", "Desconocido")
            
            # Extraer puntajes
            scores = g.get("scores", {})
            home_scores = scores.get("home", {})
            away_scores = scores.get("away", {})
            
            pts_local = home_scores.get("total")
            pts_visita = away_scores.get("total")
            
            if pts_local is None:
                pts_local = sum([home_scores.get(q, 0) or 0 for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]])
            if pts_visita is None:
                pts_visita = sum([away_scores.get(q, 0) or 0 for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]])
                
            if pts_local is not None and pts_visita is not None and (pts_local > 0 or pts_visita > 0):
                todos_los_partidos.append({
                    "Fecha": fecha,
                    "Local": local,
                    "Visitante": visita,
                    "Puntos_L": int(pts_local),
                    "Puntos_V": int(pts_visita),
                    "Total_Puntos": int(pts_local) + int(pts_visita),
                    "Diferencia": int(pts_local) - int(pts_visita)
                })

if todos_los_partidos:
    df_final = pd.DataFrame(todos_los_partidos)
    df_final.to_csv(archivo_csv, index=False)
    print(f"\n¡ÉXITO TOTAL! Archivo '{archivo_csv}' actualizado con {len(df_final)} partidos desde 2021 hasta 2026.")
    print("\nÚltimos registros guardados (principios de 2026):")
    print(df_final[['Fecha', 'Local', 'Visitante', 'Total_Puntos']].tail(3))
else:
    print("\nNo se recolectaron partidos.")