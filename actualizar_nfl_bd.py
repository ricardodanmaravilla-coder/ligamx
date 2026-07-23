import requests
import pandas as pd
import os
import time

# --- CONFIGURACIÓN API NFL ---
API_KEY = "1abc53997c1b26e3b447796665e36e44" 
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
NFL_LEAGUE_ID = 1  # ID 1 para la NFL confirmado
# Temporadas válidas de la NFL en API-Sports (ej. 2023, 2024, 2025)
TEMPORADAS_NFL = [2023, 2024, 2025] 
ARCHIVO_CSV_NFL = 'data/historico_nfl.csv'

def actualizar_base_datos_nfl():
    print("🏈 Iniciando descarga y actualización del histórico de la NFL...")
    
    # 1. Cargar base existente para evitar duplicados
    if os.path.exists(ARCHIVO_CSV_NFL):
        df_existente = pd.read_csv(ARCHIVO_CSV_NFL)
        partidos_guardados = set(df_existente['Fecha'].astype(str).str[:10] + "_" + df_existente['Local'])
    else:
        df_existente = pd.DataFrame()
        partidos_guardados = set()
        os.makedirs('data', exist_ok=True)

    nuevos_partidos = []

    # 2. Iterar por temporadas
    for temporada in TEMPORADAS_NFL:
        print(f"Buscando partidos de la NFL de la temporada {temporada}...")
        res = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={"league": NFL_LEAGUE_ID, "season": temporada})
        
        if res.status_code != 200:
            print(f"Error al conectar con la API para la temporada NFL {temporada}: {res.text}")
            continue
            
        data = res.json().get("response", [])
        print(f" -> Se encontraron {len(data)} registros para la temporada {temporada}.")
        
        for g in data:
            # Validar estado del partido (en API-Sports de americano, FT o AET significa finalizado)
            status = g.get("status", {}).get("short")
            if status not in ["FT", "AET", "AOT"]: 
                continue
                
            fecha = g.get("date", "")[:10]
            
            # Extracción segura de nombres de equipos
            teams = g.get("teams", {})
            local = teams.get("home", {}).get("name", "Desconocido")
            visita = teams.get("away", {}).get("name", "Desconocido")
            
            # Extracción segura de puntajes
            scores = g.get("scores", {})
            home_scores = scores.get("home", {})
            away_scores = scores.get("away", {})
            
            # El total puede venir como 'total' o sumando los cuartos
            pts_local = home_scores.get("total")
            pts_visita = away_scores.get("total")
            
            if pts_local is None:
                # Fallback sumando cuartos si el total viene nulo
                pts_local = sum([home_scores.get(q, 0) or 0 for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]])
            if pts_visita is None:
                pts_visita = sum([away_scores.get(q, 0) or 0 for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]])
                
            if pts_local == 0 and pts_visita == 0:
                continue # Evitar partidos sin datos reales
                
            llave = f"{fecha}_{local}"
            if llave in partidos_guardados:
                continue
                
            nuevos_partidos.append({
                "Fecha": fecha,
                "Local": local,
                "Visitante": visita,
                "Puntos_L": int(pts_local),
                "Puntos_V": int(pts_visita),
                "Total_Puntos": int(pts_local) + int(pts_visita),
                "Diferencia": int(pts_local) - int(pts_visita)
            })
            
        time.sleep(0.5)

    # 3. Guardar en el archivo CSV independiente de la NFL
    if nuevos_partidos:
        df_nuevos = pd.DataFrame(nuevos_partidos)
        df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
        df_final.to_csv(ARCHIVO_CSV_NFL, index=False)
        print(f"✅ ¡Base de datos de NFL creada/actualizada con éxito! Se añadieron {len(nuevos_partidos)} partidos nuevos en '{ARCHIVO_CSV_NFL}'.")
    else:
        print("⚠️ No se encontraron partidos nuevos para agregar, o la estructura requiere revisión.")

if __name__ == "__main__":
    actualizar_base_datos_nfl()