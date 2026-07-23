import requests
import pandas as pd
import os
import time

# --- CONFIGURACIÓN ---
API_KEY = "1abc53997c1b26e3b447796665e36e44"  # <--- Asegúrate de poner tu llave real aquí
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262
TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026] # Historial completo con tus 7,500 llamadas diarias
ARCHIVO_CSV = 'data/historico_ligamx_completo.csv'

def obtener_valor_estadistica(estadisticas, nombre_stat, tipo="int"):
    """Busca una estadística en el JSON de API-Sports y la convierte al tipo correcto."""
    for stat in estadisticas:
        if stat["type"] == nombre_stat:
            val = stat["value"]
            if val is None: return 0.0 if tipo == "float" else 0
            return float(val) if tipo == "float" else int(val)
    return 0.0 if tipo == "float" else 0

def actualizar_base_datos():
    print("⚽ Iniciando actualización completa de Base de Datos (xG, Corners, Tarjetas y Árbitros)...")
    
    # 1. Cargar base existente para no repetir partidos
    if os.path.exists(ARCHIVO_CSV):
        df_existente = pd.read_csv(ARCHIVO_CSV)
        partidos_guardados = set(df_existente['Fecha'].astype(str).str[:10] + "_" + df_existente['Local'])
    else:
        df_existente = pd.DataFrame()
        partidos_guardados = set()
        os.makedirs('data', exist_ok=True)

    nuevos_partidos = []

    # 2. Buscar partidos terminados por temporada
    for temporada in TEMPORADAS:
        print(f"Buscando partidos de la temporada {temporada}...")
        res = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"league": LIGA_MX_ID, "season": temporada, "status": "FT"})
        
        if res.status_code != 200:
            print(f"Error al conectar con la API en temporada {temporada}: {res.text}")
            continue
            
        fixtures = res.json().get("response", [])
        
        for p in fixtures:
            fix_id = p["fixture"]["id"]
            fecha = p["fixture"]["date"][:10]
            local = p["teams"]["home"]["name"]
            visita = p["teams"]["away"]["name"]
            goles_l = p["goals"]["home"]
            goles_v = p["goals"]["away"]
            
            # Extracción del árbitro asignado por la API
            arbitro_raw = p["fixture"].get("referee")
            arbitro = arbitro_raw.strip() if arbitro_raw else "Desconocido"
            
            llave = f"{fecha}_{local}"
            
            # Si el partido ya está en tu CSV, nos lo saltamos para ahorrar tiempo y llamadas
            if llave in partidos_guardados:
                continue
                
            print(f"Descargando estadísticas de: {local} vs {visita} ({fecha})")
            
            # 3. Descargar estadísticas detalladas de este partido (Corners, Tarjetas, xG)
            res_stats = requests.get(f"{BASE_URL}/fixtures/statistics", headers=HEADERS, params={"fixture": fix_id})
            stats_data = res_stats.json().get("response", [])
            
            c_l = c_v = a_l = a_v = r_l = r_v = 0
            xg_l = xg_v = 0.0
            
            if len(stats_data) == 2:
                # Local
                stats_l = stats_data[0]["statistics"]
                c_l = obtener_valor_estadistica(stats_l, "Corner Kicks")
                a_l = obtener_valor_estadistica(stats_l, "Yellow Cards")
                r_l = obtener_valor_estadistica(stats_l, "Red Cards")
                xg_l = obtener_valor_estadistica(stats_l, "expected_goals", "float")
                
                # Visitante
                stats_v = stats_data[1]["statistics"]
                c_v = obtener_valor_estadistica(stats_v, "Corner Kicks")
                a_v = obtener_valor_estadistica(stats_v, "Yellow Cards")
                r_v = obtener_valor_estadistica(stats_v, "Red Cards")
                xg_v = obtener_valor_estadistica(stats_v, "expected_goals", "float")
            
            # 4. Guardar la fila estructurada
            nuevos_partidos.append({
                "Fecha": fecha,
                "Local": local,
                "Visitante": visita,
                "Goles_L": goles_l,
                "Goles_V": goles_v,
                "Corners_L": c_l,
                "Corners_V": c_v,
                "Amarillas_L": a_l,
                "Amarillas_V": a_v,
                "Rojas_L": r_l,
                "Rojas_V": r_v,
                "xG_L": xg_l if xg_l > 0 else goles_l, # Seguro de vida: si no hay xG, usa goles reales
                "xG_V": xg_v if xg_v > 0 else goles_v,
                "Arbitro": arbitro
            })
            
            # Pausa de 1 segundo para proteger la conexión con la API
            time.sleep(1) 

    # 5. Unir y Guardar en CSV
    if nuevos_partidos:
        df_nuevos = pd.DataFrame(nuevos_partidos)
        df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
        df_final.to_csv(ARCHIVO_CSV, index=False)
        print(f"✅ ¡Actualización completada! Se añadieron {len(nuevos_partidos)} partidos nuevos con xG, Estadísticas y Árbitros.")
    else:
        print("✅ Tu base de datos ya está al día. No hubo partidos nuevos que añadir.")

if __name__ == "__main__":
    actualizar_base_datos()