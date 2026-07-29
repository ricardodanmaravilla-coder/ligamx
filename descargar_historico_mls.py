import os
import requests
import pandas as pd
import time

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

LIGAS = [772] 
TEMPORADAS = [2026]

def crear_directorio_data():
    if not os.path.exists('data'):
        os.makedirs('data')

def extraer_estadistica(stats_list, stat_name):
    if not isinstance(stats_list, list):
        return 0.0
    for stat in stats_list:
        if stat.get("type") == stat_name:
            val = stat.get("value")
            if val is None:
                return 0.0
            if isinstance(val, str) and "%" in val:
                return float(val.replace("%", ""))
            return float(val)
    return 0.0

def descargar_historico():
    crear_directorio_data()
    todos_los_partidos = []
    peticiones_api = 0
    
    for liga_id in LIGAS:
        for season in TEMPORADAS:
            print(f"\n📥 Solicitando calendario de Liga {liga_id} - Temporada {season}...")
            url_fixtures = f"{BASE_URL}/fixtures"
            params_fixtures = {"league": liga_id, "season": season}
            
            try:
                res_fixtures = requests.get(url_fixtures, headers=HEADERS, params=params_fixtures)
                peticiones_api += 1
                
                if res_fixtures.status_code != 200:
                    print(f"❌ Error HTTP {res_fixtures.status_code}")
                    continue
                    
                data_fixtures = res_fixtures.json()
                resultados = data_fixtures.get("response", [])
                print(f"⚽ Se encontraron {len(resultados)} partidos. Descargando estadísticas uno por uno...")
                
                for i, p in enumerate(resultados):
                    status = p.get("fixture", {}).get("status", {}).get("short", "")
                    if status not in ["FT", "AET", "PEN"]:
                        continue
                        
                    fixture_id = p.get("fixture", {}).get("id")
                    if not fixture_id:
                        continue

                    # --- LA CLAVE: PEDIR LAS ESTADÍSTICAS ESPECÍFICAS DE ESTE PARTIDO ---
                    url_stats = f"{BASE_URL}/fixtures/statistics"
                    res_stats = requests.get(url_stats, headers=HEADERS, params={"fixture": fixture_id})
                    peticiones_api += 1
                    
                    if res_stats.status_code != 200:
                        continue
                        
                    stats_response = res_stats.json().get("response", [])
                    
                    # Mapear correctamente quién es Local y Visita en las estadísticas
                    team_local_id = p.get("teams", {}).get("home", {}).get("id")
                    stats_loc = []
                    stats_vis = []
                    
                    if len(stats_response) >= 2:
                        if stats_response[0].get("team", {}).get("id") == team_local_id:
                            stats_loc = stats_response[0].get("statistics", [])
                            stats_vis = stats_response[1].get("statistics", [])
                        else:
                            stats_loc = stats_response[1].get("statistics", [])
                            stats_vis = stats_response[0].get("statistics", [])

                    g_loc = p.get("goals", {}).get("home")
                    g_vis = p.get("goals", {}).get("away")
                    
                    if g_loc is None or g_vis is None:
                        continue

                    arbitro = p.get("fixture", {}).get("referee")
                    arbitro = arbitro.split(',')[0].strip() if arbitro else "Desconocido"

                    todos_los_partidos.append({
                        "Fecha": p.get("fixture", {}).get("date", "")[:10],
                        "Temporada": season,
                        "Liga_ID": liga_id,
                        "Local": p.get("teams", {}).get("home", {}).get("name", "Unknown"),
                        "Visitante": p.get("teams", {}).get("away", {}).get("name", "Unknown"),
                        "Arbitro": arbitro,
                        "Goles_Local": int(g_loc),
                        "Goles_Visita": int(g_vis),
                        
                        # Estadísticas extraídas de forma individual
                        "Corners_L": int(extraer_estadistica(stats_loc, "Corner Kicks")),
                        "Corners_V": int(extraer_estadistica(stats_vis, "Corner Kicks")),
                        "Amarillas_L": int(extraer_estadistica(stats_loc, "Yellow Cards")),
                        "Amarillas_V": int(extraer_estadistica(stats_vis, "Yellow Cards")),
                        "Rojas_L": int(extraer_estadistica(stats_loc, "Red Cards")),
                        "Rojas_V": int(extraer_estadistica(stats_vis, "Red Cards")),
                        "Tiros_Puerta_L": int(extraer_estadistica(stats_loc, "Shots on Goal")),
                        "Tiros_Puerta_V": int(extraer_estadistica(stats_vis, "Shots on Goal")),
                        "Atajadas_L": int(extraer_estadistica(stats_loc, "Goalkeeper Saves")),
                        "Atajadas_V": int(extraer_estadistica(stats_vis, "Goalkeeper Saves")),
                        "xG_L": round(extraer_estadistica(stats_loc, "expected_goals"), 2),
                        "xG_V": round(extraer_estadistica(stats_vis, "expected_goals"), 2)
                    })
                    
                    print(f"   [{i+1}/{len(resultados)}] Datos y stats descargados para ID: {fixture_id}")
                    # Pausa estricta de 1.5 seg para no sobrecargar los límites por segundo de la API
                    time.sleep(1.5) 
                    
            except Exception as e:
                print(f"❌ Error procesando la temporada: {e}")
                
    if not todos_los_partidos:
        print("\n⚠️ ATENCIÓN: No se obtuvieron datos. Verifica tu conexión o límites de la API.")
    else:
        df = pd.DataFrame(todos_los_partidos).dropna()
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values(by='Fecha').reset_index(drop=True)
        ruta_csv = 'data/historico_leaguescup.csv'
        df.to_csv(ruta_csv, index=False)
        print(f"\n🎉 ¡ÉXITO! Archivo '{ruta_csv}' generado con {len(df)} partidos detallados.")
        print(f"⚠️ Peticiones totales consumidas a la API: {peticiones_api}")

if __name__ == "__main__":
    if not API_KEY:
        print("🚨 ERROR FATAL: La llave API_SPORTS_KEY no está configurada.")
    else:
        print("🚀 Iniciando descarga profunda de MLS & Leagues Cup...")
        descargar_historico()
