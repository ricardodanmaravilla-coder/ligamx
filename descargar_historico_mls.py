import os
import requests
import pandas as pd
import time

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# 253 = Major League Soccer (MLS)
# 848 = Leagues Cup
LIGAS = [253, 848] 
TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]

def crear_directorio_data():
    if not os.path.exists('data'):
        os.makedirs('data')

def extraer_estadistica(stats_list, stat_name):
    """Busca una estadística específica dentro de la lista que devuelve la API"""
    if not isinstance(stats_list, list):
        return 0.0
    for stat in stats_list:
        if stat.get("type") == stat_name:
            val = stat.get("value")
            if val is None:
                return 0.0
            # Si es un string con porcentaje (ej. "55%"), limpiarlo
            if isinstance(val, str) and "%" in val:
                return float(val.replace("%", ""))
            return float(val)
    return 0.0

def descargar_historico():
    crear_directorio_data()
    todos_los_partidos = []
    
    for liga_id in LIGAS:
        for season in TEMPORADAS:
            print(f"\n📥 Solicitando Liga {liga_id} - Temporada {season}...")
            url = f"{BASE_URL}/fixtures"
            params = {
                "league": liga_id, 
                "season": season
            }
            
            try:
                response = requests.get(url, headers=HEADERS, params=params)
                if response.status_code != 200:
                    print(f"❌ Error HTTP {response.status_code}: {response.text}")
                    continue
                    
                data = response.json()
                if data.get("errors"):
                    print(f"🚨 La API rechazó la consulta: {data.get('errors')}")
                    continue
                    
                resultados = data.get("response", [])
                print(f"⚽ Se encontraron {len(resultados)} partidos. Extrayendo estadísticas completas...")
                
                for p in resultados:
                    status = p.get("fixture", {}).get("status", {}).get("short", "")
                    if status not in ["FT", "AET", "PEN"]:
                        continue
                        
                    try:
                        g_loc = p.get("goals", {}).get("home")
                        g_vis = p.get("goals", {}).get("away")
                        
                        if g_loc is None or g_vis is None:
                            continue

                        # Extraer el árbitro
                        arbitro = p.get("fixture", {}).get("referee")
                        if not arbitro:
                            arbitro = "Desconocido"
                        else:
                            arbitro = arbitro.split(',')[0].strip() # Limpiar nombre

                        # Extraer estadísticas detalladas (si están disponibles)
                        stats = p.get("statistics", [])
                        if len(stats) >= 2:
                            # Asegurarnos de mapear Local y Visita correctamente por el ID del equipo
                            team_id_local = p.get("teams", {}).get("home", {}).get("id")
                            
                            # Identificar cuál índice es local y cuál es visita
                            if stats[0].get("team", {}).get("id") == team_id_local:
                                stats_loc = stats[0].get("statistics", [])
                                stats_vis = stats[1].get("statistics", [])
                            else:
                                stats_loc = stats[1].get("statistics", [])
                                stats_vis = stats[0].get("statistics", [])
                        else:
                            stats_loc = []
                            stats_vis = []

                        todos_los_partidos.append({
                            "Fecha": p.get("fixture", {}).get("date", "")[:10],
                            "Temporada": season,
                            "Liga_ID": liga_id,
                            "Local": p.get("teams", {}).get("home", {}).get("name", "Unknown"),
                            "Visitante": p.get("teams", {}).get("away", {}).get("name", "Unknown"),
                            "Arbitro": arbitro,
                            "Goles_Local": int(g_loc),
                            "Goles_Visita": int(g_vis),
                            
                            # Córners
                            "Corners_L": int(extraer_estadistica(stats_loc, "Corner Kicks")),
                            "Corners_V": int(extraer_estadistica(stats_vis, "Corner Kicks")),
                            
                            # Tarjetas (Amarillas y Rojas separadas para el modelo ML)
                            "Amarillas_L": int(extraer_estadistica(stats_loc, "Yellow Cards")),
                            "Amarillas_V": int(extraer_estadistica(stats_vis, "Yellow Cards")),
                            "Rojas_L": int(extraer_estadistica(stats_loc, "Red Cards")),
                            "Rojas_V": int(extraer_estadistica(stats_vis, "Red Cards")),
                            
                            # Tiros a Puerta
                            "Tiros_Puerta_L": int(extraer_estadistica(stats_loc, "Shots on Goal")),
                            "Tiros_Puerta_V": int(extraer_estadistica(stats_vis, "Shots on Goal")),
                            
                            # Atajadas
                            "Atajadas_L": int(extraer_estadistica(stats_loc, "Goalkeeper Saves")),
                            "Atajadas_V": int(extraer_estadistica(stats_vis, "Goalkeeper Saves")),
                            
                            # xG (Expectativa de Goles - Dato avanzado)
                            "xG_L": round(extraer_estadistica(stats_loc, "expected_goals"), 2),
                            "xG_V": round(extraer_estadistica(stats_vis, "expected_goals"), 2)
                        })
                    except Exception as e:
                        print(f"⚠️ Error procesando un partido: {e}")
                        continue
                        
                # Pausa para no saturar la API
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error de conexión: {e}")
                
    if not todos_los_partidos:
        print("\n⚠️ ATENCIÓN: No se obtuvieron datos válidos de la API. El archivo CSV NO fue generado.")
    else:
        df = pd.DataFrame(todos_los_partidos).dropna()
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values(by='Fecha').reset_index(drop=True)
        
        ruta_csv = 'data/historico_leaguescup.csv'
        df.to_csv(ruta_csv, index=False)
        print(f"\n🎉 ¡ÉXITO! Archivo '{ruta_csv}' generado con {len(df)} partidos detallados.")
        print(f"📊 Columnas agregadas: Arbitro, Córners, Amarillas, Rojas, Tiros, Atajadas, xG.")

if __name__ == "__main__":
    if not API_KEY:
        print("🚨 ERROR FATAL: La llave API_SPORTS_KEY no está configurada.")
    else:
        print("🚀 Iniciando descarga del histórico MLS & Leagues Cup con estadísticas completas...")
        descargar_historico()
