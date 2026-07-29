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
                print(f"⚽ Se encontraron {len(resultados)} partidos. Filtrando finalizados...")
                
                for p in resultados:
                    # 'FT' = Tiempo Completo, 'AET' = Tras Tiempo Extra, 'PEN' = Penales
                    status = p.get("fixture", {}).get("status", {}).get("short", "")
                    if status not in ["FT", "AET", "PEN"]:
                        continue
                        
                    try:
                        g_loc = p.get("goals", {}).get("home")
                        g_vis = p.get("goals", {}).get("away")
                        
                        # Ignorar si no hay goles registrados
                        if g_loc is None or g_vis is None:
                            continue

                        todos_los_partidos.append({
                            "Fecha": p.get("fixture", {}).get("date", "")[:10],
                            "Temporada": season,
                            "Liga_ID": liga_id,
                            "Local": p.get("teams", {}).get("home", {}).get("name", "Unknown"),
                            "Visitante": p.get("teams", {}).get("away", {}).get("name", "Unknown"),
                            "Goles_Local": int(g_loc),
                            "Goles_Visita": int(g_vis)
                        })
                    except Exception as e:
                        print(f"⚠️ Error procesando un partido: {e}")
                        continue
                        
                # Pausa estricta de 2 segundos para no saturar los límites de la API gratuita
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error de conexión: {e}")
                
    if not todos_los_partidos:
        print("\n⚠️ ATENCIÓN: No se obtuvieron datos válidos de la API. El archivo CSV NO fue generado.")
        print("Revisa tu cuota de peticiones o si tienes acceso a la liga en tu plan actual.")
    else:
        df = pd.DataFrame(todos_los_partidos).dropna()
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values(by='Fecha').reset_index(drop=True)
        
        # Guardamos con el nombre exacto que configuramos en app.py
        ruta_csv = 'data/historico_leaguescup.csv'
        df.to_csv(ruta_csv, index=False)
        print(f"\n🎉 ¡ÉXITO! Archivo '{ruta_csv}' generado con {len(df)} partidos reales.")

if __name__ == "__main__":
    if not API_KEY:
        print("🚨 ERROR FATAL: La llave API_SPORTS_KEY no está configurada.")
    else:
        print("🚀 Iniciando descarga del histórico MLS & Leagues Cup...")
        descargar_historico()
