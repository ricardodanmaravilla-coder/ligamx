import os
import requests
import pandas as pd
import time

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

LIGAS = [772] 
# Ampliamos el rango de búsqueda de años por si la API los cataloga diferente
TEMPORADAS = [2023, 2024, 2025, 2026] 

def descargar_historico():
    todos_los_partidos = []
    
    for liga_id in LIGAS:
        for season in TEMPORADAS:
            url_fixtures = f"{BASE_URL}/fixtures"
            params_fixtures = {"league": liga_id, "season": season}
            
            try:
                res = requests.get(url_fixtures, headers=HEADERS, params=params_fixtures)
                data = res.json()
                
                # Revisar si la API arrojó un error directo (como "not subscribed")
                if data.get("errors"):
                    print(f"🚨 ERROR DE LA API en Liga {liga_id} - Año {season}: {data.get('errors')}")
                    continue
                
                resultados = data.get("response", [])
                print(f"\n🔍 Liga {liga_id} - Año {season}: La API devolvió {len(resultados)} juegos en total.")
                
                if not resultados:
                    continue
                    
                # Analizar el primer partido para ver cómo viene la estructura
                print("   Ejemplo del primer partido recibido:")
                print(f"   Estatus: {resultados[0].get('fixture', {}).get('status', {}).get('short')}")
                print(f"   Local: {resultados[0].get('teams', {}).get('home', {}).get('name')} | Goles: {resultados[0].get('goals', {}).get('home')}")
                
            except Exception as e:
                print(f"❌ Error crítico de conexión: {e}")
                
if __name__ == "__main__":
    descargar_historico()
