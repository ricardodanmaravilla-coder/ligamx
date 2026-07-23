import requests
import pandas as pd
import os
import time

# --- Configuración API-Sports ---
API_KEY = "1abc53997c1b26e3b447796665e36e44"
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    'x-apisports-key': API_KEY
}

LIGA_MX_ID = 262

def obtener_stats_partido(fixture_id):
    """Obtiene corners y tarjetas para un partido específico."""
    url = f"{BASE_URL}/fixtures/statistics"
    response = requests.get(url, headers=HEADERS, params={"fixture": fixture_id})
    
    if response.status_code != 200:
        return {}
        
    datos = response.json().get("response", [])
    if not datos or len(datos) < 2:
        return {}

    # Mapeo de estadísticas
    stats_local = {s['type']: s['value'] for s in datos[0]['statistics']}
    stats_visita = {s['type']: s['value'] for s in datos[1]['statistics']}
    
    return {
        "Corners_L": stats_local.get("Corner Kicks") or 0,
        "Corners_V": stats_visita.get("Corner Kicks") or 0,
        "Amarillas_L": stats_local.get("Yellow Cards") or 0,
        "Amarillas_V": stats_visita.get("Yellow Cards") or 0,
        "Rojas_L": stats_local.get("Red Cards") or 0,
        "Rojas_V": stats_visita.get("Red Cards") or 0
    }

def descargar_historico_completo(temporadas=[2021, 2022, 2023, 2024, 2025, 2026]):
    """Descarga marcadores y estadísticas detalladas de todas las temporadas."""
    partidos_totales = []
    
    for temp in temporadas:
        print(f"\nObteniendo calendario de la temporada {temp}...")
        url = f"{BASE_URL}/fixtures"
        res = requests.get(url, headers=HEADERS, params={"league": LIGA_MX_ID, "season": temp})
        
        if res.status_code != 200:
            continue
            
        fixtures = res.json().get("response", [])
        
        # Filtramos partidos terminados
        terminados = [f for f in fixtures if f["fixture"]["status"]["short"] == "FT"]
        print(f"Descargando detalles de {len(terminados)} partidos procesados para {temp}...")
        
        for idx, f in enumerate(terminados):
            fix_id = f["fixture"]["id"]
            
            # Datos base
            fila = {
                "Fixture_ID": fix_id,
                "Temporada": temp,
                "Fecha": f["fixture"]["date"][:10],
                "Jornada": f["league"]["round"],
                "Local": f["teams"]["home"]["name"],
                "Visitante": f["teams"]["away"]["name"],
                "Goles_L": f["goals"]["home"],
                "Goles_V": f["goals"]["away"]
            }
            
            # Petición adicional de estadísticas
            stats = obtener_stats_partido(fix_id)
            fila.update(stats)
            
            partidos_totales.append(fila)
            
            # Progreso en consola
            if (idx + 1) % 50 == 0 or (idx + 1) == len(terminados):
                print(f"  -> Procesados {idx + 1}/{len(terminados)} partidos...")
            
            # Pequeña pausa para ser amigables con el servidor
            time.sleep(0.1)

    df = pd.DataFrame(partidos_totales)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    ruta = 'data/historico_ligamx_completo.csv'
    df.to_csv(ruta, index=False)
    print(f"\n Base de datos histórica completa guardada en: {ruta}")
    return df

if __name__ == "__main__":
    df_completo = descargar_historico_completo()