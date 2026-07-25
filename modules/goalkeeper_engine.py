import os
import requests
import streamlit as st

# Usamos la misma clave API que tienes en tu app.py
API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262

@st.cache_data(ttl=86400) # Cacheamos por 24h para no quemar cuota de tu API
def obtener_ultimos_fixtures_equipo(equipo_id, cantidad=3):
    """Busca los IDs de los últimos partidos jugados por un equipo."""
    url = f"{BASE_URL}/fixtures"
    params = {
        "league": LIGA_MX_ID,
        "team": equipo_id,
        "last": cantidad,
        "status": "FT" # Solo partidos terminados
    }
    try:
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            datos = res.json().get("response", [])
            return [p["fixture"]["id"] for p in datos]
    except Exception as e:
        print(f"Error obteniendo fixtures del equipo {equipo_id}: {e}")
    return []

@st.cache_data(ttl=86400)
def obtener_stats_portero_partido(fixture_id, equipo_id):
    """Extrae las atajadas y tiros al arco recibidos de un partido específico."""
    url = f"{BASE_URL}/fixtures/statistics"
    params = {"fixture": fixture_id, "team": equipo_id}
    
    atajadas = 0
    tiros_al_arco = 0
    
    try:
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            datos = res.json().get("response", [])
            if datos:
                stats = datos[0].get("statistics", [])
                for stat in stats:
                    # La API de football-data / api-sports devuelve estos tipos:
                    if stat["type"] == "Goalkeeper Saves" and stat["value"] is not None:
                        atajadas = int(stat["value"])
                    if stat["type"] == "Shots on Goal" and stat["value"] is not None:
                        # OJO: Estos son los tiros AL arco que hizo el equipo, 
                        # no los que recibió. Necesitamos otra lógica si no tenemos la estadística rival directa.
                        pass
                        
    except Exception as e:
        print(f"Error stats portero fix {fixture_id}: {e}")
        
    return atajadas

# --- LÓGICA REFINADA PARA API-SPORTS ---
# Como la API te da las atajadas del portero de un equipo, pero los "Tiros al arco" 
# de esa misma respuesta son los tiros ofensivos del propio equipo, necesitamos 
# comparar el total de atajadas vs el total de goles recibidos para sacar un ratio de eficiencia.

@st.cache_data(ttl=86400)
def calcular_eficiencia_portero_api(equipo_id, equipo_nombre):
    """
    Calcula un factor de portero basado en atajadas / (atajadas + goles recibidos)
    de los últimos 3 partidos, extrayendo datos directo de la API.
    """
    ultimos_fixtures = obtener_ultimos_fixtures_equipo(equipo_id, cantidad=3)
    
    if not ultimos_fixtures:
        return 1.0 # Si no hay datos, el factor es neutro
        
    atajadas_totales = 0
    goles_recibidos_totales = 0
    
    for fix_id in ultimos_fixtures:
        # 1. Obtener atajadas
        url_stats = f"{BASE_URL}/fixtures/statistics"
        res_stats = requests.get(url_stats, headers=HEADERS, params={"fixture": fix_id, "team": equipo_id})
        
        if res_stats.status_code == 200:
            datos_stats = res_stats.json().get("response", [])
            if datos_stats:
                for stat in datos_stats[0].get("statistics", []):
                    if stat["type"] == "Goalkeeper Saves" and stat["value"] is not None:
                        atajadas_totales += int(stat["value"])
                        
        # 2. Obtener goles recibidos en ese mismo partido
        url_fix = f"{BASE_URL}/fixtures"
        res_fix = requests.get(url_fix, headers=HEADERS, params={"id": fix_id})
        
        if res_fix.status_code == 200:
            datos_fix = res_fix.json().get("response", [])
            if datos_fix:
                p = datos_fix[0]
                if str(p["teams"]["home"]["id"]) == str(equipo_id):
                    goles_recibidos_totales += p["goals"]["away"]
                else:
                    goles_recibidos_totales += p["goals"]["home"]

    # --- CALCULO DEL FACTOR ---
    tiros_al_arco_rivales = atajadas_totales + goles_recibidos_totales
    
    if tiros_al_arco_rivales == 0:
        return 1.0
        
    porcentaje_atajadas = atajadas_totales / tiros_al_arco_rivales
    promedio_liga = 0.70
    
    # Si ataja más del 70%, su factor será < 1.0 (Baja los goles del rival)
    # Si ataja menos del 70%, su factor será > 1.0 (Sube los goles del rival)
    factor_portero = (1 - porcentaje_atajadas) / (1 - promedio_liga)
    factor_portero = max(0.7, min(factor_portero, 1.3))
    
    # print(f"🧤 {equipo_nombre}: Atajó {atajadas_totales} de {tiros_al_arco_rivales} ({round(porcentaje_atajadas*100)}%). Factor: {round(factor_portero, 2)}")
    return factor_portero
