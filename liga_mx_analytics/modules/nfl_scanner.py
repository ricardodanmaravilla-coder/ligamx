import requests
import pandas as pd
from modules.nfl_stats_engine import simular_partido_nfl_montecarlo
from modules.odds_engine import evaluar_mercado_avanzado

API_KEY = "1abc53997c1b26e3b447796665e36e44" 
BASE_URL = "https://v3.football.api-sports.io" # Nota: API-Sports usa el mismo dominio base para varios deportes cambiando endpoints
HEADERS = {'x-apisports-key': API_KEY}
NFL_LEAGUE_ID = 1  # ID común para NFL en API-Sports

def escanear_semana_nfl(temporada=2026):
    """Escanea los partidos de la NFL en busca de valor con >70% de probabilidad."""
    
    # Endpoint de fixtures para NFL
    url = f"{BASE_URL}/american-football/fixtures"
    params = {"league": NFL_LEAGUE_ID, "season": temporada, "status": "NS"}
    
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        return []
        
    fixtures = res.json().get("response", [])
    oportunidades_nfl = []

    for p in fixtures:
        try:
            fix_id = p["game"]["id"]
            local = p["teams"]["home"]["name"]
            visita = p["teams"]["away"]["name"]
            fecha = p["game"]["date"][:16].replace("T", " ")
            
            # Simulamos por Montecarlo el partido de NFL
            resultados = simular_partido_nfl_montecarlo(local, visita)
            
            prob_local = resultados["Moneyline"]["Gana Local"]
            prob_visita = resultados["Moneyline"]["Gana Visita"]
            
            # Evaluamos si hay alta probabilidad (>70%)
            # (Las cuotas de la NFL se obtendrían de tu módulo de odds adaptado al endpoint de americano)
            
        except Exception as e:
            continue
            
    return oportunidades_nfl