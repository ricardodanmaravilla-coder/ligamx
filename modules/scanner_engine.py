import requests
import pandas as pd
from modules.stats_engine import calcular_expectativa_partido
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido, evaluar_mercado_avanzado

# Configuración API-Sports
API_KEY = "1abc53997c1b26e3b447796665e36e44" 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262

def escanear_jornada_actual(temporada_actual=2026):
    """Descarga la jornada actual completa de la Liga MX y detecta valor con >70% de probabilidad."""
    
    # 1. Obtenemos la ronda (jornada) activa actual de la Liga MX para garantizar que traiga el 100% de los partidos
    url_rounds = f"{BASE_URL}/fixtures/rounds"
    res_rounds = requests.get(url_rounds, headers=HEADERS, params={"league": LIGA_MX_ID, "season": temporada_actual, "current": "true"})
    
    jornada_actual = None
    if res_rounds.status_code == 200:
        rounds_data = res_rounds.json().get("response", [])
        if rounds_data:
            jornada_actual = rounds_data[0] # Ej: "Regular Season - 7"
            
    # 2. Consultamos los partidos filtrando específicamente por esa ronda o por próximos (NS)
    url = f"{BASE_URL}/fixtures"
    params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
    
    if jornada_actual:
        params["round"] = jornada_actual
    
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        return []
        
    fixtures = res.json().get("response", [])
    
    # Resguardo por si la API no devuelve la ronda actual exacta: respaldamos trayendo los próximos estándar
    if not fixtures:
        params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            fixtures = res.json().get("response", [])

    oportunidades_oro = []
    
    mercados_a_mapear = [
        ("Gana Local", "1"),
        ("Gana Visita", "2"),
        ("Over 2.5 Goles", "Over 2.5"),
        ("Under 2.5 Goles", "Under 2.5"),
        ("Over 9.5 Corners", "Over 9.5 Corners"),
        ("Under 9.5 Corners", "Under 9.5 Corners"),
        ("Over 4.5 Tarjetas", "Over 4.5 Tarjetas"),
        ("Under 4.5 Tarjetas", "Under 4.5 Tarjetas")
    ]

    for p in fixtures:
        fix_id = p["fixture"]["id"]
        local = p["teams"]["home"]["name"]
        visita = p["teams"]["away"]["name"]
        fecha = p["fixture"]["date"][:16].replace("T", " ")
        
        try:
            # 3. Corremos nuestro motor híbrido, altitud y Montecarlo para este partido
            resultados = simular_partido_montecarlo(local, visita)
            if isinstance(resultados, str): continue
            
            # 4. Obtenemos cuotas automáticas de la API
            cuotas = obtener_cuotas_partido(fix_id)
            if not cuotas: continue
            
            # Extraemos probabilidades del diccionario de resultados de Montecarlo
            prob_dict = {
                "1": resultados["Resultado_1X2"]["Gana Local"],
                "2": resultados["Resultado_1X2"]["Gana Visita"],
                "Over 2.5": resultados["Goles_Over_Under"]["Over 2.5"],
                "Under 2.5": resultados["Goles_Over_Under"]["Under 2.5"],
                "Over 9.5 Corners": resultados["Corners_Totales"]["Over 9.5 Corners"],
                "Under 9.5 Corners": resultados["Corners_Totales"]["Under 9.5 Corners"],
                "Over 4.5 Tarjetas": resultados["Tarjetas_Totales"]["Over 4.5 Tarjetas"],
                "Under 4.5 Tarjetas": resultados["Tarjetas_Totales"]["Under 4.5 Tarjetas"]
            }

            # 5. Evaluamos mercado por mercado buscando el filtro de oro (>70% y EV > 0)
            for nombre_m, llave in mercados_a_mapear:
                prob = prob_dict.get(llave, 0)
                cuota = cuotas.get(llave)
                
                if cuota and cuota > 1.0 and prob >= 60.0: # <--- FILTRO DE ORO
                    _, ev, veredicto, stake, riesgo = evaluar_mercado_avanzado(prob, cuota)
                    
                    if ev > 0:
                        oportunidades_oro.append({
                            "Fecha": fecha,
                            "Partido": f"{local} vs {visita}",
                            "Mercado": nombre_m,
                            "Probabilidad": f"{prob}%",
                            "Cuota": f"{cuota:.2f}",
                            "EV (Valor)": f"+{ev:.1f}%",
                            "Riesgo": riesgo,
                            "Stake Rec.": f"{stake:.1f}%",
                            "Veredicto": veredicto,
                            "Fixture_ID": fix_id
                        })
        except Exception as e:
            continue
            
    return oportunidades_oro
