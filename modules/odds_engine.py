import os
import requests
import pandas as pd
from difflib import SequenceMatcher

# Nueva variable de entorno para The Odds API
THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY", "de66554a17bce1149445b1a883056607")

def son_similares(a, b, umbral=0.55):
    """Ayuda a emparejar los nombres de API-Sports con los nombres de The Odds API"""
    if not a or not b: return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > umbral

def obtener_cuotas_partido(local, visita, league_id=262):
    """Descarga las cuotas directamente desde The Odds API."""
    
    # Mapeo de liga a sport_key de The Odds API
    sport_key = "soccer_mexico_ligamx" if league_id == 262 else "soccer_usa_mls"
        
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us,eu", # Busca en casas de USA y Europa
        "markets": "h2h,totals", # h2h = Ganador, totals = Over/Under Goles
        "oddsFormat": "decimal"
    }
    
    # Líneas y cuotas por defecto en caso de que The Odds API no las tenga
    cuotas_limpias = {"Linea_Goles": 2.5, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
    
    if THE_ODDS_API_KEY == "de66554a17bce1149445b1a883056607" or not THE_ODDS_API_KEY:
        print("Falta THE_ODDS_API_KEY")
        return cuotas_limpias
        
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200: 
            return cuotas_limpias
        
        datos = res.json()
        
        for partido in datos:
            home_api = partido.get("home_team", "")
            away_api = partido.get("away_team", "")
            
            # Cruzamos los nombres de los equipos
            if son_similares(local, home_api) or son_similares(visita, away_api):
                bookmakers = partido.get("bookmakers", [])
                if not bookmakers: continue
                
                # Tomamos la primera casa de apuestas disponible (suelen ser las más precisas de Las Vegas/Pinnacle)
                mercados = bookmakers[0].get("markets", [])
                
                for m in mercados:
                    key = m.get("key")
                    outcomes = m.get("outcomes", [])
                    
                    # --- 1. GANADOR DEL PARTIDO (1X2) ---
                    if key == "h2h":
                        for out in outcomes:
                            name = out.get("name")
                            if name == home_api: cuotas_limpias["1"] = float(out.get("price"))
                            elif name == "Draw": cuotas_limpias["X"] = float(out.get("price"))
                            else: cuotas_limpias["2"] = float(out.get("price"))
                            
                    # --- 2. GOLES OVER/UNDER Y LÍNEA DE CASINO ---
                    elif key == "totals":
                        for out in outcomes:
                            if "point" in out:
                                cuotas_limpias["Linea_Goles"] = float(out["point"])
                            name = out.get("name")
                            if name == "Over": cuotas_limpias["Over_Goles"] = float(out.get("price"))
                            elif name == "Under": cuotas_limpias["Under_Goles"] = float(out.get("price"))
                break # Rompemos el ciclo si ya encontramos el partido
                
    except Exception as e:
        print(f"Error consultando The Odds API: {e}")
        
    return cuotas_limpias

def evaluar_mercado_avanzado(probabilidad_modelo_pct, cuota_casa):
    if not cuota_casa or cuota_casa <= 0:
        return "N/A", 0, "SIN CUOTA", 0, "N/A"
        
    prob_real = probabilidad_modelo_pct / 100
    ev_pct = ((prob_real * cuota_casa) - 1) * 100
    
    b = cuota_casa - 1.0 
    q = 1.0 - prob_real  
    kelly_pct = ((b * prob_real) - q) / b if b > 0 else 0
    
    stake_recomendado = max(0, (kelly_pct * 100) / 4)
    
    if ev_pct <= 0:
        veredicto = "❌ DESCARTAR (EV Negativo)"
        riesgo = "Alto"
        stake_recomendado = 0
    elif prob_real < 0.40:
        veredicto = "⚠️ TRAMPA DE VALOR (Prob < 40%)"
        riesgo = "Extremo"
        stake_recomendado = 0
    elif ev_pct >= 2.0 and prob_real >= 0.50:
        veredicto = "🔥 APUESTA ESTRELLA"
        riesgo = "Bajo"
    else:
        veredicto = "✅ ACEPTABLE"
        riesgo = "Medio"
        
    if stake_recomendado == 0 and "APUESTA" in veredicto:
         veredicto = "❌ DESCARTAR (Kelly = 0)"

    return cuota_casa, ev_pct, veredicto, stake_recomendado, riesgo

def analizar_apuestas(resultados_montecarlo, fixture_id, cuotas_personalizadas=None, lineas_default=None):
    if cuotas_personalizadas and len(cuotas_personalizadas) > 0:
        cuotas = cuotas_personalizadas
        l_goles = lineas_default.get("Linea_Goles", 2.5) if lineas_default else 2.5
        l_corners = lineas_default.get("Linea_Corners", 9.5) if lineas_default else 9.5
        l_tarjetas = lineas_default.get("Linea_Tarjetas", 4.5) if lineas_default else 4.5
    else:
        cuotas = obtener_cuotas_partido(fixture_id) # En The Odds API, el fixture_id aquí es ignorado si se pasa el diccionario directo
        if not cuotas: return pd.DataFrame() 
        l_goles = cuotas.get("Linea_Goles", 2.5)
        l_corners = cuotas.get("Linea_Corners", 9.5)
        l_tarjetas = cuotas.get("Linea_Tarjetas", 4.5)
        
    analisis = []
    mercados_a_evaluar = [
        ("Gana Local", resultados_montecarlo["Resultado_1X2"]["Gana Local"], "1"),
        ("Empate", resultados_montecarlo["Resultado_1X2"]["Empate"], "X"),
        ("Gana Visita", resultados_montecarlo["Resultado_1X2"]["Gana Visita"], "2"),
        (f"Over {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"][f"Over {l_goles}"], "Over_Goles"),
        (f"Under {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"][f"Under {l_goles}"], "Under_Goles"),
        (f"Over {l_corners} Corners", resultados_montecarlo["Corners_Totales"][f"Over {l_corners} Corners"], "Over_Corners"),
        (f"Under {l_corners} Corners", resultados_montecarlo["Corners_Totales"][f"Under {l_corners} Corners"], "Under_Corners"),
        (f"Over {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"][f"Over {l_tarjetas} Tarjetas"], "Over_Tarjetas"),
        (f"Under {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"][f"Under {l_tarjetas} Tarjetas"], "Under_Tarjetas")
    ]
    
    for nombre_m, prob, llave_cuota in mercados_a_evaluar:
        cuota = cuotas.get(llave_cuota)
        
        if cuota is None or cuota == 0.0:
            analisis.append([nombre_m, f"{prob}%", "Sin Cuota", "N/A", "N/A", "0%", "🕒 Ingresa Cuota"])
            continue

        c_fmt, ev, veredicto, stake, riesgo = evaluar_mercado_avanzado(prob, cuota)
        ev_str = f"{ev:.1f}%"
        stake_str = f"{stake:.1f}%" if stake > 0 else "0%"
        
        analisis.append([nombre_m, f"{prob}%", c_fmt, ev_str, riesgo, stake_str, veredicto])
    
    return pd.DataFrame(analisis, columns=["Mercado", "Prob. Modelo", "Cuota", "EV", "Riesgo", "Stake (Bankroll)", "Veredicto"])
