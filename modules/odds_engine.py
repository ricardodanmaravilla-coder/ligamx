import os
import requests
import pandas as pd
import unicodedata
from difflib import SequenceMatcher
import streamlit as st

try:
    THE_ODDS_API_KEY = st.secrets["THE_ODDS_API_KEY"]
except Exception:
    THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY", "4725b4a69b90b1310a23134c58f3de9c")

def limpiar_nombre(texto):
    if not texto: return ""
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8')
    return texto.lower().strip()

def son_similares(a, b, umbral=0.35):
    if not a or not b: return False
    a_limpio = limpiar_nombre(a)
    b_limpio = limpiar_nombre(b)
    if a_limpio in b_limpio or b_limpio in a_limpio: return True
    return SequenceMatcher(None, a_limpio, b_limpio).ratio() > umbral

# ---------------------------------------------------------
# EL TRUCO: GUARDAMOS LA RESPUESTA EN MEMORIA POR 6 HORAS
# ---------------------------------------------------------
@st.cache_data(ttl=21600) 
def descargar_cuotas_liga(league_id=262):
    """Hace una sola llamada a la API y guarda todos los partidos en RAM"""
    sport_key = "soccer_mexico_ligamx" if league_id == 262 else "soccer_usa_mls"
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    
    if not THE_ODDS_API_KEY:
        return []
        
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us,eu,uk", 
        "markets": "h2h,totals", 
        "oddsFormat": "decimal"
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    
    return []

def obtener_cuotas_partido(local, visita, league_id=262):
    """Busca el partido específico dentro de los datos ya descargados en memoria"""
    
    # Cuotas base limpias sin números falsos
    cuotas_limpias = {
        "Linea_Goles": 2.5, 
        "Linea_Corners": 9.5, 
        "Linea_Tarjetas": 4.5
    }
    
    # Obtenemos los datos de la memoria (Costo de API = 0 si ya está en caché)
    datos_liga = descargar_cuotas_liga(league_id)
    
    if not datos_liga:
        return cuotas_limpias
            
    encontrado = False
    for partido in datos_liga:
        home_api = partido.get("home_team", "")
        away_api = partido.get("away_team", "")
        
        if son_similares(local, home_api) or son_similares(visita, away_api):
            bookmakers = partido.get("bookmakers", [])
            if not bookmakers: continue
            
            for bookmaker in bookmakers:
                mercados = bookmaker.get("markets", [])
                for m in mercados:
                    key = m.get("key")
                    outcomes = m.get("outcomes", [])
                    
                    if key == "h2h":
                        for out in outcomes:
                            name = out.get("name")
                            if name == home_api: cuotas_limpias["1"] = float(out.get("price"))
                            elif name == "Draw": cuotas_limpias["X"] = float(out.get("price"))
                            else: cuotas_limpias["2"] = float(out.get("price"))
                            encontrado = True
                            
                    elif key == "totals":
                        for out in outcomes:
                            if "point" in out: cuotas_limpias["Linea_Goles"] = float(out["point"])
                            name = out.get("name")
                            if name == "Over": cuotas_limpias["Over_Goles"] = float(out.get("price"))
                            elif name == "Under": cuotas_limpias["Under_Goles"] = float(out.get("price"))
                            encontrado = True
            if encontrado:
                break
                
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

def analizar_apuestas(resultados_montecarlo, local, visita, cuotas_personalizadas=None, lineas_default=None):
    if cuotas_personalizadas and len(cuotas_personalizadas) > 0:
        cuotas = cuotas_personalizadas
        l_goles = lineas_default.get("Linea_Goles", 2.5) if lineas_default else 2.5
        l_corners = lineas_default.get("Linea_Corners", 9.5) if lineas_default else 9.5
        l_tarjetas = lineas_default.get("Linea_Tarjetas", 4.5) if lineas_default else 4.5
    else:
        cuotas = obtener_cuotas_partido(local, visita)
        if not cuotas: return pd.DataFrame() 
        l_goles = cuotas.get("Linea_Goles", 2.5)
        l_corners = cuotas.get("Linea_Corners", 9.5)
        l_tarjetas = cuotas.get("Linea_Tarjetas", 4.5)
        
    analisis = []
    mercados_a_evaluar = [
        ("Gana Local", resultados_montecarlo["Resultado_1X2"]["Gana Local"], "1"),
        ("Empate", resultados_montecarlo["Resultado_1X2"]["Empate"], "X"),
        ("Gana Visita", resultados_montecarlo["Resultado_1X2"]["Gana Visita"], "2"),
        (f"Over {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"].get(f"Over {l_goles}", 50.0), "Over_Goles"),
        (f"Under {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"].get(f"Under {l_goles}", 50.0), "Under_Goles"),
        (f"Over {l_corners} Corners", resultados_montecarlo["Corners_Totales"].get(f"Over {l_corners} Corners", 50.0), "Over_Corners"),
        (f"Under {l_corners} Corners", resultados_montecarlo["Corners_Totales"].get(f"Under {l_corners} Corners", 50.0), "Under_Corners"),
        (f"Over {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"].get(f"Over {l_tarjetas} Tarjetas", 50.0), "Over_Tarjetas"),
        (f"Under {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"].get(f"Under {l_tarjetas} Tarjetas", 50.0), "Under_Tarjetas")
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
