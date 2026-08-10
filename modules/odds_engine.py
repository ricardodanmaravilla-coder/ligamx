import os
import requests
import pandas as pd
import unicodedata
from difflib import SequenceMatcher
import streamlit as st

# Extrae la API Key desde los Secrets de Streamlit o las variables locales
try:
    THE_ODDS_API_KEY = st.secrets["THE_ODDS_API_KEY"]
except Exception:
    THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY", "")

def limpiar_nombre(texto):
    if not texto: return ""
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8')
    return texto.lower().strip()

def son_similares(a, b, umbral=0.45):
    if not a or not b: return False
    a_limpio = limpiar_nombre(a)
    b_limpio = limpiar_nombre(b)
    if a_limpio in b_limpio or b_limpio in a_limpio: return True
    return SequenceMatcher(None, a_limpio, b_limpio).ratio() > umbral

def obtener_cuotas_partido(local, visita, league_id=262):
    sport_key = "soccer_mexico_ligamx" if league_id == 262 else "soccer_usa_mls"
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us,eu,uk,au", # Ampliamos las regiones para buscar más casas de apuestas
        "markets": "h2h,totals", 
        "oddsFormat": "decimal"
    }
    
    cuotas_limpias = {"Linea_Goles": 2.5, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
    
    if not THE_ODDS_API_KEY:
        st.error("❌ THE_ODDS_API_KEY no detectada en los Secrets. Las cuotas automáticas no funcionarán.")
        return cuotas_limpias
        
    try:
        res = requests.get(url, params=params, timeout=10)
        
        if res.status_code != 200: 
            return cuotas_limpias
        
        datos = res.json()
        if not datos:
            return cuotas_limpias
            
        for partido in datos:
            home_api = partido.get("home_team", "")
            away_api = partido.get("away_team", "")
            
            if son_similares(local, home_api) or son_similares(visita, away_api):
                bookmakers = partido.get("bookmakers", [])
                
                if not bookmakers: 
                    continue
                
                # Buscamos en las casas de apuestas hasta encontrar una que tenga tanto 1X2 como Totales
                for bookmaker in bookmakers:
                    mercados = bookmaker.get("markets", [])
                    encontro_goles = False
                    
                    for m in mercados:
                        key = m.get("key")
                        outcomes = m.get("outcomes", [])
                        
                        if key == "h2h" and "1" not in cuotas_limpias:
                            for out in outcomes:
                                name = out.get("name")
                                if name == home_api: cuotas_limpias["1"] = float(out.get("price"))
                                elif name == "Draw": cuotas_limpias["X"] = float(out.get("price"))
                                else: cuotas_limpias["2"] = float(out.get("price"))
                                
                        elif key == "totals":
                            for out in outcomes:
                                if "point" in out:
                                    cuotas_limpias["Linea_Goles"] = float(out["point"])
                                name = out.get("name")
                                if name == "Over": 
                                    cuotas_limpias["Over_Goles"] = float(out.get("price"))
                                    encontro_goles = True
                                elif name == "Under": 
                                    cuotas_limpias["Under_Goles"] = float(out.get("price"))
                                    encontro_goles = True
                                    
                    # Si ya logramos capturar al menos el ganador y los goles, podemos detener la búsqueda en esta casa
                    if "1" in cuotas_limpias and encontro_goles:
                        break
                break 
                
    except Exception as e:
        st.error(f"⚠️ Error procesando The Odds API: {e}")
        
    return cuotas_limpias
        
        datos = res.json()
        if not datos:
            st.info("ℹ️ The Odds API NO tiene partidos publicados para la Liga MX en este momento (los casinos aún no abren las líneas).")
            return cuotas_limpias
            
        match_encontrado = False
        
        for partido in datos:
            home_api = partido.get("home_team", "")
            away_api = partido.get("away_team", "")
            
            if son_similares(local, home_api) or son_similares(visita, away_api):
                match_encontrado = True
                bookmakers = partido.get("bookmakers", [])
                
                if not bookmakers: 
                    st.warning(f"⚠️ Partido encontrado ({home_api} vs {away_api}), pero los casinos aún no suben las cuotas.")
                    continue
                
                st.success(f"✅ ¡Cuotas obtenidas con éxito para {local} vs {visita}!")
                mercados = bookmakers[0].get("markets", [])
                
                for m in mercados:
                    key = m.get("key")
                    outcomes = m.get("outcomes", [])
                    
                    if key == "h2h":
                        for out in outcomes:
                            name = out.get("name")
                            if name == home_api: cuotas_limpias["1"] = float(out.get("price"))
                            elif name == "Draw": cuotas_limpias["X"] = float(out.get("price"))
                            else: cuotas_limpias["2"] = float(out.get("price"))
                            
                    elif key == "totals":
                        for out in outcomes:
                            if "point" in out:
                                cuotas_limpias["Linea_Goles"] = float(out["point"])
                            name = out.get("name")
                            if name == "Over": cuotas_limpias["Over_Goles"] = float(out.get("price"))
                            elif name == "Under": cuotas_limpias["Under_Goles"] = float(out.get("price"))
                break 
                
        if not match_encontrado:
            st.warning(f"🔍 The Odds API tiene {len(datos)} partidos, pero ninguno coincide con: {local} vs {visita}.")
            with st.expander("Ver los partidos que The Odds API sí encontró (Para diagnóstico):"):
                for d in datos:
                    st.write(f"- {d.get('home_team')} vs {d.get('away_team')}")
                
    except Exception as e:
        st.error(f"⚠️ Error procesando The Odds API: {e}")
        
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
