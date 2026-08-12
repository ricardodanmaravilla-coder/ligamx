import os
import requests
import pandas as pd

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def obtener_cuotas_partido(fixture_id):
    """Descarga cuotas reales desde API-Football usando el fixture_id del partido y extrae líneas dinámicas."""
    if not fixture_id or not API_KEY:
        return {}
        
    url = f"{BASE_URL}/odds"
    cuotas_limpias = {}
    
    # Lista de bookmakers comunes a probar (8 = Bet365 / Playdoit, 6 = Bwin, etc.)
    bookmakers_a_probar = [8, 6, 11, 1]
    
    linea_goles = "2.5"
    linea_corners = "9.5"
    linea_tarjetas = "4.5"
    
    for bookmaker_id in bookmakers_a_probar:
        try:
            response = requests.get(url, headers=HEADERS, params={"fixture": fixture_id, "bookmaker": bookmaker_id}, timeout=10)
            if response.status_code != 200:
                continue
                
            data = response.json().get("response", [])
            if not data:
                continue
                
            bookmakers_data = data[0].get("bookmakers", [])
            if not bookmakers_data:
                continue
                
            mercados = bookmakers_data[0].get("bets", [])
            
            for mercado in mercados:
                nombre = mercado.get("name", "")
                
                # --- 1. GANADOR DEL PARTIDO (1X2) ---
                if nombre == "Match Winner":
                    for val in mercado.get("values", []):
                        if val.get("value") == "Home": cuotas_limpias["1"] = float(val.get("odd", 0))
                        elif val.get("value") == "Draw": cuotas_limpias["X"] = float(val.get("odd", 0))
                        elif val.get("value") == "Away": cuotas_limpias["2"] = float(val.get("odd", 0))
                        
                # --- 2. GOLES (Over/Under dinámico) ---
                elif nombre == "Goals Over/Under":
                    mejor_dif = 999
                    for val in mercado.get("values", []):
                        if "Over" in val.get("value", ""):
                            dif = abs(float(val.get("odd", 0)) - 1.90)
                            if dif < mejor_dif:
                                mejor_dif = dif
                                linea_goles = val.get("value", "").replace("Over ", "")
                                
                    for val in mercado.get("values", []):
                        if val.get("value") == f"Over {linea_goles}": cuotas_limpias[f"Over {linea_goles}"] = float(val.get("odd", 0))
                        elif val.get("value") == f"Under {linea_goles}": cuotas_limpias[f"Under {linea_goles}"] = float(val.get("odd", 0))
                        
                # --- 3. CORNERS (Over/Under dinámico) ---
                elif nombre in ["Corners Over Under", "Corners", "Total Corners"]:
                    mejor_dif = 999
                    for val in mercado.get("values", []):
                        if "Over" in val.get("value", ""):
                            dif = abs(float(val.get("odd", 0)) - 1.90)
                            if dif < mejor_dif:
                                mejor_dif = dif
                                linea_corners = val.get("value", "").replace("Over ", "")
                                
                    for val in mercado.get("values", []):
                        if val.get("value") == f"Over {linea_corners}": cuotas_limpias[f"Over {linea_corners} Corners"] = float(val.get("odd", 0))
                        elif val.get("value") == f"Under {linea_corners}": cuotas_limpias[f"Under {linea_corners} Corners"] = float(val.get("odd", 0))
                        
                # --- 4. TARJETAS (Over/Under dinámico) ---
                elif nombre in ["Cards Over/Under", "Cards", "Total Cards"]:
                    mejor_dif = 999
                    for val in mercado.get("values", []):
                        if "Over" in val.get("value", ""):
                            dif = abs(float(val.get("odd", 0)) - 1.90)
                            if dif < mejor_dif:
                                mejor_dif = dif
                                linea_tarjetas = val.get("value", "").replace("Over ", "")
                                
                    for val in mercado.get("values", []):
                        if val.get("value") == f"Over {linea_tarjetas}": cuotas_limpias[f"Over {linea_tarjetas} Tarjetas"] = float(val.get("odd", 0))
                        elif val.get("value") == f"Under {linea_tarjetas}": cuotas_limpias[f"Under {linea_tarjetas} Tarjetas"] = float(val.get("odd", 0))
            
            # Guardamos las líneas detectadas en el diccionario de cuotas limpias
            cuotas_limpias["linea_goles_detectada"] = linea_goles
            cuotas_limpias["linea_corners_detectada"] = linea_corners
            cuotas_limpias["linea_tarjetas_detectada"] = linea_tarjetas

            # Si logramos extraer al menos el 1X2, detenemos la búsqueda en otras casas
            if "1" in cuotas_limpias and "X" in cuotas_limpias and "2" in cuotas_limpias:
                break
                
        except Exception as e:
            print(f"Error consultando cuotas para fixture {fixture_id} con bookmaker {bookmaker_id}: {e}")
            
    return cuotas_limpias

def evaluar_mercado_avanzado(probabilidad_modelo_pct, cuota_casa):
    """Evalúa usando EV, Criterio de Kelly Fraccional y Umbrales de Seguridad."""
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

def analizar_apuestas(resultados_montecarlo, fixture_id, cuotas_personalizadas=None):
    """Une las predicciones con cuotas automáticas de API-Football o inyectadas manualmente."""
    if cuotas_personalizadas and len(cuotas_personalizadas) > 0:
        cuotas = cuotas_personalizadas
    else:
        cuotas = obtener_cuotas_partido(fixture_id)
        
    if not cuotas: 
        cuotas = {}
        
    # Extracción de líneas dinámicas reales detectadas
    l_goles = cuotas.get("linea_goles_detectada", "2.5")
    l_corners = cuotas.get("linea_corners_detectada", "9.5")
    l_tarjetas = cuotas.get("linea_tarjetas_detectada", "4.5")
        
    analisis = []
    mercados_a_evaluar = [
        ("Gana Local", resultados_montecarlo["Resultado_1X2"]["Gana Local"], "1"),
        ("Empate", resultados_montecarlo["Resultado_1X2"]["Empate"], "X"),
        ("Gana Visita", resultados_montecarlo["Resultado_1X2"]["Gana Visita"], "2"),
        (f"Over {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"].get(f"Over {l_goles}", 0), f"Over {l_goles}"),
        (f"Under {l_goles} Goles", resultados_montecarlo["Goles_Over_Under"].get(f"Under {l_goles}", 0), f"Under {l_goles}"),
        (f"Over {l_corners} Corners", resultados_montecarlo["Corners_Totales"].get(f"Over {l_corners} Corners", 0), f"Over {l_corners} Corners"),
        (f"Under {l_corners} Corners", resultados_montecarlo["Corners_Totales"].get(f"Under {l_corners} Corners", 0), f"Under {l_corners} Corners"),
        (f"Over {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"].get(f"Over {l_tarjetas} Tarjetas", 0), f"Over {l_tarjetas} Tarjetas"),
        (f"Under {l_tarjetas} Tarjetas", resultados_montecarlo["Tarjetas_Totales"].get(f"Under {l_tarjetas} Tarjetas", 0), f"Under {l_tarjetas} Tarjetas")
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
