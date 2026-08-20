import os
import requests
import pandas as pd
import datetime

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def _obtener_fallback_espn():
    """
    Función de rescate (Fallback) usando la API oculta de ESPN para Liga MX.
    Se activa solo si API-Football falla o no tiene datos.
    Nota: ESPN raramente provee córners o tarjetas, solo rescatará 1X2 y Goles.
    """
    url_espn = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
    cuotas_rescate = {}
    
    try:
        res = requests.get(url_espn, timeout=5)
        if res.status_code == 200:
            data = res.json()
            # Toma el primer evento disponible para buscar cuotas generales
            for event in data.get("events", []):
                competitions = event.get("competitions", [])
                if competitions:
                    odds = competitions[0].get("odds", [])
                    if odds:
                        # Extraer momios y convertirlos a formato de nuestro motor
                        cuotas_rescate["linea_goles_detectada"] = str(odds[0].get("overUnder", "2.5"))
                        # ESPN normalmente da cuotas en formato Americano o Decimal según la región
                        # Asumiremos que no siempre están limpias, pero rescatamos la línea O/U
                        
                        # No podemos asegurar el 1X2 exacto sin cruzar nombres de equipos,
                        # pero devolvemos las líneas base para que la interfaz no colapse.
                        return cuotas_rescate
    except Exception as e:
        print(f"Fallo en el respaldo de ESPN: {e}")
        
    return cuotas_rescate

def obtener_cuotas_partido(fixture_id):
    """Descarga cuotas reales desde API-Football. Si falla, activa el respaldo de ESPN."""
    cuotas_limpias = {}
    linea_goles = "2.5"
    linea_corners = "9.5"
    linea_tarjetas = "4.5"

    if not fixture_id or not API_KEY:
        # Activar respaldo de ESPN inmediatamente si no hay API Key
        respaldo = _obtener_fallback_espn()
        respaldo["linea_corners_detectada"] = linea_corners
        respaldo["linea_tarjetas_detectada"] = linea_tarjetas
        if not respaldo.get("linea_goles_detectada"):
            respaldo["linea_goles_detectada"] = linea_goles
        return respaldo
        
    url = f"{BASE_URL}/odds"
    bookmakers_a_probar = [8, 6, 11, 1] # Bet365, Bwin, 1xBet, 10Bet
    exito_api_principal = False
    
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
                
                if nombre == "Match Winner":
                    for val in mercado.get("values", []):
                        if val.get("value") == "Home": cuotas_limpias["1"] = float(val.get("odd", 0))
                        elif val.get("value") == "Draw": cuotas_limpias["X"] = float(val.get("odd", 0))
                        elif val.get("value") == "Away": cuotas_limpias["2"] = float(val.get("odd", 0))
                        
                elif nombre == "Goals Over/Under":
                    mejor_dif = 999
                    for val in mercado.get("values", []):
                        if "Over" in val.get("value", ""):
                            dif = abs(float(val.get("odd", 0)) - 1.90)
                            if dif < mejor_dif:
                                mejor_dif = dif
                                linea_goles = val.get("value", "").replace("Over ", "")
                                
                    for val in mercado.get("values", []):
                        if val.get("value") == f"Over {linea_goles}": 
                            cuotas_limpias[f"Over {linea_goles}"] = float(val.get("odd", 0))
                            cuotas_limpias[f"Over {linea_goles} Goles"] = float(val.get("odd", 0))
                        elif val.get("value") == f"Under {linea_goles}": 
                            cuotas_limpias[f"Under {linea_goles}"] = float(val.get("odd", 0))
                            cuotas_limpias[f"Under {linea_goles} Goles"] = float(val.get("odd", 0))
                        
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
            
            cuotas_limpias["linea_goles_detectada"] = linea_goles
            cuotas_limpias["linea_corners_detectada"] = linea_corners
            cuotas_limpias["linea_tarjetas_detectada"] = linea_tarjetas

            if "1" in cuotas_limpias and "X" in cuotas_limpias and "2" in cuotas_limpias:
                exito_api_principal = True
                break
                
        except Exception as e:
            print(f"Error consultando cuotas para fixture {fixture_id}: {e}")
            
    # --- ACTIVACIÓN DEL RESPALDO (FALLBACK) ---
    if not exito_api_principal or len(cuotas_limpias) < 4:
        print("API Principal falló o devolvió datos incompletos. Activando respaldo de ESPN...")
        respaldo = _obtener_fallback_espn()
        
        # Mezclamos lo que haya sobrevivido de la API principal con el respaldo
        cuotas_limpias["linea_goles_detectada"] = respaldo.get("linea_goles_detectada", linea_goles)
        cuotas_limpias["linea_corners_detectada"] = linea_corners # ESPN no da corners
        cuotas_limpias["linea_tarjetas_detectada"] = linea_tarjetas # ESPN no da tarjetas

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

def analizar_apuestas(resultados_montecarlo, fixture_id, cuotas_personalizadas=None):
    if cuotas_personalizadas and len(cuotas_personalizadas) > 0:
        cuotas = cuotas_personalizadas
    else:
        cuotas = obtener_cuotas_partido(fixture_id)
        
    if not cuotas: 
        cuotas = {}
        
    l_goles = cuotas.get("linea_goles_detectada", "2.5")
    l_corners = cuotas.get("linea_corners_detectada", "9.5")
    l_tarjetas = cuotas.get("linea_tarjetas_detectada", "4.5")
    
    if cuotas_personalizadas:
        for k in cuotas_personalizadas.keys():
            if "Over " in k and " Goles" in k:
                l_goles = k.replace("Over ", "").replace(" Goles", "").strip()
            elif "Over " in k and " Corners" in k:
                l_corners = k.replace("Over ", "").replace(" Corners", "").strip()
            elif "Over " in k and " Tarjetas" in k:
                l_tarjetas = k.replace("Over ", "").replace(" Tarjetas", "").strip()

    analisis = []
    
    prob_1x2 = resultados_montecarlo.get("Resultado_1X2", {})
    prob_goles = resultados_montecarlo.get("Goles_Over_Under", {})
    prob_corners = resultados_montecarlo.get("Corners_Totales", {})
    prob_tarjetas = resultados_montecarlo.get("Tarjetas_Totales", {})

    p_og = prob_goles.get(f"Over {l_goles}", prob_goles.get(f"Over {float(l_goles)}", prob_goles.get("Over 2.5", 0)))
    p_ug = prob_goles.get(f"Under {l_goles}", prob_goles.get(f"Under {float(l_goles)}", prob_goles.get("Under 2.5", 0)))
    
    p_oc = prob_corners.get(f"Over {l_corners} Corners", prob_corners.get(f"Over {float(l_corners)} Corners", prob_corners.get("Over 9.5 Corners", 0)))
    p_uc = prob_corners.get(f"Under {l_corners} Corners", prob_corners.get(f"Under {float(l_corners)} Corners", prob_corners.get("Under 9.5 Corners", 0)))
    
    p_ot = prob_tarjetas.get(f"Over {l_tarjetas} Tarjetas", prob_tarjetas.get(f"Over {float(l_tarjetas)} Tarjetas", prob_tarjetas.get("Over 4.5 Tarjetas", 0)))
    p_ut = prob_tarjetas.get(f"Under {l_tarjetas} Tarjetas", prob_tarjetas.get(f"Under {float(l_tarjetas)} Tarjetas", prob_tarjetas.get("Under 4.5 Tarjetas", 0)))

    mercados_a_evaluar = [
        ("Gana Local", prob_1x2.get("Gana Local", 0), "1"),
        ("Empate", prob_1x2.get("Empate", 0), "X"),
        ("Gana Visita", prob_1x2.get("Gana Visita", 0), "2"),
        (f"Over {l_goles} Goles", p_og, f"Over {l_goles} Goles"),
        (f"Under {l_goles} Goles", p_ug, f"Under {l_goles} Goles"),
        (f"Over {l_corners} Corners", p_oc, f"Over {l_corners} Corners"),
        (f"Under {l_corners} Corners", p_uc, f"Under {l_corners} Corners"),
        (f"Over {l_tarjetas} Tarjetas", p_ot, f"Over {l_tarjetas} Tarjetas"),
        (f"Under {l_tarjetas} Tarjetas", p_ut, f"Under {l_tarjetas} Tarjetas")
    ]
    
    for nombre_m, prob, llave_cuota in mercados_a_evaluar:
        cuota = cuotas.get(llave_cuota, cuotas.get(llave_cuota.replace(" Goles", "").replace(" Corners", "").replace(" Tarjetas", "")))
        if not cuota:
            cuota = cuotas.get(f"Over {l_goles}") if "Over" in nombre_m and "Goles" in nombre_m else cuotas.get(f"Under {l_goles}")
            
        if cuota is None or cuota == 0.0:
            analisis.append([nombre_m, f"{prob}%", "Sin Cuota", "N/A", "N/A", "0%", "🕒 Ingresa Cuota"])
            continue

        c_fmt, ev, veredicto, stake, riesgo = evaluar_mercado_avanzado(prob, cuota)
        ev_str = f"{ev:.1f}%"
        stake_str = f"{stake:.1f}%" if stake > 0 else "0%"
        
        analisis.append([nombre_m, f"{prob}%", c_fmt, ev_str, riesgo, stake_str, veredicto])
    
    return pd.DataFrame(analisis, columns=["Mercado", "Prob. Modelo", "Cuota", "EV", "Riesgo", "Stake (Bankroll)", "Veredicto"])
