import os
import requests
import pandas as pd

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def obtener_cuotas_partido(fixture_id):
    """Descarga cuotas reales desde API-Football usando el fixture_id del partido."""
    if not fixture_id or not API_KEY:
        return {}
        
    url = f"{BASE_URL}/odds"
    cuotas_limpias = {}
    
    # Lista de bookmakers comunes a probar (8 = Bet365 / Playdoit, 6 = Bwin, etc.)
    bookmakers_a_probar = [8, 6, 11, 1]
    
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
                        
                # --- 2. GOLES (Over/Under) ---
                elif nombre == "Goals Over/Under":
                    for val in mercado.get("values", []):
                        if val.get("value") == "Over 2.5": cuotas_limpias["Over 2.5"] = float(val.get("odd", 0))
                        elif val.get("value") == "Under 2.5": cuotas_limpias["Under 2.5"] = float(val.get("odd", 0))
                        
                # --- 3. CORNERS ---
                elif nombre in ["Corners Over Under", "Corners", "Total Corners"]:
                    for val in mercado.get("values", []):
                        if val.get("value") == "Over 9.5": cuotas_limpias["Over 9.5 Corners"] = float(val.get("odd", 0))
                        elif val.get("value") == "Under 9.5": cuotas_limpias["Under 9.5 Corners"] = float(val.get("odd", 0))
                        
                # --- 4. TARJETAS ---
                elif nombre in ["Cards Over/Under", "Cards", "Total Cards"]:
                    for val in mercado.get("values", []):
                        if val.get("value") == "Over 4.5": cuotas_limpias["Over 4.5 Tarjetas"] = float(val.get("odd", 0))
                        elif val.get("value") == "Under 4.5": cuotas_limpias["Under 4.5 Tarjetas"] = float(val.get("odd", 0))
            
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
        
    analisis = []
    mercados_a_evaluar = [
        ("Gana Local", resultados_montecarlo["Resultado_1X2"]["Gana Local"], "1"),
        ("Empate", resultados_montecarlo["Resultado_1X2"]["Empate"], "X"),
        ("Gana Visita", resultados_montecarlo["Resultado_1X2"]["Gana Visita"], "2"),
        ("Over 2.5 Goles", resultados_montecarlo["Goles_Over_Under"]["Over 2.5"], "Over 2.5"),
        ("Under 2.5 Goles", resultados_montecarlo["Goles_Over_Under"]["Under 2.5"], "Under 2.5"),
        ("Over 9.5 Corners", resultados_montecarlo["Corners_Totales"]["Over 9.5 Corners"], "Over 9.5 Corners"),
        ("Under 9.5 Corners", resultados_montecarlo["Corners_Totales"]["Under 9.5 Corners"], "Under 9.5 Corners"),
        ("Over 4.5 Tarjetas", resultados_montecarlo["Tarjetas_Totales"]["Over 4.5 Tarjetas"], "Over 4.5 Tarjetas"),
        ("Under 4.5 Tarjetas", resultados_montecarlo["Tarjetas_Totales"]["Under 4.5 Tarjetas"], "Under 4.5 Tarjetas")
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
