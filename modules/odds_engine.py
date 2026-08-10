import os
import requests
import pandas as pd

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def obtener_cuotas_partido(fixture_id, bookmaker_id=8):
    """Descarga las cuotas de Playdoit (Proxy Bet365) para un partido y detecta las líneas dinámicas."""
    url = f"{BASE_URL}/odds"
    response = requests.get(url, headers=HEADERS, params={"fixture": fixture_id, "bookmaker": bookmaker_id})
    if response.status_code != 200: return None
    
    datos = response.json().get("response", [])
    if not datos: return None
        
    mercados = datos[0]["bookmakers"][0]["bets"]
    cuotas_limpias = {"Linea_Goles": 2.5, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
    
    for mercado in mercados:
        nombre = mercado["name"]
        
        # --- 1. GANADOR DEL PARTIDO (1X2) ---
        if nombre == "Match Winner":
            for val in mercado["values"]:
                if val["value"] == "Home": cuotas_limpias["1"] = float(val["odd"])
                if val["value"] == "Draw": cuotas_limpias["X"] = float(val["odd"])
                if val["value"] == "Away": cuotas_limpias["2"] = float(val["odd"])
                
        # --- 2. GOLES (Over/Under) ---
        elif nombre == "Goals Over/Under" and "Over_Goles" not in cuotas_limpias:
            linea = 2.5
            for val in mercado["values"]:
                if "Over" in val["value"]:
                    linea = float(val["value"].replace("Over ", ""))
                    break
            cuotas_limpias["Linea_Goles"] = linea
            for val in mercado["values"]:
                if val["value"] == f"Over {linea}": cuotas_limpias["Over_Goles"] = float(val["odd"])
                if val["value"] == f"Under {linea}": cuotas_limpias["Under_Goles"] = float(val["odd"])
                
        # --- 3. CORNERS ---
        elif nombre in ["Corners Over Under", "Corners"] and "Over_Corners" not in cuotas_limpias:
            linea = 9.5
            for val in mercado["values"]:
                if "Over" in val["value"]:
                    linea = float(val["value"].replace("Over ", ""))
                    break
            cuotas_limpias["Linea_Corners"] = linea
            for val in mercado["values"]:
                if val["value"] == f"Over {linea}": cuotas_limpias["Over_Corners"] = float(val["odd"])
                if val["value"] == f"Under {linea}": cuotas_limpias["Under_Corners"] = float(val["odd"])
        
        # --- 4. TARJETAS ---
        elif nombre in ["Cards Over/Under", "Cards"] and "Over_Tarjetas" not in cuotas_limpias:
            linea = 4.5
            for val in mercado["values"]:
                if "Over" in val["value"]:
                    linea = float(val["value"].replace("Over ", ""))
                    break
            cuotas_limpias["Linea_Tarjetas"] = linea
            for val in mercado["values"]:
                if val["value"] == f"Over {linea}": cuotas_limpias["Over_Tarjetas"] = float(val["odd"])
                if val["value"] == f"Under {linea}": cuotas_limpias["Under_Tarjetas"] = float(val["odd"])
                    
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

def analizar_apuestas(resultados_montecarlo, fixture_id, cuotas_personalizadas=None, lineas_default=None):
    """Une las predicciones con cuotas automáticas o inyectadas manualmente y usa líneas dinámicas."""
    if cuotas_personalizadas and len(cuotas_personalizadas) > 0:
        cuotas = cuotas_personalizadas
        l_goles = lineas_default.get("Linea_Goles", 2.5) if lineas_default else 2.5
        l_corners = lineas_default.get("Linea_Corners", 9.5) if lineas_default else 9.5
        l_tarjetas = lineas_default.get("Linea_Tarjetas", 4.5) if lineas_default else 4.5
    else:
        cuotas = obtener_cuotas_partido(fixture_id)
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
