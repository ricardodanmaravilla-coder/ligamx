import os
import requests
import pandas as pd
import unicodedata
from difflib import SequenceMatcher

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

def obtener_cuotas_partido(local, visita, league_id="soccer_mexico_ligamx"):
    """Consulta The Odds API. Si no hay datos reales disponibles, devuelve ceros estrictos sin inventar nada."""
    cuotas_limpias = {
        "1": 0.0, "X": 0.0, "2": 0.0,
        "Linea_Goles": 2.5, "Over_Goles": 0.0, "Under_Goles": 0.0,
        "Linea_Corners": 9.5, "Over_Corners": 0.0, "Under_Corners": 0.0,
        "Linea_Tarjetas": 4.5, "Over_Tarjetas": 0.0, "Under_Tarjetas": 0.0
    }
    
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return cuotas_limpias
        
    url = f"https://api.the-odds-api.com/v4/sports/{league_id}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "mx,us,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for partido in data:
                home_team = partido.get("home_team", "")
                away_team = partido.get("away_team", "")
                
                if son_similares(local, home_team) and son_similares(visita, away_team):
                    bookmakers = partido.get("bookmakers", [])
                    if bookmakers:
                        bookmaker = bookmakers[0] 
                        markets = bookmaker.get("markets", [])
                        
                        for market in markets:
                            if market.get("key") == "h2h":
                                for outcome in market.get("outcomes", []):
                                    name = outcome.get("name")
                                    price = outcome.get("price")
                                    if son_similares(name, local): cuotas_limpias["1"] = float(price)
                                    elif son_similares(name, visita): cuotas_limpias["2"] = float(price)
                                    elif "draw" in name.lower() or "empate" in name.lower(): cuotas_limpias["X"] = float(price)
                                    
                            elif market.get("key") == "totals":
                                for outcome in market.get("outcomes", []):
                                    if outcome.get("name") == "Over":
                                        cuotas_limpias["Linea_Goles"] = float(outcome.get("point", 2.5))
                                        cuotas_limpias["Over_Goles"] = float(outcome.get("price", 0.0))
                                    elif outcome.get("name") == "Under":
                                        cuotas_limpias["Under_Goles"] = float(outcome.get("price", 0.0))
                    break
    except Exception as e:
        print(f"Error consultando The Odds API: {e}")
        
    return cuotas_limpias

def analizar_apuestas(resultados, local, visita, cuotas_personalizadas=None, lineas_default=None):
    """Procesa el valor esperado (EV) únicamente con cuotas reales de la API. Si la cuota es 0.0, omite o marca sin mercado."""
    apuestas_lista = []
    
    if not lineas_default:
        lineas_default = {"Linea_Goles": 2.5, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
        
    l_goles = lineas_default.get("Linea_Goles", 2.5)
    
    cuotas = cuotas_personalizadas if cuotas_personalizadas else obtener_cuotas_partido(local, visita)
    
    # 1. Mercado 1X2
    prob_1x2 = resultados.get('Resultado_1X2', {})
    mercados_1x2 = [
        ("Gana Local", f"Victoria {local}", prob_1x2.get('Gana Local', 0) / 100.0, cuotas.get('1', 0.0)),
        ("Empate", "Empate", prob_1x2.get('Empate', 0) / 100.0, cuotas.get('X', 0.0)),
        ("Gana Visita", f"Victoria {visita}", prob_1x2.get('Gana Visita', 0) / 100.0, cuotas.get('2', 0.0))
    ]
    
    for key_m, nombre_m, prob_real, cuota in mercados_1x2:
        if cuota > 1.0 and prob_real > 0:
            ev = (prob_real * cuota) - 1.0
            kelly = max(0.0, ((prob_real * cuota - 1) / (cuota - 1))) * 100 if cuota > 1 else 0.0
            
            veredicto = "Paso"
            if ev > 0.05: veredicto = "🔥 Valor Alto (EV+)"
            elif ev > 0.0: veredicto = "✅ Valor Moderado"
            
            apuestas_lista.append({
                "Mercado": nombre_m,
                "Prob. Modelo": f"{round(prob_real * 100, 1)}%",
                "Cuota": f"{cuota:.2f}",
                "EV (%)": f"{round(ev * 100, 2)}%",
                "Kelly (%)": f"{round(kelly, 2)}%",
                "Veredicto": veredicto
            })
        else:
            apuestas_lista.append({
                "Mercado": nombre_m,
                "Prob. Modelo": f"{round(prob_real * 100, 1)}%",
                "Cuota": "No disponible",
                "EV (%)": "N/A",
                "Kelly (%)": "N/A",
                "Veredicto": "❌ Sin cuota en API"
            })

    # 2. Mercado Goles Over
    goles_ou = resultados.get('Goles_Over_Under', {})
    prob_over_goles = goles_ou.get(f'Over {l_goles}', 50.0) / 100.0
    cuota_over_goles = cuotas.get('Over_Goles', 0.0)
    
    if cuota_over_goles > 1.0:
        ev_og = (prob_over_goles * cuota_over_goles) - 1.0
        kelly_og = max(0.0, ((prob_over_goles * cuota_over_goles - 1) / (cuota_over_goles - 1))) * 100 if cuota_over_goles > 1 else 0.0
        ver_og = "🔥 Valor Alto (EV+)" if ev_og > 0.05 else ("✅ Valor Moderado" if ev_og > 0.0 else "Paso")
        
        apuestas_lista.append({
            "Mercado": f"Over {l_goles} Goles",
            "Prob. Modelo": f"{round(prob_over_goles * 100, 1)}%",
            "Cuota": f"{cuota_over_goles:.2f}",
            "EV (%)": f"{round(ev_og * 100, 2)}%",
            "Kelly (%)": f"{round(kelly_og, 2)}%",
            "Veredicto": ver_og
        })
    else:
        apuestas_lista.append({
            "Mercado": f"Over {l_goles} Goles",
            "Prob. Modelo": f"{round(prob_over_goles * 100, 1)}%",
            "Cuota": "No disponible",
            "EV (%)": "N/A",
            "Kelly (%)": "N/A",
            "Veredicto": "❌ Sin cuota en API"
        })

    return pd.DataFrame(apuestas_lista)
