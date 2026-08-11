import os
import requests
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
    """Consulta cuotas reales directamente desde The Odds API."""
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
        "regions": "mx,us",
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
                    # Extraer bookmakers (ej. Caliente o Bet365 si están disponibles)
                    bookmakers = partido.get("bookmakers", [])
                    if bookmakers:
                        bookmaker = bookmakers[0] # Tomamos la primera casa disponible
                        markets = bookmaker.get("markets", [])
                        
                        for market in markets:
                            if market.get("key") == "h2h":
                                for outcome in market.get("outcomes", []):
                                    name = outcome.get("name")
                                    price = outcome.get("price")
                                    if son_similares(name, local): cuotas_limpias["1"] = price
                                    elif son_similares(name, visita): cuotas_limpias["2"] = price
                                    elif "draw" in name.lower() or "empate" in name.lower(): cuotas_limpias["X"] = price
                                    
                            elif market.get("key") == "totals":
                                for outcome in market.get("outcomes", []):
                                    if outcome.get("name") == "Over":
                                        cuotas_limpias["Linea_Goles"] = outcome.get("point", 2.5)
                                        cuotas_limpias["Over_Goles"] = outcome.get("price", 1.90)
                                    elif outcome.get("name") == "Under":
                                        cuotas_limpias["Under_Goles"] = outcome.get("price", 1.90)
                    break
    except Exception as e:
        print(f"Error consultando The Odds API: {e}")
        
    return cuotas_limpias
