import os
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

def obtener_cuotas_partido(local, visita, league_id=262):
    """Lee las cuotas pre-cargadas en el CSV generado por la automatización."""
    cuotas_limpias = {
        "1": 2.10, "X": 3.40, "2": 3.20,
        "Linea_Goles": 2.5, "Over_Goles": 1.90, "Under_Goles": 1.90,
        "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5
    }
    
    ruta_csv = "data/cuotas_jornada.csv"
    if not os.path.exists(ruta_csv):
        return cuotas_limpias
        
    try:
        df = pd.read_csv(ruta_csv)
        for _, fila in df.iterrows():
            f_local = str(fila.get("Local", ""))
            f_visita = str(fila.get("Visitante", ""))
            
            if son_similares(local, f_local) or son_similares(visita, f_visita):
                cuotas_limpias["1"] = float(fila.get("1", 2.10))
                cuotas_limpias["X"] = float(fila.get("X", 3.40))
                cuotas_limpias["2"] = float(fila.get("2", 3.20))
                cuotas_limpias["Linea_Goles"] = float(fila.get("Linea_Goles", 2.5))
                cuotas_limpias["Over_Goles"] = float(fila.get("Over_Goles", 1.90))
                break
    except Exception as e:
        print(f"Error leyendo CSV local de cuotas: {e}")
        
    return cuotas_limpias
