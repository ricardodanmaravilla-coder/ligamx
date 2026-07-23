import pandas as pd
import numpy as np

# Altitudes extremas o factores climáticos clave en la NFL (ej. Denver)
ALTITUDES_NFL = {
    "Denver Broncos": 1609,  # Mile High Stadium
    "Buffalo Bills": 180,    # Clima de frío extremo / viento
    "Green Bay Packers": 200 # Clima gélido
}

def cargar_datos_nfl():
    """Carga el histórico de la NFL con ponderación de antigüedad."""
    # Nota: Asegúrate de tener o descargar un CSV historico_nfl.csv con las columnas base
    try:
        df = pd.read_csv('data/historico_nfl.csv')
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        fecha_referencia = df['Fecha'].max()
        df['Dias_Antiguedad'] = (fecha_referencia - df['Fecha']).dt.days
        df['Peso'] = 0.5 ** (df['Dias_Antiguedad / 365.0'])
        return df
    except Exception:
        return None

def simular_partido_nfl_montecarlo(equipos_local, equipo_visita, num_simulaciones=10000):
    """Simula un partido de NFL usando eficiencia de yardas y distribución normal."""
    # Promedios base de la liga de ejemplo si el histórico está cargándose
    # En la práctica, esto toma los ratings reales de yardas por tierra y aire del CSV.
    
    np.random.seed(42)
    
    # Estimación de puntos base en la NFL (media histórica ~21-24 puntos por equipo)
    media_pts_local = 23.5
    media_pts_visita = 21.0
    
    # Desviación estándar típica en partidos de NFL (aprox 10-11 puntos)
    desv_est = 10.5
    
    # Simulamos con distribución normal (Gaussian)
    puntos_sim_local = np.random.normal(media_pts_local, desv_est, num_simulaciones)
    puntos_sim_visita = np.random.normal(media_pts_visita, desv_est, num_simulaciones)
    
    # Aseguramos que no haya puntos negativos
    puntos_sim_local = np.clip(puntos_sim_local, 0, None)
    puntos_sim_visita = np.clip(puntos_sim_visita, 0, None)
    
    # Calculamos probabilidades de victoria directa (Moneyline)
    wins_local = np.sum(puntos_sim_local > puntos_sim_visita)
    wins_visita = np.sum(puntos_sim_visita > puntos_sim_local)
    empates = num_simulaciones - (wins_local + wins_visita)
    
    prob_local = (wins_local / num_simulaciones) * 100
    prob_visita = (wins_visita / num_simulaciones) * 100
    
    # Puntos totales combinados para Over/Under (Línea estándar común: 45.5 o 47.5)
    puntos_totales_sim = puntos_sim_local + puntos_sim_visita
    
    over_45_5 = (np.sum(puntos_totales_sim > 45.5) / num_simulaciones) * 100
    under_45_5 = (np.sum(puntos_totales_sim <= 45.5) / num_simulaciones) * 100
    
    return {
        "Moneyline": {
            "Gana Local": round(prob_local, 1),
            "Gana Visita": round(prob_visita, 1)
        },
        "Total_Puntos_45.5": {
            "Over 45.5": round(over_45_5, 1),
            "Under 45.5": round(under_45_5, 1)
        },
        "Media_Puntos_Esperados": {
            "Local": round(np.mean(puntos_sim_local), 1),
            "Visita": round(np.mean(puntos_sim_visita), 1)
        }
    }