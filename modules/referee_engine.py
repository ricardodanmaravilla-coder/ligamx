import pandas as pd

def calcular_perfiles_arbitros():
    """Calcula matemáticamente el factor de tarjetas real de cada árbitro basándose en el historial."""
    try:
        df = pd.read_csv('data/historico_ligamx_completo.csv')
        
        # Verificamos si la columna de árbitro existe en el CSV
        if 'Arbitro' not in df.columns:
            return {}
            
        # Limpiamos y sumamos las tarjetas totales del partido (Local + Visita)
        df['Tarjetas_Totales'] = (
            df['Amarillas_L'].fillna(0) + (df['Rojas_L'].fillna(0) * 2) + 
            df['Amarillas_V'].fillna(0) + (df['Rojas_V'].fillna(0) * 2)
        )
        
        # Promedio general de tarjetas en toda la liga para usarlo como base (1.0)
        media_liga_tarjetas = df['Tarjetas_Totales'].mean()
        if media_liga_tarjetas == 0:
            return {}

        # Agrupamos por árbitro para sacar su promedio personal de tarjetas por partido
        perfiles = {}
        grouped = df.groupby('Arbitro')['Tarjetas_Totales'].agg(['mean', 'count'])
        
        # Filtramos árbitros con al menos 3 partidos registrados para evitar datos falsos por muestra pequeña
        for arbitro, row in grouped[grouped['count'] >= 3].iterrows():
            if pd.notna(arbitro) and arbitro.strip() != "":
                promedio_arbitro = row['mean']
                # Factor = Promedio del árbitro / Promedio de la liga
                factor = promedio_arbitro / media_liga_tarjetas
                perfiles[arbitro.strip()] = round(factor, 3)
                
        return perfiles
    except Exception:
        return {}

def obtener_factor_arbitro(nombre_arbitro):
    """Obtiene el factor real calculado de la base de datos."""
    if not nombre_arbitro or nombre_arbitro == "Sin Asignar / Neutral":
        return 1.0
        
    perfiles = calcular_perfiles_arbitros()
    # Si el árbitro existe en el historial real, devuelve su factor matemático. Si es nuevo, retorna 1.0 neutro.
    return perfiles.get(nombre_arbitro, 1.0)

def obtener_lista_arbitros_reales():
    """Devuelve únicamente los árbitros que realmente tienen partidos registrados en el sistema."""
    perfiles = calcular_perfiles_arbitros()
    return sorted(list(perfiles.keys()))