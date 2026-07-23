import pandas as pd
import numpy as np

def aplicar_dixon_coles(lambda_l, lambda_v, prob_matriz, rho=-0.15):
    """
    Ajusta la matriz de probabilidades de marcadores exactos utilizando 
    la correlación de Dixon-Coles para inflar los empates de baja anotación.
    """
    # Evitar resultados negativos en tau limitando el impacto de rho
    if (1 - lambda_l * lambda_v * rho) < 0:
        rho = 0
        
    tau_0_0 = 1 - lambda_l * lambda_v * rho
    tau_1_0 = 1 + lambda_v * rho
    tau_0_1 = 1 + lambda_l * rho
    tau_1_1 = 1 - rho

    # prob_matriz es un dict con llave "x-y" y valor probabilistico
    if "0-0" in prob_matriz: prob_matriz["0-0"] *= tau_0_0
    if "1-0" in prob_matriz: prob_matriz["1-0"] *= tau_1_0
    if "0-1" in prob_matriz: prob_matriz["0-1"] *= tau_0_1
    if "1-1" in prob_matriz: prob_matriz["1-1"] *= tau_1_1

    # Normalizar la matriz para que vuelva a sumar 100%
    total = sum(prob_matriz.values())
    for k in prob_matriz.keys():
        prob_matriz[k] = (prob_matriz[k] / total)

    return prob_matriz

def calcular_lambdas_ewma_y_fuerza(df, equipo_local, equipo_visita, alpha=0.2):
    """
    Calcula las Lambdas combinando Fuerza del Oponente y Ponderación Temporal (EWMA).
    """
    if 'Fecha' in df.columns:
        df = df.sort_values(by='Fecha', ascending=True)

    col_l = 'Goles_L' if 'Goles_L' in df.columns else 'Puntos_L'
    col_v = 'Goles_V' if 'Goles_V' in df.columns else 'Puntos_V'
    
    media_goles_local_liga = df[col_l].mean() if not df.empty else 1.3
    media_goles_visita_liga = df[col_v].mean() if not df.empty else 1.0
    
    media_goles_local_liga = max(media_goles_local_liga, 0.01)
    media_goles_visita_liga = max(media_goles_visita_liga, 0.01)

    df_l_loc = df[df['Local'] == equipo_local]
    df_l_vis = df[df['Visitante'] == equipo_local]
    df_v_loc = df[df['Local'] == equipo_visita]
    df_v_vis = df[df['Visitante'] == equipo_visita]

    def ewma_ser(serie, a=alpha):
        if serie.empty or len(serie) < 2:
            return 1.2
        return float(serie.ewm(alpha=a).mean().iloc[-1])

    goles_fav_local = pd.concat([df_l_loc[col_l], df_l_vis[col_v]]) if not df_l_loc.empty or not df_l_vis.empty else pd.Series([1.2])
    goles_rec_local = pd.concat([df_l_loc[col_v], df_l_vis[col_l]]) if not df_l_loc.empty or not df_l_vis.empty else pd.Series([1.2])
    
    goles_fav_visita = pd.concat([df_v_loc[col_l], df_v_vis[col_v]]) if not df_v_loc.empty or not df_v_vis.empty else pd.Series([1.0])
    goles_rec_visita = pd.concat([df_v_loc[col_v], df_v_vis[col_l]]) if not df_v_loc.empty or not df_v_vis.empty else pd.Series([1.0])

    att_l = ewma_ser(goles_fav_local) / media_goles_local_liga
    def_l = ewma_ser(goles_rec_local) / media_goles_visita_liga
    
    att_v = ewma_ser(goles_fav_visita) / media_goles_visita_liga
    def_v = ewma_ser(goles_rec_visita) / media_goles_local_liga

    lambda_local = att_l * def_v * media_goles_local_liga
    lambda_visita = att_v * def_l * media_goles_visita_liga

    return max(lambda_local, 0.2), max(lambda_visita, 0.2)

def calcular_lambdas_ewma(df, equipo_local, equipo_visita, alpha=0.15):
    """
    Calcula las Lambdas utilizando Media Móvil Exponencial (EWMA).
    Los partidos más recientes tienen mayor peso (alpha) que los antiguos.
    """
    # Asegurar que el DataFrame esté ordenado por fecha si existe la columna
    if 'Fecha' in df.columns:
        df = df.sort_values(by='Fecha', ascending=True)

    col_l = 'Goles_L' if 'Goles_L' in df.columns else 'Puntos_L'
    col_v = 'Goles_V' if 'Goles_V' in df.columns else 'Puntos_V'
    
    # 1. Filtrar partidos de los equipos
    df_l_loc = df[df['Local'] == equipo_local]
    df_l_vis = df[df['Visitante'] == equipo_local]
    df_v_loc = df[df['Local'] == equipo_visita]
    df_v_vis = df[df['Visitante'] == equipo_visita]

    # Función interna para calcular promedio ponderado exponencial (EWMA)
    def ewma_ser(serie, a=alpha):
        if serie.empty:
            return 1.2 # Valor predeterminado de respaldo
        # .ewm(alpha=a).mean() calcula la media ponderada dando más peso al final de la serie
        return float(serie.ewm(alpha=a).mean().iloc[-1])

    # Unificar goles anotados y recibidos en orden cronológico para aplicar EWMA
    # (Para simplificar y mantener robustez, aplicamos EWMA a las series históricas de cada equipo)
    goles_fav_local = pd.concat([df_l_loc[col_l], df_l_vis[col_v]])
    goles_rec_local = pd.concat([df_l_loc[col_v], df_l_vis[col_l]])
    
    goles_fav_visita = pd.concat([df_v_loc[col_l], df_v_vis[col_v]])
    goles_rec_visita = pd.concat([df_v_loc[col_v], df_v_vis[col_l]])

    # Obtener la expectativa ponderada reciente
    lambda_l_calc = ewma_ser(goles_fav_local) * 0.5 + ewma_ser(goles_rec_visita) * 0.5
    lambda_v_calc = ewma_ser(goles_fav_visita) * 0.5 + ewma_ser(goles_rec_local) * 0.5

    # Acotar para evitar valores absurdos o ceros absolutos
    return max(lambda_l_calc, 0.2), max(lambda_v_calc, 0.2)

def simular_partido_montecarlo(equipo_local, equipo_visita, df_historico=None):
    """
    Ejecuta 10,000 iteraciones de Montecarlo reales para Goles, Corners y Tarjetas.
    """
    if df_historico is None:
        try:
            # Aquí está la corrección: ponemos el nombre exacto de tu archivo
            df_historico = pd.read_csv('data/historico_ligamx_completo.csv')
        except FileNotFoundError:
            try:
                df_historico = pd.read_csv('liga_mx_analytics/data/historico_ligamx_completo.csv')
            except FileNotFoundError:
                return "Error: No se encontró el archivo historico_ligamx_completo.csv"

    # ==========================================
    # 1. SIMULACIÓN DE GOLES (Fuerza Relativa y Dixon-Coles)
    # ==========================================
    lambda_local, lambda_visita = calcular_lambdas_ewma_y_fuerza(df_historico, equipo_local, equipo_visita)
    n_sims = 10000
    
    goles_l = np.random.poisson(lambda_local, n_sims)
    goles_v = np.random.poisson(lambda_visita, n_sims)
    
    resultados_exactos = {}
    for i in range(n_sims):
        marcador = f"{goles_l[i]}-{goles_v[i]}"
        resultados_exactos[marcador] = resultados_exactos.get(marcador, 0) + 1
        
    for k in resultados_exactos.keys():
        resultados_exactos[k] = resultados_exactos[k] / n_sims
        
    matriz_corregida = aplicar_dixon_coles(lambda_local, lambda_visita, resultados_exactos, rho=-0.15)
    
    prob_local, prob_visita, prob_empate, prob_over = 0.0, 0.0, 0.0, 0.0
    for marcador, prob in matriz_corregida.items():
        gl, gv = map(int, marcador.split('-'))
        if gl > gv: prob_local += prob
        elif gv > gl: prob_visita += prob
        else: prob_empate += prob
        if (gl + gv) > 2.5: prob_over += prob

    # ==========================================
    # 2. SIMULACIÓN REAL DE CORNERS (Poisson)
    # ==========================================
    # Definir nombres de columnas (Ajusta si en tu CSV se llaman diferente)
    col_corn_l = 'Corners_L' if 'Corners_L' in df_historico.columns else 'Corners_Local'
    col_corn_v = 'Corners_V' if 'Corners_V' in df_historico.columns else 'Corners_Visita'

    if col_corn_l in df_historico.columns and col_corn_v in df_historico.columns:
        df_l_loc = df_historico[df_historico['Local'] == equipo_local]
        df_v_vis = df_historico[df_historico['Visitante'] == equipo_visita]
        
        lambda_corn_l = df_l_loc[col_corn_l].mean() if not df_l_loc.empty else 5.0
        lambda_corn_v = df_v_vis[col_corn_v].mean() if not df_v_vis.empty else 4.5
        
        if pd.isna(lambda_corn_l): lambda_corn_l = 5.0
        if pd.isna(lambda_corn_v): lambda_corn_v = 4.5
    else:
        # Valores promedio de la Liga MX si no existen las columnas
        lambda_corn_l, lambda_corn_v = 5.2, 4.3

    corners_l_sim = np.random.poisson(lambda_corn_l, n_sims)
    corners_v_sim = np.random.poisson(lambda_corn_v, n_sims)
    corners_totales_sim = corners_l_sim + corners_v_sim
    prob_over_9_5_corners = (np.sum(corners_totales_sim > 9.5) / n_sims) * 100

    # Probabilidad del valor más frecuente de corners
    val_freq_corn_l = int(lambda_corn_l)
    prob_freq_corn_l = (np.sum(corners_l_sim == val_freq_corn_l) / n_sims) * 100
    val_freq_corn_v = int(lambda_corn_v)
    prob_freq_corn_v = (np.sum(corners_v_sim == val_freq_corn_v) / n_sims) * 100

    # ==========================================
    # 3. SIMULACIÓN REAL DE TARJETAS (Poisson)
    # ==========================================
    col_tarj_l = 'Tarjetas_L' if 'Tarjetas_L' in df_historico.columns else 'Amarillas_L'
    col_tarj_v = 'Tarjetas_V' if 'Tarjetas_V' in df_historico.columns else 'Amarillas_V'

    if col_tarj_l in df_historico.columns and col_tarj_v in df_historico.columns:
        lambda_tarj_l = df_l_loc[col_tarj_l].mean() if not df_l_loc.empty else 2.5
        lambda_tarj_v = df_v_vis[col_tarj_v].mean() if not df_v_vis.empty else 2.5
        
        if pd.isna(lambda_tarj_l): lambda_tarj_l = 2.5
        if pd.isna(lambda_tarj_v): lambda_tarj_v = 2.5
    else:
        lambda_tarj_l, lambda_tarj_v = 2.5, 2.7

    tarjetas_l_sim = np.random.poisson(lambda_tarj_l, n_sims)
    tarjetas_v_sim = np.random.poisson(lambda_tarj_v, n_sims)
    tarjetas_totales_sim = tarjetas_l_sim + tarjetas_v_sim
    prob_over_4_5_tarjetas = (np.sum(tarjetas_totales_sim > 4.5) / n_sims) * 100

    val_freq_tarj_l = int(lambda_tarj_l)
    prob_freq_tarj_l = (np.sum(tarjetas_l_sim == val_freq_tarj_l) / n_sims) * 100
    val_freq_tarj_v = int(lambda_tarj_v)
    prob_freq_tarj_v = (np.sum(tarjetas_v_sim == val_freq_tarj_v) / n_sims) * 100

    # ==========================================
    # 4. CONSOLIDACIÓN DE RESULTADOS
    # ==========================================
    return {
        "Resultado_1X2": {
            "Gana Local": round(prob_local * 100, 1),
            "Empate": round(prob_empate * 100, 1),
            "Gana Visita": round(prob_visita * 100, 1)
        },
        "Goles_Over_Under": {
            "Over 2.5": round(prob_over * 100, 1),
            "Under 2.5": round(100.0 - (prob_over * 100), 1)  # <--- Faltaba esto
        },
        "Goles_Individuales": {
            equipo_local: {"goles": round(lambda_local, 1), "prob": round((sum(1 for g in goles_l if g == round(lambda_local)) / n_sims) * 100, 1)},
            equipo_visita: {"goles": round(lambda_visita, 1), "prob": round((sum(1 for g in goles_v if g == round(lambda_visita)) / n_sims) * 100, 1)}
        },
        "Corners_Individuales": {
            equipo_local: {"corners": val_freq_corn_l, "prob": round(prob_freq_corn_l, 1)},
            equipo_visita: {"corners": val_freq_corn_v, "prob": round(prob_freq_corn_v, 1)}
        },
        "Tarjetas_Individuales": {
            equipo_local: {"tarjetas": val_freq_tarj_l, "prob": round(prob_freq_tarj_l, 1)},
            equipo_visita: {"tarjetas": val_freq_tarj_v, "prob": round(prob_freq_tarj_v, 1)}
        },
        "Corners_Totales": {
            "Over 9.5 Corners": round(prob_over_9_5_corners, 1),
            "Under 9.5 Corners": round(100.0 - prob_over_9_5_corners, 1) # <--- Prevención de error
        },
        "Tarjetas_Totales": {
            "Over 4.5 Tarjetas": round(prob_over_4_5_tarjetas, 1),
            "Under 4.5 Tarjetas": round(100.0 - prob_over_4_5_tarjetas, 1) # <--- Prevención de error
        }
    }
