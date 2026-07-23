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

def calcular_lambdas_ajustadas(df, equipo_local, equipo_visita):
    """
    Calcula la expectativa de goles (Lambda) ajustada por la fuerza de ataque 
    y defensa relativa al promedio de toda la Liga MX.
    """
    # 1. Calcular promedios globales de la liga
    # (Asumiendo que las columnas se llaman 'Goles_L' y 'Goles_V' o similar en tu CSV)
    col_l = 'Goles_L' if 'Goles_L' in df.columns else 'Puntos_L'
    col_v = 'Goles_V' if 'Goles_V' in df.columns else 'Puntos_V'
    
    media_goles_local_liga = df[col_l].mean()
    media_goles_visita_liga = df[col_v].mean()
    
    # Prevención de división por cero en casos extremos
    media_goles_local_liga = max(media_goles_local_liga, 0.01)
    media_goles_visita_liga = max(media_goles_visita_liga, 0.01)

    # 2. Extraer historial de los equipos involucrados
    df_local_como_local = df[df['Local'] == equipo_local]
    df_local_como_visita = df[df['Visitante'] == equipo_local]
    
    df_visita_como_local = df[df['Local'] == equipo_visita]
    df_visita_como_visita = df[df['Visitante'] == equipo_visita]

    # Promedios del Equipo Local
    avg_goles_anotados_local = df_local_como_local[col_l].mean() if not df_local_como_local.empty else media_goles_local_liga
    avg_goles_recibidos_local = df_local_como_local[col_v].mean() if not df_local_como_local.empty else media_goles_visita_liga
    
    # Promedios del Equipo Visitante
    avg_goles_anotados_visita = df_visita_como_visita[col_v].mean() if not df_visita_como_visita.empty else media_goles_visita_liga
    avg_goles_recibidos_visita = df_visita_como_visita[col_l].mean() if not df_visita_como_visita.empty else media_goles_local_liga

    # 3. Calcular Fuerza de Ataque (Ratio vs Liga)
    fuerza_ataque_local = avg_goles_anotados_local / media_goles_local_liga
    fuerza_ataque_visita = avg_goles_anotados_visita / media_goles_visita_liga

    # 4. Calcular Fuerza de Defensa (Ratio vs Liga)
    # Una defensa fuerte tendrá un valor menor a 1.0 (permite menos goles que el promedio)
    fuerza_defensa_local = avg_goles_recibidos_local / media_goles_visita_liga
    fuerza_defensa_visita = avg_goles_recibidos_visita / media_goles_local_liga

    # 5. Cálculo Final de Lambdas Cruzadas
    lambda_local = fuerza_ataque_local * fuerza_defensa_visita * media_goles_local_liga
    lambda_visita = fuerza_ataque_visita * fuerza_defensa_local * media_goles_visita_liga

    # Si tienes variables de altitud o impacto arbitral, puedes multiplicarlas aquí.
    # Ejemplo: lambda_local *= factor_altitud
    
    return max(lambda_local, 0.1), max(lambda_visita, 0.1)

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
    lambda_local, lambda_visita = calcular_lambdas_ajustadas(df_historico, equipo_local, equipo_visita)
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
            "Over 2.5": round(prob_over * 100, 1)
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
        "Corners_Totales": {"Over 9.5 Corners": round(prob_over_9_5_corners, 1)},
        "Tarjetas_Totales": {"Over 4.5 Tarjetas": round(prob_over_4_5_tarjetas, 1)}
    }
