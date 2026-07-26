import os
import pandas as pd
import numpy as np

from modules.goalkeeper_engine import calcular_eficiencia_portero_api

def aplicar_dixon_coles(lambda_l, lambda_v, prob_matriz, rho=-0.05):
    """
    Ajusta la matriz de probabilidades de marcadores exactos utilizando 
    la correlación de Dixon-Coles para inflar los empates de baja anotación.
    """
    if (1 - lambda_l * lambda_v * rho) < 0:
        rho = 0
        
    tau_0_0 = 1 - lambda_l * lambda_v * rho
    tau_1_0 = 1 + lambda_v * rho
    tau_0_1 = 1 + lambda_l * rho
    tau_1_1 = 1 - rho

    if "0-0" in prob_matriz: prob_matriz["0-0"] *= tau_0_0
    if "1-0" in prob_matriz: prob_matriz["1-0"] *= tau_1_0
    if "0-1" in prob_matriz: prob_matriz["0-1"] *= tau_0_1
    if "1-1" in prob_matriz: prob_matriz["1-1"] *= tau_1_1

    total = sum(prob_matriz.values())
    for k in prob_matriz.keys():
        prob_matriz[k] = (prob_matriz[k] / total)

    return prob_matriz

def calcular_lambdas_estables(df, equipo_local, equipo_visita):
    """
    Calcula expectativas de goles (Lambdas) realistas. 
    Si faltan datos, usa promedios globales dinámicos de la liga.
    """
    if 'Fecha' in df.columns:
        df = df.sort_values(by='Fecha', ascending=True)

    col_l = 'Goles_L' if 'Goles_L' in df.columns else 'Puntos_L'
    col_v = 'Goles_V' if 'Goles_V' in df.columns else 'Puntos_V'
    
    promedio_global_goles_local = df[col_l].mean() if not df[col_l].empty else 1.3
    promedio_global_goles_visita = df[col_v].mean() if not df[col_v].empty else 1.1

    df_l_loc = df[df['Local'] == equipo_local].tail(6)
    df_l_vis = df[df['Visitante'] == equipo_local].tail(6)
    
    df_v_loc = df[df['Local'] == equipo_visita].tail(6)
    df_v_vis = df[df['Visitante'] == equipo_visita].tail(6)

    goles_fav_local = pd.concat([df_l_loc[col_l], df_l_vis[col_v]])
    goles_rec_local = pd.concat([df_l_loc[col_v], df_l_vis[col_l]])
    
    goles_fav_visita = pd.concat([df_v_loc[col_l], df_v_vis[col_v]])
    goles_rec_visita = pd.concat([df_v_loc[col_v], df_v_vis[col_l]])

    prom_anota_l = goles_fav_local.mean() if not goles_fav_local.empty else promedio_global_goles_local
    prom_recibe_v = goles_rec_visita.mean() if not goles_rec_visita.empty else promedio_global_goles_local
    
    prom_anota_v = goles_fav_visita.mean() if not goles_fav_visita.empty else promedio_global_goles_visita
    prom_recibe_l = goles_rec_local.mean() if not goles_rec_local.empty else promedio_global_goles_visita

    prom_anota_l = promedio_global_goles_local if pd.isna(prom_anota_l) else prom_anota_l
    prom_recibe_v = promedio_global_goles_local if pd.isna(prom_recibe_v) else prom_recibe_v
    prom_anota_v = promedio_global_goles_visita if pd.isna(prom_anota_v) else prom_anota_v
    prom_recibe_l = promedio_global_goles_visita if pd.isna(prom_recibe_l) else prom_recibe_l

    lambda_local = (prom_anota_l + prom_recibe_v) / 2.0
    lambda_visita = (prom_anota_v + prom_recibe_l) / 2.0

    return lambda_local, lambda_visita

IDS_EQUIPOS_LIGAMX = {
    "America": 228, "Guadalajara": 229, "Cruz Azul": 214, "UNAM": 215, 
    "Monterrey": 211, "Tigres": 227, "Toluca": 222, "Pachuca": 216, 
    "Leon": 218, "Santos": 212, "Atlas": 213, "Necaxa": 217, 
    "Tijuana": 224, "Puebla": 220, "Queretaro": 221, "Mazatlan": 3288, 
    "Juarez": 3217, "Atletico de San Luis": 3287
}

def simular_partido_montecarlo(equipo_local, equipo_visita, df_historico=None, elo_local=1500, elo_visita=1500):
    """
    Ejecuta 100,000 iteraciones de Montecarlo reales para Goles, Corners y Tarjetas.
    Ahora acepta opcionalmente el rating ELO para calibrar la balanza de goles.
    """
    if df_historico is None:
        try:
            df_historico = pd.read_csv('data/historico_ligamx_completo.csv')
        except FileNotFoundError:
            try:
                df_historico = pd.read_csv('liga_mx_analytics/data/historico_ligamx_completo.csv')
            except FileNotFoundError:
                return "Error: No se encontró el archivo historico_ligamx_completo.csv"

    # ==========================================
    # 1. CÁLCULO DE LAMBDAS Y EFECTO PORTEROS
    # ==========================================
    lambda_local_base, lambda_visita_base = calcular_lambdas_estables(df_historico, equipo_local, equipo_visita)
    
    id_local = IDS_EQUIPOS_LIGAMX.get(equipo_local, None)
    id_visita = IDS_EQUIPOS_LIGAMX.get(equipo_visita, None)
    
    factor_portero_local = calcular_eficiencia_portero_api(id_local, equipo_local) if id_local else 1.0
    factor_portero_visita = calcular_eficiencia_portero_api(id_visita, equipo_visita) if id_visita else 1.0
    
    lambda_local = lambda_local_base * factor_portero_visita
    lambda_visita = lambda_visita_base * factor_portero_local
    
    # ==========================================
    # 1.1 AJUSTE POR DIFERENCIA DE ELO
    # ==========================================
    # Si hay una diferencia de ELO, ajustamos sutilmente los goles esperados
    diff_elo = elo_local - elo_visita
    factor_elo_local = 1.0 + (diff_elo / 2000.0) # Ajuste conservador
    factor_elo_visita = 1.0 - (diff_elo / 2000.0)
    
    lambda_local *= max(0.8, min(factor_elo_local, 1.2))
    lambda_visita *= max(0.8, min(factor_elo_visita, 1.2))

    # Topes lógicos para la Liga MX
    lambda_local = max(min(lambda_local, 2.2), 0.5)
    lambda_visita = max(min(lambda_visita, 2.0), 0.4)

    # ==========================================
    # 2. SIMULACIÓN DE GOLES (Montecarlo + Dixon-Coles)
    # ==========================================
    n_sims = 100000
    
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
    # 3. SIMULACIÓN REAL DE CORNERS (Poisson)
    # ==========================================
    col_corn_l = 'Corners_L' if 'Corners_L' in df_historico.columns else 'Corners_Local'
    col_corn_v = 'Corners_V' if 'Corners_V' in df_historico.columns else 'Corners_Visita'

    prom_global_corn_l = df_historico[col_corn_l].mean() if col_corn_l in df_historico.columns else 5.2
    prom_global_corn_v = df_historico[col_corn_v].mean() if col_corn_v in df_historico.columns else 4.3

    if col_corn_l in df_historico.columns and col_corn_v in df_historico.columns:
        df_l_loc = df_historico[df_historico['Local'] == equipo_local]
        df_v_vis = df_historico[df_historico['Visitante'] == equipo_visita]
        
        lambda_corn_l = df_l_loc[col_corn_l].mean() if not df_l_loc.empty else prom_global_corn_l
        lambda_corn_v = df_v_vis[col_corn_v].mean() if not df_v_vis.empty else prom_global_corn_v
        
        if pd.isna(lambda_corn_l): lambda_corn_l = prom_global_corn_l
        if pd.isna(lambda_corn_v): lambda_corn_v = prom_global_corn_v
    else:
        lambda_corn_l, lambda_corn_v = prom_global_corn_l, prom_global_corn_v

    corners_l_sim = np.random.poisson(lambda_corn_l, n_sims)
    corners_v_sim = np.random.poisson(lambda_corn_v, n_sims)
    corners_totales_sim = corners_l_sim + corners_v_sim
    prob_over_9_5_corners = (np.sum(corners_totales_sim > 9.5) / n_sims) * 100

    val_freq_corn_l = int(lambda_corn_l)
    prob_freq_corn_l = (np.sum(corners_l_sim == val_freq_corn_l) / n_sims) * 100
    val_freq_corn_v = int(lambda_corn_v)
    prob_freq_corn_v = (np.sum(corners_v_sim == val_freq_corn_v) / n_sims) * 100

    # ==========================================
    # 4. SIMULACIÓN REAL DE TARJETAS (Poisson)
    # ==========================================
    col_tarj_l = 'Tarjetas_L' if 'Tarjetas_L' in df_historico.columns else 'Amarillas_L'
    col_tarj_v = 'Tarjetas_V' if 'Tarjetas_V' in df_historico.columns else 'Amarillas_V'

    prom_global_tarj_l = df_historico[col_tarj_l].mean() if col_tarj_l in df_historico.columns else 2.5
    prom_global_tarj_v = df_historico[col_tarj_v].mean() if col_tarj_v in df_historico.columns else 2.7

    if col_tarj_l in df_historico.columns and col_tarj_v in df_historico.columns:
        lambda_tarj_l = df_l_loc[col_tarj_l].mean() if not df_l_loc.empty else prom_global_tarj_l
        lambda_tarj_v = df_v_vis[col_tarj_v].mean() if not df_v_vis.empty else prom_global_tarj_v
        
        if pd.isna(lambda_tarj_l): lambda_tarj_l = prom_global_tarj_l
        if pd.isna(lambda_tarj_v): lambda_tarj_v = prom_global_tarj_v
    else:
        lambda_tarj_l, lambda_tarj_v = prom_global_tarj_l, prom_global_tarj_v

    tarjetas_l_sim = np.random.poisson(lambda_tarj_l, n_sims)
    tarjetas_v_sim = np.random.poisson(lambda_tarj_v, n_sims)
    tarjetas_totales_sim = tarjetas_l_sim + tarjetas_v_sim
    prob_over_4_5_tarjetas = (np.sum(tarjetas_totales_sim > 4.5) / n_sims) * 100

    val_freq_tarj_l = int(lambda_tarj_l)
    prob_freq_tarj_l = (np.sum(tarjetas_l_sim == val_freq_tarj_l) / n_sims) * 100
    val_freq_tarj_v = int(lambda_tarj_v)
    prob_freq_tarj_v = (np.sum(tarjetas_v_sim == val_freq_tarj_v) / n_sims) * 100

    # ==========================================
    # 5. CONSOLIDACIÓN DE RESULTADOS
    # ==========================================
    return {
        "Resultado_1X2": {
            "Gana Local": round(prob_local * 100, 1),
            "Empate": round(prob_empate * 100, 1),
            "Gana Visita": round(prob_visita * 100, 1)
        },
        "Goles_Over_Under": {
            "Over 2.5": round(prob_over * 100, 1),
            "Under 2.5": round(100.0 - (prob_over * 100), 1)  
        },
       "Goles_Individuales": {
            equipo_local: {
                "goles": int(round(lambda_local)), 
                "prob": round((np.sum(goles_l == int(round(lambda_local))) / n_sims) * 100, 1) 
            },
            equipo_visita: {
                "goles": int(round(lambda_visita)), 
                "prob": round((np.sum(goles_v == int(round(lambda_visita))) / n_sims) * 100, 1)
            }
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
            "Under 9.5 Corners": round(100.0 - prob_over_9_5_corners, 1) 
        },
        "Tarjetas_Totales": {
            "Over 4.5 Tarjetas": round(prob_over_4_5_tarjetas, 1),
            "Under 4.5 Tarjetas": round(100.0 - prob_over_4_5_tarjetas, 1) 
        }
    }
    
