import os
import pandas as pd
import numpy as np

# --- DICCIONARIO DE ALTITUDES LIGA MX (Nombres Oficiales de la API) ---
ALTITUDES_LIGA_MX = {
    "Toluca": 2660,
    "CF Pachuca": 2432,
    "U.N.A.M. - Pumas": 2240,
    "Club America": 2240,
    "Cruz Azul": 2240,
    "Puebla": 2135,
    "Club Tijuana": 20, 
    "Leon": 1815,
    "Club Queretaro": 1820,
    "Atletico San Luis": 1850,
    "Necaxa": 1888,
    "Atlas": 1560,
    "Guadalajara Chivas": 1560,
    "FC Juarez": 1120,
    "Santos Laguna": 1120,
    "Monterrey": 540,
    "Tigres UANL": 540,
    "Mazatlán": 10,
    "Atlante": 2240
}

def cargar_datos():
    df = pd.read_csv('data/historico_ligamx_completo.csv')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    fecha_referencia = df['Fecha'].max()
    df['Dias_Antiguedad'] = (fecha_referencia - df['Fecha']).dt.days
    df['Peso'] = 0.5 ** (df['Dias_Antiguedad'] / 365.0)

    # Sistema Híbrido: 60% xG + 40% Goles Reales
    if 'xG_L' not in df.columns: df['xG_L'] = df['Goles_L']
    if 'xG_V' not in df.columns: df['xG_V'] = df['Goles_V']
        
    df['xG_L'] = df['xG_L'].fillna(df['Goles_L'])
    df['xG_V'] = df['xG_V'].fillna(df['Goles_V'])

    df['Goles_Blend_L'] = (df['xG_L'] * 0.60) + (df['Goles_L'] * 0.40)
    df['Goles_Blend_V'] = (df['xG_V'] * 0.60) + (df['Goles_V'] * 0.40)
    
    return df

def media_ponderada(valores, pesos):
    df_temp = pd.DataFrame({'val': valores, 'peso': pesos}).dropna()
    if len(df_temp) == 0 or df_temp['peso'].sum() == 0: return 0
    return np.average(df_temp['val'], weights=df_temp['peso'])

def calcular_promedios_liga(df):
    pesos = df['Peso']
    
    # REGLA CASINO: 1 Amarilla = 1 punto, 1 Roja = 2 puntos
    t_locales = df['Amarillas_L'].fillna(0) + (df['Rojas_L'].fillna(0) * 2)
    t_visitas = df['Amarillas_V'].fillna(0) + (df['Rojas_V'].fillna(0) * 2)
    
    return {
        "media_goles_local": media_ponderada(df['Goles_Blend_L'], pesos),
        "media_goles_visita": media_ponderada(df['Goles_Blend_V'], pesos),
        "media_corners_local": media_ponderada(df['Corners_L'], pesos),
        "media_corners_visita": media_ponderada(df['Corners_V'], pesos),
        "media_tarjetas_local": media_ponderada(t_locales, pesos),
        "media_tarjetas_visita": media_ponderada(t_visitas, pesos)
    }

def obtener_ratings_equipo(df, equipo):
    prom = calcular_promedios_liga(df)
    df_local = df[df['Local'] == equipo]
    df_visita = df[df['Visitante'] == equipo]
    
    if len(df_local) == 0 or len(df_visita) == 0: return None

    al = media_ponderada(df_local['Goles_Blend_L'], df_local['Peso']) / prom['media_goles_local']
    dl = media_ponderada(df_local['Goles_Blend_V'], df_local['Peso']) / prom['media_goles_visita']
    av = media_ponderada(df_visita['Goles_Blend_V'], df_visita['Peso']) / prom['media_goles_visita']
    dv = media_ponderada(df_visita['Goles_Blend_L'], df_visita['Peso']) / prom['media_goles_local']

    cf_l = media_ponderada(df_local['Corners_L'], df_local['Peso'])
    cc_l = media_ponderada(df_local['Corners_V'], df_local['Peso'])
    cf_v = media_ponderada(df_visita['Corners_V'], df_visita['Peso'])
    cc_v = media_ponderada(df_visita['Corners_L'], df_visita['Peso'])

    # Aplicando regla de rojas dobles en estadísticas de equipos
    tf_l = media_ponderada(df_local['Amarillas_L'].fillna(0) + (df_local['Rojas_L'].fillna(0) * 2), df_local['Peso'])
    tc_l = media_ponderada(df_local['Amarillas_V'].fillna(0) + (df_local['Rojas_V'].fillna(0) * 2), df_local['Peso'])
    tf_v = media_ponderada(df_visita['Amarillas_V'].fillna(0) + (df_visita['Rojas_V'].fillna(0) * 2), df_visita['Peso'])
    tc_v = media_ponderada(df_visita['Amarillas_L'].fillna(0) + (df_visita['Rojas_L'].fillna(0) * 2), df_visita['Peso'])

    return {
        "Ataque_Local": al, "Defensa_Local": dl, "Ataque_Visita": av, "Defensa_Visita": dv,
        "Corners_Favor_L": cf_l, "Corners_Contra_L": cc_l, "Corners_Favor_V": cf_v, "Corners_Contra_V": cc_v,
        "Tarjetas_Favor_L": tf_l, "Tarjetas_Contra_L": tc_l, "Tarjetas_Favor_V": tf_v, "Tarjetas_Contra_V": tc_v
    }

def obtener_factor_h2h(df, local, visitante):
    h2h_df = df[((df['Local'] == local) & (df['Visitante'] == visitante)) | 
                ((df['Local'] == visitante) & (df['Visitante'] == local))]
    if len(h2h_df) < 2: return {"goles": 1.0, "corners": 1.0, "tarjetas": 1.0}
        
    prom = calcular_promedios_liga(df)
    pesos = h2h_df['Peso']
    
    goles_h2h = media_ponderada(h2h_df['Goles_Blend_L'] + h2h_df['Goles_Blend_V'], pesos)
    fact_g = goles_h2h / (prom['media_goles_local'] + prom['media_goles_visita'])
    
    corners_h2h = media_ponderada(h2h_df['Corners_L'].fillna(0) + h2h_df['Corners_V'].fillna(0), pesos)
    fact_c = corners_h2h / (prom['media_corners_local'] + prom['media_corners_visita'])

    tarjetas_h2h = media_ponderada(
        h2h_df['Amarillas_L'].fillna(0) + (h2h_df['Rojas_L'].fillna(0) * 2) + 
        h2h_df['Amarillas_V'].fillna(0) + (h2h_df['Rojas_V'].fillna(0) * 2), 
        pesos
    )
    fact_t = tarjetas_h2h / (prom['media_tarjetas_local'] + prom['media_tarjetas_visita'])

    return {
        "goles": max(0.7, min(1.3, fact_g if fact_g > 0 else 1.0)),
        "corners": max(0.8, min(1.2, fact_c if fact_c > 0 else 1.0)),
        "tarjetas": max(0.7, min(1.4, fact_t if fact_t > 0 else 1.0))
    }

def calcular_expectativa_partido(local, visitante, arbitro=None):
    df = cargar_datos()
    prom = calcular_promedios_liga(df)
    
    stats_l = obtener_ratings_equipo(df, local)
    stats_v = obtener_ratings_equipo(df, visitante)
    
    if stats_l is None or stats_v is None:
        raise ValueError(f"Faltan datos históricos para {local} o {visitante}")
    
    h2h = obtener_factor_h2h(df, local, visitante)
    
    lambda_goles_l = stats_l["Ataque_Local"] * stats_v["Defensa_Visita"] * prom["media_goles_local"] * h2h["goles"]
    lambda_goles_v = stats_v["Ataque_Visita"] * stats_l["Defensa_Local"] * prom["media_goles_visita"] * h2h["goles"]
    
    exp_corners_l = ((stats_l["Corners_Favor_L"] + stats_v["Corners_Contra_V"]) / 2) * h2h["corners"]
    exp_corners_v = ((stats_v["Corners_Favor_V"] + stats_l["Corners_Contra_L"]) / 2) * h2h["corners"]

    exp_tarjetas_l = ((stats_l["Tarjetas_Favor_L"] + stats_v["Tarjetas_Contra_V"]) / 2) * h2h["tarjetas"]
    exp_tarjetas_v = ((stats_v["Tarjetas_Favor_V"] + stats_l["Tarjetas_Contra_L"]) / 2) * h2h["tarjetas"]

    # Factor de Altitud (Hipoxia)
    alt_local = ALTITUDES_LIGA_MX.get(local, 1500)
    alt_visita = ALTITUDES_LIGA_MX.get(visitante, 1500)
    delta_altitud = alt_local - alt_visita

    if delta_altitud >= 1200:
        lambda_goles_l *= 1.10
        exp_corners_l *= 1.10
        exp_tarjetas_v *= 1.15
    elif delta_altitud >= 600:
        lambda_goles_l *= 1.05
        exp_corners_l *= 1.05
        exp_tarjetas_v *= 1.08

    # Factor Árbitro Real (si fue proporcionado)
    if arbitro:
        from modules.referee_engine import obtener_factor_arbitro
        factor_arb = obtener_factor_arbitro(arbitro)
        exp_tarjetas_l *= factor_arb
        exp_tarjetas_v *= factor_arb

    return {
        "lambda_goles_local": max(0.1, lambda_goles_l),
        "lambda_goles_visita": max(0.1, lambda_goles_v),
        "exp_corners_local": max(0.1, exp_corners_l),
        "exp_corners_visita": max(0.1, exp_corners_v),
        "exp_tarjetas_local": max(0.1, exp_tarjetas_l),  
        "exp_tarjetas_visita": max(0.1, exp_tarjetas_v)
    }
