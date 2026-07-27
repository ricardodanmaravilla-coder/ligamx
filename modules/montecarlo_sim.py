import numpy as np
import pandas as pd
from modules.stats_engine import calcular_expectativa_partido

def normalizar_nombre_equipo(nombre):
    """Mapea y estandariza los nombres usando la lista oficial de la Liga MX."""
    if not isinstance(nombre, str):
        return ""
    n = nombre.upper().strip()
    if "TOLUCA" in n: return "Toluca"
    if "PACHUCA" in n: return "CF Pachuca"
    if "PUMAS" in n or "U.N.A.M." in n: return "U.N.A.M. - Pumas"
    if "AMERICA" in n or "AMÉRICA" in n: return "Club America"
    if "CRUZ AZUL" in n: return "Cruz Azul"
    if "PUEBLA" in n: return "Puebla"
    if "TIJUANA" in n: return "Club Tijuana"
    if "LEON" in n or "LEÓN" in n: return "Leon"
    if "QUERETARO" in n or "QUERÉTARO" in n: return "Club Queretaro"
    if "SAN LUIS" in n: return "Atletico San Luis"
    if "NECAXA" in n: return "Necaxa"
    if "ATLAS" in n: return "Atlas"
    if "GUADALAJARA" in n or "CHIVAS" in n: return "Guadalajara Chivas"
    if "JUAREZ" in n or "JUÁREZ" in n: return "FC Juarez"
    if "SANTOS" in n: return "Santos Laguna"
    if "MONTERREY" in n or "RAYADOS" in n: return "Monterrey"
    if "TIGRES" in n: return "Tigres UANL"
    if "MAZATLAN" in n or "MAZATLÁN" in n: return "Mazatlán"
    if "ATLANTE" in n: return "Atlante"
    return nombre

def simular_partido_montecarlo(local_raw, visita_raw, df_historico=None, elo_local=None, elo_visita=None, num_simulaciones=100000000, arbitro=None):
    local = normalizar_nombre_equipo(local_raw)
    visita = normalizar_nombre_equipo(visita_raw)
    
    # Obtener ELO real dinámicamente si no se proporciona explícitamente
    if elo_local is None or elo_visita is None:
        try:
            from modules.elo_engine import SistemaEloLigaMX
            motor_elo = SistemaEloLigaMX()
            tabla_elo = motor_elo.calcular_historico(df_historico)
            if elo_local is None:
                elo_local = float(tabla_elo.loc[tabla_elo['Equipo'] == local, 'ELO_Rating'].values[0])
            if elo_visita is None:
                elo_visita = float(tabla_elo.loc[tabla_elo['Equipo'] == visita, 'ELO_Rating'].values[0])
        except Exception:
            elo_local = elo_local if elo_local is not None else 1500.0
            elo_visita = elo_visita if elo_visita is not None else 1500.0

    try:
        expectativas = calcular_expectativa_partido(local, visita, arbitro=arbitro)
        goles_l_exp = expectativas["lambda_goles_local"]
        goles_v_exp = expectativas["lambda_goles_visita"]
        corners_l_exp = expectativas["exp_corners_local"]
        corners_v_exp = expectativas["exp_corners_visita"]
        tarjetas_l_exp = expectativas["exp_tarjetas_local"]
        tarjetas_v_exp = expectativas["exp_tarjetas_visita"]
    except Exception:
        goles_l_exp = 1.35
        goles_v_exp = 1.05
        corners_l_exp = 5.2
        corners_v_exp = 4.3
        tarjetas_l_exp = 2.4
        tarjetas_v_exp = 2.6

    factor_elo = (elo_local - elo_visita) / 400.0
    goles_l_exp = max(0.2, goles_l_exp + (factor_elo * 0.15))
    goles_v_exp = max(0.2, goles_v_exp - (factor_elo * 0.15))

    goles_loc_sim = np.random.poisson(goles_l_exp, num_simulaciones)
    goles_vis_sim = np.random.poisson(goles_v_exp, num_simulaciones)
    
    corners_loc_sim = np.random.poisson(corners_l_exp, num_simulaciones)
    corners_vis_sim = np.random.poisson(corners_v_exp, num_simulaciones)
    corners_totales_sim = corners_loc_sim + corners_vis_sim
    
    tarjetas_loc_sim = np.random.poisson(tarjetas_l_exp, num_simulaciones)
    tarjetas_vis_sim = np.random.poisson(tarjetas_v_exp, num_simulaciones)
    tarjetas_totales_sim = tarjetas_loc_sim + tarjetas_vis_sim

    wins_local = np.sum(goles_loc_sim > goles_vis_sim)
    wins_visita = np.sum(goles_loc_sim < goles_vis_sim)
    empates = np.sum(goles_loc_sim == goles_vis_sim)

    prob_local = round((wins_local / num_simulaciones) * 100, 1)
    prob_visita = round((wins_visita / num_simulaciones) * 100, 1)
    prob_empate = round((empates / num_simulaciones) * 100, 1)

    over_25_goles = round((np.sum((goles_loc_sim + goles_vis_sim) > 2.5) / num_simulaciones) * 100, 1)
    under_25_goles = round(100.0 - over_25_goles, 1)

    over_95_corners = round((np.sum(corners_totales_sim > 9.5) / num_simulaciones) * 100, 1)
    under_95_corners = round(100.0 - over_95_corners, 1)

    over_45_tarjetas = round((np.sum(tarjetas_totales_sim > 4.5) / num_simulaciones) * 100, 1)
    under_45_tarjetas = round(100.0 - over_45_tarjetas, 1)

    return {
        "Resultado_1X2": {
            "Gana Local": prob_local,
            "Empate": prob_empate,
            "Gana Visita": prob_visita
        },
        "Goles_Over_Under": {
            "Over 2.5": over_25_goles,
            "Under 2.5": under_25_goles
        },
        "Corners_Totales": {
            "Over 9.5 Corners": over_95_corners,
            "Under 9.5 Corners": under_95_corners
        },
        "Tarjetas_Totales": {
            "Over 4.5 Tarjetas": over_45_tarjetas,
            "Under 4.5 Tarjetas": under_45_tarjetas
        },
        "Goles_Individuales": {
            local_raw: {"goles": round(goles_l_exp, 2)},
            visita_raw: {"goles": round(goles_v_exp, 2)}
        },
        "Corners_Individuales": {
            local_raw: {"corners": round(corners_l_exp, 2)},
            visita_raw: {"corners": round(corners_v_exp, 2)}
        },
        "Tarjetas_Individuales": {
            local_raw: {"tarjetas": round(tarjetas_l_exp, 2)},
            visita_raw: {"tarjetas": round(tarjetas_v_exp, 2)}
        }
    }
