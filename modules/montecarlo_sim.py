import numpy as np
from modules.stats_engine import calcular_expectativa_partido

def simular_partido_montecarlo(local, visitante, n_simulaciones=10000):
    try:
        exp = calcular_expectativa_partido(local, visitante)
    except Exception as e:
        return f"Error al cargar expectativas: {e}"

    # 1. Simulación Montecarlo (Poisson)
    sim_goles_l = np.random.poisson(exp["lambda_goles_local"], n_simulaciones)
    sim_goles_v = np.random.poisson(exp["lambda_goles_visita"], n_simulaciones)
    
    sim_corners_l = np.random.poisson(exp["exp_corners_local"], n_simulaciones)
    sim_corners_v = np.random.poisson(exp["exp_corners_visita"], n_simulaciones)
    
    sim_tarj_l = np.random.poisson(exp["exp_tarjetas_local"], n_simulaciones)
    sim_tarj_v = np.random.poisson(exp["exp_tarjetas_visita"], n_simulaciones)

    # 2. Función auxiliar para calcular el valor exacto más probable y su %
    def obtener_exacto_mas_probable(array_simulaciones):
        valoresUnicos, cuentas = np.unique(array_simulaciones, return_counts=True)
        idx_max = np.argmax(cuentas)
        goles_o_eventos = int(valoresUnicos[idx_max])
        probabilidad = (cuentas[idx_max] / n_simulaciones) * 100
        return goles_o_eventos, round(probabilidad, 2)

    # Goles exactos más probables individuales
    goles_exactos_l, prob_g_l = obtener_exacto_mas_probable(sim_goles_l)
    goles_exactos_v, prob_g_v = obtener_exacto_mas_probable(sim_goles_v)

    # Corners exactos más probables individuales
    corners_exactos_l, prob_c_l = obtener_exacto_mas_probable(sim_corners_l)
    corners_exactos_v, prob_c_v = obtener_exacto_mas_probable(sim_corners_v)

    # Tarjetas exactas más probables individuales
    tarj_exactas_l, prob_t_l = obtener_exacto_mas_probable(sim_tarj_l)
    tarj_exactas_v, prob_t_v = obtener_exacto_mas_probable(sim_tarj_v)

    # 3. Sumatorias totales (para Over/Under)
    total_goles = sim_goles_l + sim_goles_v
    total_corners = sim_corners_l + sim_corners_v
    total_tarjetas = sim_tarj_l + sim_tarj_v

    # 4. Estructura de resultados (REEMPLAZA ESTA PARTE AL FINAL DEL ARCHIVO)
    resultados = {
        "Resultado_1X2": {
            "Gana Local": round(np.sum(sim_goles_l > sim_goles_v) / n_simulaciones * 100, 2),
            "Empate": round(np.sum(sim_goles_l == sim_goles_v) / n_simulaciones * 100, 2),
            "Gana Visita": round(np.sum(sim_goles_l < sim_goles_v) / n_simulaciones * 100, 2)
        },
        "Goles_Individuales": {
            f"{local}": {"goles": goles_exactos_l, "prob": prob_g_l},
            f"{visitante}": {"goles": goles_exactos_v, "prob": prob_g_v}
        },
        "Corners_Individuales": {
            f"{local}": {"corners": corners_exactos_l, "prob": prob_c_l},
            f"{visitante}": {"corners": corners_exactos_v, "prob": prob_c_v}
        },
        "Tarjetas_Individuales": {
            f"{local}": {"tarjetas": tarj_exactas_l, "prob": prob_t_l},
            f"{visitante}": {"tarjetas": tarj_exactas_v, "prob": prob_t_v}
        },
        "Goles_Over_Under": {
            "Over 1.5": round(np.sum(total_goles > 1.5) / n_simulaciones * 100, 2),
            "Over 2.5": round(np.sum(total_goles > 2.5) / n_simulaciones * 100, 2),
            "Under 2.5": round(np.sum(total_goles <= 2.5) / n_simulaciones * 100, 2) # NUEVO
        },
        "Corners_Totales": {
            "Over 9.5 Corners": round(np.sum(total_corners > 9.5) / n_simulaciones * 100, 2),
            "Under 9.5 Corners": round(np.sum(total_corners <= 9.5) / n_simulaciones * 100, 2) # NUEVO
        },
        "Tarjetas_Totales": {
            "Over 4.5 Tarjetas": round(np.sum(total_tarjetas > 4.5) / n_simulaciones * 100, 2),
            "Under 4.5 Tarjetas": round(np.sum(total_tarjetas <= 4.5) / n_simulaciones * 100, 2) # NUEVO
        }
    }
    
    return resultados