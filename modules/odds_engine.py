import pandas as pd

def obtener_cuotas_partido(local, visita, league_id=262):
    """Devuelve un diccionario base limpio."""
    return {
        "1": 0.0, "X": 0.0, "2": 0.0,
        "Linea_Goles": 2.5, "Over_Goles": 0.0, "Under_Goles": 0.0,
        "Linea_Corners": 9.5, "Over_Corners": 0.0, "Under_Corners": 0.0,
        "Linea_Tarjetas": 4.5, "Over_Tarjetas": 0.0, "Under_Tarjetas": 0.0
    }

def analizar_apuestas(resultados, local, visita, cuotas_personalizadas=None, lineas_default=None):
    """Calcula el EV y el Criterio de Kelly basándose estrictamente en las cuotas reales que tú ingreses."""
    apuestas_lista = []
    
    if not lineas_default:
        lineas_default = {"Linea_Goles": 2.5, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
        
    l_goles = lineas_default.get("Linea_Goles", 2.5)
    l_corners = lineas_default.get("Linea_Corners", 9.5)
    l_tarjetas = lineas_default.get("Linea_Tarjetas", 4.5)
    
    cuotas = cuotas_personalizadas if cuotas_personalizadas else {}
    
    # 1. Mercado 1X2
    prob_1x2 = resultados.get('Resultado_1X2', {})
    mercados_1x2 = [
        ("Gana Local", f"Victoria {local}", prob_1x2.get('Gana Local', 0) / 100.0, float(cuotas.get('1', 0.0))),
        ("Empate", "Empate", prob_1x2.get('Empate', 0) / 100.0, float(cuotas.get('X', 0.0))),
        ("Gana Visita", f"Victoria {visita}", prob_1x2.get('Gana Visita', 0) / 100.0, float(cuotas.get('2', 0.0)))
    ]
    
    for key_m, nombre_m, prob_real, cuota in mercados_1x2:
        if cuota > 1.0 and prob_real > 0:
            ev = (prob_real * cuota) - 1.0
            kelly = max(0.0, ((prob_real * cuota - 1) / (cuota - 1))) * 100 if cuota > 1 else 0.0
            veredicto = "🔥 Valor Alto (EV+)" if ev > 0.05 else ("✅ Valor Moderado" if ev > 0.0 else "Paso")
        else:
            ev, kelly, veredicto = 0.0, 0.0, "⚠️ Sin cuota ingresada"
            cuota = 0.0
            
        apuestas_lista.append({
            "Mercado": nombre_m,
            "Prob. Modelo": f"{round(prob_real * 100, 1)}%",
            "Cuota": f"{cuota:.2f}" if cuota > 0 else "N/A",
            "EV (%)": f"{round(ev * 100, 2)}%" if cuota > 0 else "N/A",
            "Kelly (%)": f"{round(kelly, 2)}%" if cuota > 0 else "N/A",
            "Veredicto": veredicto
        })

    # 2. Goles Over
    goles_ou = resultados.get('Goles_Over_Under', {})
    prob_over_goles = goles_ou.get(f'Over {l_goles}', 50.0) / 100.0
    cuota_over_goles = float(cuotas.get('Over_Goles', 0.0))
    
    if cuota_over_goles > 1.0 and prob_over_goles > 0:
        ev_og = (prob_over_goles * cuota_over_goles) - 1.0
        kelly_og = max(0.0, ((prob_over_goles * cuota_over_goles - 1) / (cuota_over_goles - 1))) * 100 if cuota_over_goles > 1 else 0.0
        ver_og = "🔥 Valor Alto (EV+)" if ev_og > 0.05 else ("✅ Valor Moderado" if ev_og > 0.0 else "Paso")
    else:
        ev_og, kelly_og, ver_og = 0.0, 0.0, "⚠️ Sin cuota ingresada"
        cuota_over_goles = 0.0

    apuestas_lista.append({
        "Mercado": f"Over {l_goles} Goles",
        "Prob. Modelo": f"{round(prob_over_goles * 100, 1)}%",
        "Cuota": f"{cuota_over_goles:.2f}" if cuota_over_goles > 0 else "N/A",
        "EV (%)": f"{round(ev_og * 100, 2)}%" if cuota_over_goles > 0 else "N/A",
        "Kelly (%)": f"{round(kelly_og, 2)}%" if cuota_over_goles > 0 else "N/A",
        "Veredicto": ver_og
    })

    # 3. Córners Over
    corners_tot = resultados.get('Corners_Totales', {})
    prob_over_corners = corners_tot.get(f'Over {l_corners} Corners', 50.0) / 100.0
    cuota_over_corners = float(cuotas.get('Over_Corners', 0.0))
    
    if cuota_over_corners > 1.0 and prob_over_corners > 0:
        ev_oc = (prob_over_corners * cuota_over_corners) - 1.0
        kelly_oc = max(0.0, ((prob_over_corners * cuota_over_corners - 1) / (cuota_over_corners - 1))) * 100 if cuota_over_corners > 1 else 0.0
        ver_oc = "🔥 Valor Alto (EV+)" if ev_oc > 0.05 else ("✅ Valor Moderado" if ev_oc > 0.0 else "Paso")
    else:
        ev_oc, kelly_oc, ver_oc = 0.0, 0.0, "⚠️ Sin cuota ingresada"
        cuota_over_corners = 0.0

    apuestas_lista.append({
        "Mercado": f"Over {l_corners} Córners",
        "Prob. Modelo": f"{round(prob_over_corners * 100, 1)}%",
        "Cuota": f"{cuota_over_corners:.2f}" if cuota_over_corners > 0 else "N/A",
        "EV (%)": f"{round(ev_oc * 100, 2)}%" if cuota_over_corners > 0 else "N/A",
        "Kelly (%)": f"{round(kelly_oc, 2)}%" if cuota_over_corners > 0 else "N/A",
        "Veredicto": ver_oc
    })

    return pd.DataFrame(apuestas_lista)
