import os
import requests
import pandas as pd
from datetime import datetime
import streamlit as st

from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido
from modules.ml_engine import PredictorML

def obtener_ultimo_elo(df, equipo):
    try:
        if df is not None and not df.empty:
            df_eq = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)]
            if not df_eq.empty:
                ultima_fila = df_eq.iloc[-1]
                if ultima_fila['Local'] == equipo:
                    for col in ['ELO_Local', 'ELO_L', 'Elo_Local']:
                        if col in df.columns and not pd.isna(ultima_fila[col]):
                            return float(ultima_fila[col])
                else:
                    for col in ['ELO_Visita', 'ELO_V', 'Elo_Visita', 'ELO_Visitante']:
                        if col in df.columns and not pd.isna(ultima_fila[col]):
                            return float(ultima_fila[col])
    except:
        pass
    return 1500.0

def escanear_jornada_actual():
    """Escanea los próximos partidos de la Liga MX usando la cartelera oficial de ESPN."""
    ml_escanner = PredictorML()
    df_historico_ml = None
    ruta_csv = 'data/historico_ligamx_completo.csv'
    url_github_raw = f'https://raw.githubusercontent.com/ricardodanmaravilla-coder/ligamx/main/{ruta_csv}'
    rutas_locales = [ruta_csv, os.path.basename(ruta_csv)]
    
    try:
        for r in rutas_locales:
            if os.path.exists(r):
                df_historico_ml = pd.read_csv(r)
                break
        if df_historico_ml is None:
            df_historico_ml = pd.read_csv(url_github_raw)
            
        df_historico_ml['Local'] = df_historico_ml['Local'].str.strip()
        df_historico_ml['Visitante'] = df_historico_ml['Visitante'].str.strip()
        ml_escanner.entrenar(df_historico_ml)
    except Exception:
        pass

    # Consultamos la cartelera pública de ESPN para Liga MX
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return []
        data = res.json()
        fixtures = data.get('events', [])
    except Exception:
        return []

    oportunidades_oro = []
    st.write("---")
    st.write("🔎 **Escaneando partidos oficiales de la Liga MX (Cartelera ESPN) y validando modelos...**")

    for event in fixtures:
        try:
            fecha = event.get('date', '')[:16].replace("T", " ")
            competencia = event.get('competitions', [{}])[0]
            competitors = competencia.get('competitors', [])
            
            local, visita = "", ""
            for comp in competitors:
                team_name = comp.get('team', {}).get('displayName', '')
                if comp.get('homeAway') == 'home':
                    local = team_name
                else:
                    visita = team_name
            
            if not local or not visita:
                continue

            st.write(f"⚙️ Analizando: **{local} vs {visita}**")
            
            cuotas = obtener_cuotas_partido(local, visita)
            linea_goles = cuotas.get("Linea_Goles", 2.5)
            linea_corners = cuotas.get("Linea_Corners", 9.5)
            linea_tarjetas = cuotas.get("Linea_Tarjetas", 4.5)
                
            elo_loc = obtener_ultimo_elo(df_historico_ml, local)
            elo_vis = obtener_ultimo_elo(df_historico_ml, visita)
            
            resultados = simular_partido_montecarlo(
                local, visita, 
                df_historico=df_historico_ml, 
                elo_local=elo_loc, 
                elo_visita=elo_vis, 
                linea_goles=linea_goles, 
                linea_corners=linea_corners, 
                linea_tarjetas=linea_tarjetas
            )
            
            if isinstance(resultados, str): 
                continue

            prob_ml_local = prob_ml_empate = prob_ml_visita = 0.0
            prob_ml_over_g = prob_ml_under_g = 0.0
            prob_ml_over_c = prob_ml_under_c = 0.0
            prob_ml_over_t = prob_ml_under_t = 0.0

            if ml_escanner.is_trained and df_historico_ml is not None:
                g_l_sim = resultados.get('Goles_Individuales', {}).get(local, {}).get('goles', 1.2)
                g_v_sim = resultados.get('Goles_Individuales', {}).get(visita, {}).get('goles', 1.0)
                
                preds_ml = ml_escanner.predecir_mercados_completos(
                    df_historico_ml, local, visita, g_l_sim, g_v_sim, elo_loc, elo_vis, 
                    linea_goles=linea_goles, linea_corners=linea_corners, linea_tarjetas=linea_tarjetas
                )
                
                if "Resultado_1X2" in preds_ml:
                    prob_ml_local = preds_ml['Resultado_1X2']['Gana Local']
                    prob_ml_empate = preds_ml['Resultado_1X2']['Empate']
                    prob_ml_visita = preds_ml['Resultado_1X2']['Gana Visita']
                    
                    prob_ml_over_g = preds_ml['Goles_Over_Under'].get(f'Over {linea_goles}', 50.0)
                    prob_ml_under_g = preds_ml['Goles_Over_Under'].get(f'Under {linea_goles}', 50.0)
                    
                    prob_ml_over_c = preds_ml['Corners_Totales'].get(f'Over {linea_corners} Corners', 50.0)
                    prob_ml_under_c = preds_ml['Corners_Totales'].get(f'Under {linea_corners} Corners', 50.0)
                    
                    prob_ml_over_t = preds_ml['Tarjetas_Totales'].get(f'Over {linea_tarjetas} Tarjetas', 50.0)
                    prob_ml_under_t = preds_ml['Tarjetas_Totales'].get(f'Under {linea_tarjetas} Tarjetas', 50.0)

            prob_mc_dict = {
                "Gana Local": resultados.get('Resultado_1X2', {}).get('Gana Local', 0.0),
                "Empate": resultados.get('Resultado_1X2', {}).get('Empate', 0.0),
                "Gana Visita": resultados.get('Resultado_1X2', {}).get('Gana Visita', 0.0),
                f"Over {linea_goles} Goles": resultados.get('Goles_Over_Under', {}).get(f'Over {linea_goles}', 0.0),
                f"Under {linea_goles} Goles": resultados.get('Goles_Over_Under', {}).get(f'Under {linea_goles}', 0.0),
                f"Over {linea_corners} Corners": resultados.get('Corners_Totales', {}).get(f'Over {linea_corners} Corners', 0.0),
                f"Under {linea_corners} Corners": resultados.get('Corners_Totales', {}).get(f'Under {linea_corners} Corners', 0.0),
                f"Over {linea_tarjetas} Tarjetas": resultados.get('Tarjetas_Totales', {}).get(f'Over {linea_tarjetas} Tarjetas', 0.0),
                f"Under {linea_tarjetas} Tarjetas": resultados.get('Tarjetas_Totales', {}).get(f'Under {linea_tarjetas} Tarjetas', 0.0)
            }

            prob_ml_dict = {
                "Gana Local": prob_ml_local,
                "Empate": prob_ml_empate,
                "Gana Visita": prob_ml_visita,
                f"Over {linea_goles} Goles": prob_ml_over_g,
                f"Under {linea_goles} Goles": prob_ml_under_g,
                f"Over {linea_corners} Corners": prob_ml_over_c,
                f"Under {linea_corners} Corners": prob_ml_under_c,
                f"Over {linea_tarjetas} Tarjetas": prob_ml_over_t,
                f"Under {linea_tarjetas} Tarjetas": prob_ml_under_t
            }

            mercados_a_mapear = [
                ("Gana Local", "1"),
                ("Empate", "X"),
                ("Gana Visita", "2"),
                (f"Over {linea_goles} Goles", "Over_Goles"),
                (f"Under {linea_goles} Goles", "Under_Goles"),
                (f"Over {linea_corners} Corners", "Over_Corners"),
                (f"Under {linea_corners} Corners", "Under_Corners"),
                (f"Over {linea_tarjetas} Tarjetas", "Over_Tarjetas"),
                (f"Under {linea_tarjetas} Tarjetas", "Under_Tarjetas")
            ]

            for nombre_m, llave_api in mercados_a_mapear:
                p_mc = float(prob_mc_dict.get(nombre_m, 0.0))
                p_ml = float(prob_ml_dict.get(nombre_m, 0.0))
                
                raw_cuota = cuotas.get(llave_api)
                try:
                    cuota = float(raw_cuota) if raw_cuota is not None else 0.0
                except:
                    cuota = 0.0

                if cuota > 1.40 and p_mc >= 58.0 and p_ml >= 58.0:
                    prob_combinada = round((p_mc + p_ml) / 2.0, 1)
                    prob_implicita_casa = (1.0 / cuota) * 100.0
                    diferencia_anomala = prob_combinada - prob_implicita_casa
                    
                    if diferencia_anomala > 40.0:
                        continue 
                    
                    ev_real = ((prob_combinada / 100.0) * cuota) - 1.0
                    ev_porcentaje = ev_real * 100.0
                    
                    if ev_porcentaje >= 2.0:
                        stake_recomendado = (ev_real / (cuota - 1.0)) * 10.0 
                        stake_recomendado = max(0.5, min(stake_recomendado, 3.0)) 
                        
                        oportunidades_oro.append({
                            "Fecha": fecha,
                            "Partido": f"{local} vs {visita}",
                            "Mercado": nombre_m,
                            "P. Montecarlo": f"{p_mc}%",
                            "P. ML": f"{p_ml}%",
                            "Cuota": f"{cuota:.2f}",
                            "EV (Valor)": f"+{ev_porcentaje:.1f}%",
                            "Stake Rec.": f"{stake_recomendado:.1f}%",
                            "Veredicto": "✅ CONSENSO BLINDADO"
                        })
        except Exception:
            continue
            
    st.write("✅ **Escaneo completado con éxito.**")
    return oportunidades_oro
