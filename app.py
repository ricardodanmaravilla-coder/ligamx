import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
from modules.elo_engine import SistemaEloLigaMX
from modules.ml_engine import PredictorML
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido, analizar_apuestas

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Liga MX & Analítica de Apuestas", layout="wide")

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262
LEAGUES_CUP_ID = 772 # ID oficial actualizado de Leagues Cup

def cargar_historico_seguro():
    url_github_raw = 'https://raw.githubusercontent.com/ricardodanmaravilla-coder/ligamx/main/data/historico_ligamx_completo.csv'
    rutas_locales = ['data/historico_ligamx_completo.csv', 'historico_ligamx_completo.csv']
    
    df = None
    for r in rutas_locales:
        if os.path.exists(r):
            try:
                df = pd.read_csv(r)
                break
            except Exception:
                pass
                
    if df is None:
        try:
            df = pd.read_csv(url_github_raw)
        except Exception as e:
            st.error(f"Error crítico al cargar el archivo histórico: {e}")
            return pd.DataFrame()
        
    df['Local'] = df['Local'].str.strip()
    df['Visitante'] = df['Visitante'].str.strip()
    return df

@st.cache_data(ttl=3600)
def obtener_proximos_partidos():
    """Descarga los próximos partidos de la Liga MX de forma segura"""
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(LIGA_MX_ID), "season": "2026", "next": "10"} 
    
    response = requests.get(url, headers=HEADERS, params=querystring)
    if response.status_code != 200:
        return {}
        
    datos = response.json().get("response", [])
    
    if not datos:
        querystring_fallback = {"league": str(LIGA_MX_ID), "next": "10"}
        response = requests.get(url, headers=HEADERS, params=querystring_fallback)
        if response.status_code == 200:
            datos = response.json().get("response", [])

    partidos_dict = {}
    for p in datos:
        local = p["teams"]["home"]["name"]
        visita = p["teams"]["away"]["name"]
        fix_id = p["fixture"]["id"]
        fecha = p["fixture"]["date"][:10]
        
        llave = f"📅 {fecha} | {local} vs {visita}"
        partidos_dict[llave] = {
            "local": local,
            "visita": visita,
            "fixture_id": fix_id
        }
    return partidos_dict

# --- FUNCIONES PARA LEAGUES CUP ---
def cargar_historico_lc():
    url_github_raw = 'https://raw.githubusercontent.com/ricardodanmaravilla-coder/ligamx/main/data/historico_leaguescup.csv'
    rutas_locales = ['data/historico_leaguescup.csv', 'historico_leaguescup.csv']
    df = None
    for r in rutas_locales:
        if os.path.exists(r):
            try:
                df = pd.read_csv(r)
                break
            except Exception:
                pass
    if df is None:
        try:
            df = pd.read_csv(url_github_raw)
        except Exception:
            return pd.DataFrame()
    df['Local'] = df['Local'].str.strip()
    df['Visitante'] = df['Visitante'].str.strip()
    return df

@st.cache_data(ttl=3600)
def obtener_proximos_partidos_lc():
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(LEAGUES_CUP_ID), "season": "2026", "next": "15"} 
    response = requests.get(url, headers=HEADERS, params=querystring)
    if response.status_code != 200:
        return {}
    datos = response.json().get("response", [])
    if not datos:
        querystring_fallback = {"league": str(LEAGUES_CUP_ID), "next": "15"}
        response = requests.get(url, headers=HEADERS, params=querystring_fallback)
        if response.status_code == 200:
            datos = response.json().get("response", [])
    partidos_dict = {}
    for p in datos:
        local = p["teams"]["home"]["name"]
        visita = p["teams"]["away"]["name"]
        fix_id = p["fixture"]["id"]
        fecha = p["fixture"]["date"][:10]
        llave = f"🏆 {fecha} | {local} vs {visita}"
        partidos_dict[llave] = {"local": local, "visita": visita, "fixture_id": fix_id}
    return partidos_dict
# -----------------------------------

st.title("⚽ Liga MX Analytics & Value Betting (2026)")
st.write("Simulador Montecarlo (Goles, Corners, Tarjetas) + Criterio de Kelly")

# --- SISTEMA DE PESTAÑAS ---
tab1, tab2 = st.tabs(["🇲🇽 Liga MX", "🏆 Leagues Cup"])

# ==========================================
# PESTAÑA 1: LIGA MX
# ==========================================
with tab1:
    st.markdown("### 1. Selecciona el Encuentro")
    partidos_reales = obtener_proximos_partidos()

    if not partidos_reales:
        st.warning("⚠️ No se encontraron partidos próximos en la API para la Liga MX.")
    else:
        seleccion = st.selectbox("Próximos partidos de Liga MX:", list(partidos_reales.keys()))
        datos_partido = partidos_reales[seleccion]

        if st.button("Ejecutar Simulación y Buscar Cuotas", type="primary") or st.session_state.get('simulacion_activa', False):
            st.session_state['simulacion_activa'] = True

            with st.spinner('Procesando simulación y cuotas en tiempo real...'):
                try:
                    df_hist_base = cargar_historico_seguro()
                    
                    motor_elo_temp = SistemaEloLigaMX()
                    tabla_elo_temp = motor_elo_temp.calcular_historico(df_hist_base)
                    
                    try:
                        e_loc = float(tabla_elo_temp.loc[tabla_elo_temp['Equipo'] == datos_partido['local'], 'ELO_Rating'].values[0])
                    except:
                        e_loc = 1500.0
                    try:
                        e_vis = float(tabla_elo_temp.loc[tabla_elo_temp['Equipo'] == datos_partido['visita'], 'ELO_Rating'].values[0])
                    except:
                        e_vis = 1500.0

                    # --- EXTRACCIÓN DE LÍNEAS REALES PREVIA A LA SIMULACIÓN ---
                    cuotas_automaticas = obtener_cuotas_partido(datos_partido["local"], datos_partido["visita"])
                    linea_goles = cuotas_automaticas.get("Linea_Goles", 2.5) if cuotas_automaticas else 2.5
                    linea_corners = cuotas_automaticas.get("Linea_Corners", 9.5) if cuotas_automaticas else 9.5
                    linea_tarjetas = cuotas_automaticas.get("Linea_Tarjetas", 4.5) if cuotas_automaticas else 4.5

                    resultados = simular_partido_montecarlo(
                        datos_partido["local"], 
                        datos_partido["visita"],
                        df_historico=df_hist_base,
                        elo_local=e_loc,
                        elo_visita=e_vis,
                        linea_goles=linea_goles,
                        linea_corners=linea_corners,
                        linea_tarjetas=linea_tarjetas
                    )
                    
                    if isinstance(resultados, str):
                        st.error(f"🚨 Problema con los datos: {resultados}")
                    else:
                        st.subheader("📊 Probabilidades Reales (Montecarlo)")
                        
                        st.markdown("**🏆 Resultado del Encuentro (1X2)**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"Victoria {datos_partido['local']}", f"{resultados['Resultado_1X2']['Gana Local']}%")
                        col2.metric("Empate", f"{resultados['Resultado_1X2']['Empate']}%")
                        col3.metric(f"Victoria {datos_partido['visita']}", f"{resultados['Resultado_1X2']['Gana Visita']}%")
                        
                        st.markdown("---")
                        
                        st.markdown("🎯 **Goles, Corners y Tarjetas Más Probables del Partido**")
                        col4, col5, col6 = st.columns(3)
                        
                        over_goles = resultados['Goles_Over_Under'][f'Over {linea_goles}']
                        col4.metric(f"Más de {linea_goles} Goles", f"{over_goles}%", f"Under: {round(100-over_goles, 2)}%")
                        
                        over_corners = resultados['Corners_Totales'][f'Over {linea_corners} Corners']
                        col5.metric(f"Más de {linea_corners} Corners", f"{over_corners}%", f"Under: {round(100-over_corners, 2)}%")
                        
                        over_tarjetas = resultados['Tarjetas_Totales'][f'Over {linea_tarjetas} Tarjetas']
                        col6.metric(f"Más de {linea_tarjetas} Tarjetas", f"{over_tarjetas}%", f"Under: {round(100-over_tarjetas, 2)}%")
                        
                        st.markdown("---")
                        st.markdown("📈 **Pronósticos Detallados por Equipo (Expectativa Matemática)**")
                        
                        g_l_ind = round(resultados['Goles_Individuales'][datos_partido['local']]['goles'])
                        g_v_ind = round(resultados['Goles_Individuales'][datos_partido['visita']]['goles'])
                        c_l_ind = round(resultados['Corners_Individuales'][datos_partido['local']]['corners'])
                        c_v_ind = round(resultados['Corners_Individuales'][datos_partido['visita']]['corners'])
                        t_l_ind = round(resultados['Tarjetas_Individuales'][datos_partido['local']]['tarjetas'])
                        t_v_ind = round(resultados['Tarjetas_Individuales'][datos_partido['visita']]['tarjetas'])

                        ind_col1, ind_col2, ind_col3 = st.columns(3)
                        with ind_col1:
                            st.markdown(f"⚽ **Goles Esperados**")
                            st.write(f"- {datos_partido['local']}: **{g_l_ind}** goles")
                            st.write(f"- {datos_partido['visita']}: **{g_v_ind}** goles")
                        with ind_col2:
                            st.markdown(f"🚩 **Córners Esperados**")
                            st.write(f"- {datos_partido['local']}: **{c_l_ind}** córners")
                            st.write(f"- {datos_partido['visita']}: **{c_v_ind}** córners")
                        with ind_col3:
                            st.markdown(f"🟨 **Tarjetas Esperadas**")
                            st.write(f"- {datos_partido['local']}: **{t_l_ind}** puntos")
                            st.write(f"- {datos_partido['visita']}: **{t_v_ind}** puntos")

                        st.markdown("---")
                        
                        with st.container():
                            st.markdown("⚙️ **Gestión de Cuotas (Automáticas / Manuales)**")
                            mercados_keys = {
                                "Gana Local": "1", 
                                "Empate": "X",
                                "Gana Visita": "2", 
                                f"Over {linea_goles} Goles": "Over_Goles", 
                                f"Under {linea_goles} Goles": "Under_Goles",
                                f"Over {linea_corners} Corners": "Over_Corners",
                                f"Under {linea_corners} Corners": "Under_Corners",
                                f"Over {linea_tarjetas} Tarjetas": "Over_Tarjetas",
                                f"Under {linea_tarjetas} Tarjetas": "Under_Tarjetas"
                            }
                            
                            cuotas_usuario = {}
                            cols = st.columns(3)
                            
                            for i, (nombre_m, llave) in enumerate(mercados_keys.items()):
                                val_default = cuotas_automaticas.get(llave) if cuotas_automaticas and cuotas_automaticas.get(llave) else 0.0
                                with cols[i % 3]:
                                    cuotas_usuario[llave] = st.number_input(
                                        f"{nombre_m}", 
                                        min_value=0.0, 
                                        value=float(val_default), 
                                        step=0.05,
                                        format="%.2f",
                                        key=f"input_cuota_mx_{llave}"
                                    )

                        lineas_default = {"Linea_Goles": linea_goles, "Linea_Corners": linea_corners, "Linea_Tarjetas": linea_tarjetas}
                        df_apuestas = analizar_apuestas(resultados, datos_partido["fixture_id"], cuotas_personalizadas=cuotas_usuario, lineas_default=lineas_default)
                        
                        if not df_apuestas.empty:
                            def color_veredicto(val):
                                if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                                elif '✅' in str(val): return 'color: #adff2f'
                                elif '⚠️' in str(val): return 'color: #ffa500'
                                elif '❌' in str(val): return 'color: #ff4d4d'
                                return ''
                                
                            st.dataframe(
                                df_apuestas.style.map(color_veredicto, subset=['Veredicto']), 
                                use_container_width=True,
                                hide_index=True
                            )
                        
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado durante la simulación: {e}")

    st.markdown("---")

    with st.expander("🚨 Escáner Automático de Oportunidades (Jornada Completa)", expanded=False):
        st.info("Este escáner analiza todos los partidos de la próxima jornada de la Liga MX de golpe y filtra exclusivamente las jugadas de valor.")
        
        if st.button("🔍 Ejecutar Escáner de Jornada", key="btn_scanner_mx"):
            with st.spinner("Analizando la jornada completa con Montecarlo... Esto puede tomar unos segundos."):
                from modules.scanner_engine import escanear_jornada_actual
                df_oro = pd.DataFrame(escanear_jornada_actual())
                
                if not df_oro.empty:
                    st.success(f"¡Se encontraron {len(df_oro)} oportunidades de alta probabilidad con valor!")
                    def color_veredicto_oro(val):
                        if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                        elif '✅' in str(val): return 'color: #adff2f'
                        return ''
                    st.dataframe(
                        df_oro.style.map(color_veredicto_oro, subset=['Veredicto']), 
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No hay partidos próximos en la jornada con los criterios requeridos en este momento.")

    st.markdown("---")

    st.subheader("📊 Ranking de Poder ELO (Fuerza Actual)")

    try:
        df_historico = cargar_historico_seguro()

        correccion_equipos = {
            "Atletico San Luis": "Atlético de San Luis",
            "Atlético San Luis": "Atlético de San Luis",
            "San Luis": "Atlético de San Luis",
            "Mazatlan": "Mazatlán",
            "Mazatlan FC": "Mazatlán",
            "Queretaro": "Querétaro",
            "Leon": "León"
        }

        df_historico['Local'] = df_historico['Local'].replace(correccion_equipos)
        df_historico['Visitante'] = df_historico['Visitante'].replace(correccion_equipos)

        motor_elo = SistemaEloLigaMX()
        tabla_posiciones_elo = motor_elo.calcular_historico(df_historico)
        st.dataframe(tabla_posiciones_elo, use_container_width=True)

        ml_predictor = PredictorML()
        if ml_predictor.entrenar(df_historico):
            if 'resultados' in locals() and isinstance(resultados, dict):
                g_l_sim = resultados['Goles_Individuales'][datos_partido['local']]['goles']
                g_v_sim = resultados['Goles_Individuales'][datos_partido['visita']]['goles']
                
                try:
                    puntos_local = float(tabla_posiciones_elo.loc[tabla_posiciones_elo['Equipo'] == datos_partido['local'], 'ELO_Rating'].values[0])
                except (IndexError, KeyError):
                    puntos_local = 1500.0
                    
                try:
                    puntos_visita = float(tabla_posiciones_elo.loc[tabla_posiciones_elo['Equipo'] == datos_partido['visita'], 'ELO_Rating'].values[0])
                except (IndexError, KeyError):
                    puntos_visita = 1500.0
                
                # (Dentro de la sección de Machine Learning)
                preds_ml = ml_predictor.predecir_mercados_completos(
                    df_historico, 
                    datos_partido['local'], 
                    datos_partido['visita'], 
                    g_l_sim, 
                    g_v_sim,
                    puntos_local,  
                    puntos_visita,
                    linea_goles=linea_goles,
                    linea_corners=linea_corners,
                    linea_tarjetas=linea_tarjetas 
                )
                
                st.markdown("---")
                st.subheader("🤖 Predicciones Independientes de Machine Learning (Random Forest)")
                
                if "Resultado_1X2" in preds_ml:
                    st.markdown("##### 🏆 Mercado 1X2 (Ganador del Partido)")
                    ml_col1, ml_col2, ml_col3 = st.columns(3)
                    ml_col1.metric("Local", f"{preds_ml['Resultado_1X2']['Gana Local']}%")
                    ml_col2.metric("Empate", f"{preds_ml['Resultado_1X2']['Empate']}%")
                    ml_col3.metric("Visita", f"{preds_ml['Resultado_1X2']['Gana Visita']}%")
                    
                    st.write("")
                    
                    st.markdown("##### 📊 Mercados de Goles, Córners y Tarjetas")
                    ml_col4, ml_col5, ml_col6 = st.columns(3)
                    
                    with ml_col4:
                        st.metric(f"Over {linea_goles} Goles", f"{preds_ml['Goles_Over_Under'][f'Over {linea_goles}']}%")
                        st.metric(f"Under {linea_goles} Goles", f"{preds_ml['Goles_Over_Under'][f'Under {linea_goles}']}%")
                        
                    with ml_col5:
                        st.metric(f"Over {linea_corners} Córners", f"{preds_ml['Corners_Totales'][f'Over {linea_corners} Corners']}%")
                        st.metric(f"Under {linea_corners} Córners", f"{preds_ml['Corners_Totales'][f'Under {linea_corners} Corners']}%")
                        
                    with ml_col6:
                        st.metric(f"Over {linea_tarjetas} Tarjetas", f"{preds_ml['Tarjetas_Totales'][f'Over {linea_tarjetas} Tarjetas']}%")
                        st.metric(f"Under {linea_tarjetas} Tarjetas", f"{preds_ml['Tarjetas_Totales'][f'Under {linea_tarjetas} Tarjetas']}%")
                else:
                    st.warning("⚠️ El modelo ML no pudo generar predicciones. Revisa los datos históricos.")
        
        if 'resultados' in locals() and isinstance(resultados, dict):
            st.subheader("🤖 Predicciones Híbridas (Poisson + ELO)")
            
            df_predicciones = pd.DataFrame([{
                'Local': datos_partido['local'],
                'Visitante': datos_partido['visita'],
                'Probabilidad_Local': resultados['Resultado_1X2']['Gana Local'] / 100.0
            }])

            df_predicciones = df_predicciones.merge(
                tabla_posiciones_elo, left_on='Local', right_on='Equipo', how='left'
            ).rename(columns={'ELO_Rating': 'ELO_Local'}).drop('Equipo', axis=1, errors='ignore')

            df_predicciones = df_predicciones.merge(
                tabla_posiciones_elo, left_on='Visitante', right_on='Equipo', how='left'
            ).rename(columns={'ELO_Rating': 'ELO_Visita'}).drop('Equipo', axis=1, errors='ignore')

            def evaluar_apuesta_hibrida(fila):
                prob_poisson_local = fila['Probabilidad_Local']
                elo_local = fila.get('ELO_Local', 1500)
                elo_visita = fila.get('ELO_Visita', 1500)
                
                if prob_poisson_local > 0.55 and elo_local > elo_visita:
                    return "✅ Aprobada: Doble Validación"
                elif prob_poisson_local > 0.55 and elo_visita > elo_local:
                    return "⚠️ Alerta: El visitante trae mejor racha"
                else:
                    return "Paso"

            df_predicciones['Veredicto_Hibrido'] = df_predicciones.apply(evaluar_apuesta_hibrida, axis=1)
            st.dataframe(df_predicciones[['Local', 'Visitante', 'Probabilidad_Local', 'ELO_Local', 'ELO_Visita', 'Veredicto_Hibrido']], use_container_width=True)

    except Exception as e:
        st.info("ℹ️ Ejecuta la simulación de arriba para ver el cruce híbrido detallado con ELO.")
