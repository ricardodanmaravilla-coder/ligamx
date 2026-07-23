import streamlit as st
import pandas as pd
import requests
import numpy as np
from modules.nfl_odds_engine import analizar_apuestas_nfl

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Deportivo & Analítica de Apuestas", layout="wide")

# --- MENÚ LATERAL PARA SELECCIONAR DEPORTE ---
st.sidebar.title("⚙️ Configuración")
deporte = st.sidebar.selectbox("🏟️ Selecciona el Deporte", ["⚽ Liga MX (Soccer)", "🏈 NFL (Football Americano)"])

# ==========================================
# SECCIÓN 1: LIGA MX (FÚTBOL)
# ==========================================
if deporte == "⚽ Liga MX (Soccer)":
    API_KEY = os.environ.get("API_SPORTS_KEY")
    BASE_URL = "https://v3.football.api-sports.io"
    HEADERS = {'x-apisports-key': API_KEY}
    LIGA_MX_ID = 262

    @st.cache_data(ttl=3600)
    def obtener_proximos_partidos():
        """Descarga los próximos partidos de la Liga MX de forma segura"""
        url = f"{BASE_URL}/fixtures"
        # Probamos primero con el año actual 2026 o el parámetro 'next' directo
        querystring = {"league": str(LIGA_MX_ID), "season": "2026", "next": "10"} 
        
        response = requests.get(url, headers=HEADERS, params=querystring)
        if response.status_code != 200:
            return {}
            
        datos = response.json().get("response", [])
        
        # Si la temporada 2026 no arroja partidos directos por calendario actual, intentamos sin el año estricto
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

    st.title("⚽ Liga MX Analytics & Value Betting (2026)")
    st.write("Simulador Montecarlo (Goles, Corners, Tarjetas) + Criterio de Kelly")

    st.markdown("### 1. Selecciona el Encuentro")
    partidos_reales = obtener_proximos_partidos()

    if not partidos_reales:
        st.warning("⚠️ No se encontraron partidos próximos en la API para la Liga MX (es posible que sea fecha FIFA, pretemporada o fin de torneo). Puedes usar el respaldo histórico si lo requieres.")
    else:
        seleccion = st.selectbox("Próximos partidos de Liga MX:", list(partidos_reales.keys()))
        datos_partido = partidos_reales[seleccion]

        if st.button("Ejecutar Simulación y Buscar Cuotas", type="primary") or st.session_state.get('simulacion_activa', False):
            st.session_state['simulacion_activa'] = True

            with st.spinner('Procesando simulación y cuotas...'):
                try:
                    from modules.montecarlo_sim import simular_partido_montecarlo
                    resultados = simular_partido_montecarlo(
                        datos_partido["local"], 
                        datos_partido["visita"]
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
                        
                        st.markdown("🎯 **Goles, Corners y Tarjetas Más Probables por Equipo**")
                        g_loc, g_vis = datos_partido['local'], datos_partido['visita']
                        c_ind1, c_ind2, c_ind3 = st.columns(3)
                        
                        with c_ind1:
                            st.markdown(f"**⚽ Goles Exactos**")
                            st.write(f"• **{g_loc}:** {resultados['Goles_Individuales'][g_loc]['goles']} goles ({resultados['Goles_Individuales'][g_loc]['prob']}%)")
                            st.write(f"• **{g_vis}:** {resultados['Goles_Individuales'][g_vis]['goles']} goles ({resultados['Goles_Individuales'][g_vis]['prob']}%)")
                            
                        with c_ind2:
                            st.markdown(f"**⛳ Corners Exactos**")
                            st.write(f"• **{g_loc}:** ~{resultados['Corners_Individuales'][g_loc]['corners']} corners ({resultados['Corners_Individuales'][g_loc]['prob']}%)")
                            st.write(f"• **{g_vis}:** ~{resultados['Corners_Individuales'][g_vis]['corners']} corners ({resultados['Corners_Individuales'][g_vis]['prob']}%)")
                            
                        with c_ind3:
                            st.markdown(f"**🟨 Tarjetas Exactas**")
                            st.write(f"• **{g_loc}:** ~{resultados['Tarjetas_Individuales'][g_loc]['tarjetas']} tarjetas ({resultados['Tarjetas_Individuales'][g_loc]['prob']}%)")
                            st.write(f"• **{g_vis}:** ~{resultados['Tarjetas_Individuales'][g_vis]['tarjetas']} tarjetas ({resultados['Tarjetas_Individuales'][g_vis]['prob']}%)")

                        st.markdown("---")
                        
                        st.markdown("**📈 Mercados Totales del Partido**")
                        col4, col5, col6 = st.columns(3)
                        over_goles = resultados['Goles_Over_Under']['Over 2.5']
                        col4.metric("Más de 2.5 Goles", f"{over_goles}%", f"Under: {round(100-over_goles, 2)}%")
                        over_corners = resultados['Corners_Totales']['Over 9.5 Corners']
                        col5.metric("Más de 9.5 Corners", f"{over_corners}%", f"Under: {round(100-over_corners, 2)}%")
                        over_tarjetas = resultados['Tarjetas_Totales']['Over 4.5 Tarjetas']
                        col6.metric("Más de 4.5 Tarjetas", f"{over_tarjetas}%", f"Under: {round(100-over_tarjetas, 2)}%")

                        st.markdown("---")
                        
                        st.subheader("💰 Análisis de Valor Inteligente (Kelly Criterion)")
                        from modules.odds_engine import obtener_cuotas_partido, analizar_apuestas
                        cuotas_automaticas = obtener_cuotas_partido(datos_partido["fixture_id"])
                        
                        with st.container():
                            st.markdown("⚙️ **Gestión de Cuotas (Automáticas / Manuales)**")
                            
                            mercados_keys = {
                                "Gana Local": "1", 
                                "Empate": "X",
                                "Gana Visita": "2", 
                                "Over 2.5 Goles": "Over 2.5", 
                                "Under 2.5 Goles": "Under 2.5",
                                "Over 9.5 Corners": "Over 9.5 Corners",
                                "Under 9.5 Corners": "Under 9.5 Corners",
                                "Over 4.5 Tarjetas": "Over 4.5 Tarjetas",
                                "Under 4.5 Tarjetas": "Under 4.5 Tarjetas"
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

                        df_apuestas = analizar_apuestas(resultados, datos_partido["fixture_id"], cuotas_personalizadas=cuotas_usuario)
                        
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
                            
                            estrellas = df_apuestas[df_apuestas["Veredicto"] == "🔥 APUESTA ESTRELLA"]
                            if len(estrellas) > 0:
                                st.success("¡Hay apuestas Estrella! Usa el Stake recomendado como porcentaje de tu bankroll.")
                        else:
                            st.warning("Ingresa las cuotas arriba para calcular el valor esperado.")
                            
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado durante la simulación: {e}")

    st.markdown("---")
    with st.expander("🚨 Escáner Automático de Oportunidades (Probabilidad > 60% + EV)", expanded=False):
        st.info("Este escáner analiza todos los partidos de la próxima jornada de la Liga MX de golpe y filtra exclusivamente las jugadas donde el modelo detecta más del 60% de probabilidad de acierto.")
        
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
                    st.warning("No hay partidos próximos en la jornada con más del 70% de probabilidad y valor positivo en este momento.")
# ==========================================
# SECCIÓN 2: NFL (FUTBOL AMERICANO)
# ==========================================
else:
    st.title("🏈 NFL Analytics & Value Betting (Calendario Histórico)")
    st.info("Simulador Montecarlo de Emparrillado + Criterio de Kelly con datos reales de la liga.")
    
    @st.cache_data
    def cargar_nfl():
        try:
            return pd.read_csv('data/historico_nfl.csv')
        except FileNotFoundError:
            return None

    df_nfl = cargar_nfl()

    if df_nfl is None or df_nfl.empty:
        st.error("No se encontró el archivo 'data/historico_nfl.csv'.")
    else:
        # Generar lista de partidos recientes reales del archivo CSV para simular un calendario activo
        df_nfl['Fecha'] = pd.to_datetime(df_nfl['Fecha'])
        df_recientes = df_nfl.sort_values(by='Fecha', ascending=False).head(50)
        
        partidos_nfl_dict = {}
        for _, row in df_recientes.iterrows():
            f_str = row['Fecha'].strftime('%Y-%m-%d')
            loc = row['Local']
            vis = row['Visitante']
            llave = f"📅 {f_str} | {loc} vs {vis}"
            partidos_nfl_dict[llave] = {"local": loc, "visita": vis}

        st.markdown("### 1. Selecciona el Encuentro de la NFL")
        seleccion_nfl = st.selectbox("Partidos de referencia histórica recientes:", list(partidos_nfl_dict.keys()))
        info_nfl = partidos_nfl_dict[seleccion_nfl]
        
        equipo_local_nfl = info_nfl["local"]
        equipo_visita_nfl = info_nfl["visita"]

        st.write(f"**Enfrentamiento seleccionado:** {equipo_local_nfl} (Local) vs {equipo_visita_nfl} (Visitante)")

        if st.button("Ejecutar Simulación NFL y Analizar Cuotas", type="primary"):
            # Filtrar estadísticas históricas
            l_loc = df_nfl[df_nfl['Local'] == equipo_local_nfl]
            l_vis = df_nfl[df_nfl['Visitante'] == equipo_local_nfl]
            v_loc = df_nfl[df_nfl['Local'] == equipo_visita_nfl]
            v_vis = df_nfl[df_nfl['Visitante'] == equipo_visita_nfl]
            
            puntos_favor_local = pd.concat([l_loc['Puntos_L'], l_vis['Puntos_V']]).mean()
            puntos_contra_local = pd.concat([l_loc['Puntos_V'], l_vis['Puntos_L']]).mean()
            puntos_favor_visita = pd.concat([v_loc['Puntos_L'], v_vis['Puntos_V']]).mean()
            puntos_contra_visita = pd.concat([v_loc['Puntos_V'], v_vis['Puntos_L']]).mean()

            # Simulación de Montecarlo (10,000 iteraciones)
            lambda_l = max(10.0, (puntos_favor_local + puntos_contra_visita) / 2)
            lambda_v = max(10.0, (puntos_favor_visita + puntos_contra_local) / 2)
            
            n_sims = 10000
            sim_puntos_l = np.random.normal(loc=lambda_l, scale=7.0, size=n_sims)
            sim_puntos_v = np.random.normal(loc=lambda_v, scale=7.0, size=n_sims)
            sim_totales = sim_puntos_l + sim_puntos_v
            
            wins_l = np.sum(sim_puntos_l > sim_puntos_v)
            wins_v = np.sum(sim_puntos_l < sim_puntos_v)
            
            p_l = (wins_l / n_sims) * 100
            p_v = (wins_v / n_sims) * 100
            
            # Promedio de línea Over/Under estimada basada en simulaciones
            linea_base_ou = 45.5
            p_over = (np.sum(sim_totales > linea_base_ou) / n_sims) * 100
            p_under = 100.0 - p_over

            resultados_nfl = {
                "prob_local": p_l,
                "prob_visita": p_v,
                "prob_over": p_over,
                "prob_under": p_under
            }

            st.markdown("---")
            st.subheader("📊 Probabilidades Reales (Montecarlo NFL)")
            rc1, rc2 = st.columns(2)
            rc1.metric(f"Victoria {equipo_local_nfl}", f"{p_l:.1f}%")
            rc2.metric(f"Victoria {equipo_visita_nfl}", f"{p_v:.1f}%")

            st.markdown("---")
            st.subheader("💰 Análisis de Valor Inteligente (Criterio de Kelly - NFL)")
            
            # Panel de entrada manual de cuotas para la NFL
            c_n1, c_n2, c_n3, c_n4 = st.columns(4)
            with c_n1:
                cuota_l = st.number_input(f"Cuota {equipo_local_nfl}", min_value=1.0, value=1.90, step=0.05, format="%.2f")
            with c_n2:
                cuota_v = st.number_input(f"Cuota {equipo_visita_nfl}", min_value=1.0, value=1.90, step=0.05, format="%.2f")
            with c_n3:
                cuota_over = st.number_input("Cuota Over Puntos", min_value=1.0, value=1.90, step=0.05, format="%.2f")
            with c_n4:
                cuota_under = st.number_input("Cuota Under Puntos", min_value=1.0, value=1.90, step=0.05, format="%.2f")

            cuotas_nfl_usuario = {
                "1": cuota_l,
                "2": cuota_v,
                "Over_Puntos": cuota_over,
                "Under_Puntos": cuota_under
            }

            df_apuestas_nfl = analizar_apuestas_nfl(resultados_nfl, cuotas_nfl_usuario)

            def color_veredicto_nfl(val):
                if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                elif '✅' in str(val): return 'color: #adff2f'
                elif '⚠️' in str(val): return 'color: #ffa500'
                elif '❌' in str(val): return 'color: #ff4d4d'
                return ''

            st.dataframe(df_apuestas_nfl.style.map(color_veredicto_nfl, subset=['Veredicto']), use_container_width=True, hide_index=True)
            
            estrellas_nfl = df_apuestas_nfl[df_apuestas_nfl["Veredicto"] == "🔥 APUESTA ESTRELLA"]
            if len(estrellas_nfl) > 0:
                st.success("🔥 ¡Apuesta Estrella detectada en la NFL con más del 70% de probabilidad y valor positivo!")
