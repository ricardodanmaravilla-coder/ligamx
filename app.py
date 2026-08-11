import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
import datetime

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
def obtener_proximos_partidos_espn(liga_espn="mex.1"):
    """Descarga la jornada completa desde ESPN forzando una ventana de 15 días."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{liga_espn}/scoreboard"
    
    hoy = datetime.date.today()
    futuro = hoy + datetime.timedelta(days=15)
    rango_fechas = f"{hoy.strftime('%Y%m%d')}-{futuro.strftime('%Y%m%d')}"
    
    params = {
        "limit": 10,  
        "dates": rango_fechas
    }
    
    partidos_dict = {}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for event in data.get('events', []):
                fecha = event.get('date', '')[:10]
                competencia = event.get('competitions', [{}])[0]
                fix_id = event.get('id')
                
                competitors = competencia.get('competitors', [])
                local, visita = "", ""
                
                for comp in competitors:
                    team_name = comp.get('team', {}).get('displayName', '')
                    if comp.get('homeAway') == 'home':
                        local = team_name
                    else:
                        visita = team_name
                
                if local and visita:
                    llave = f"📅 {fecha} | {local} vs {visita}"
                    partidos_dict[llave] = {
                        "local": local,
                        "visita": visita,
                        "fixture_id": fix_id
                    }
    except Exception as e:
        st.error(f"Error conectando a la API de ESPN: {e}")
        
    return partidos_dict

st.title("⚽ Liga MX Analytics & Value Betting (2026)")
st.write("Simulador Montecarlo (Goles, Corners, Tarjetas) + Criterio de Kelly")

# --- SECCIÓN 1: INGRESO MASIVO DE TODA LA JORNADA (TODOS LOS MERCADOS + OVERS/UNDERS) ---
st.markdown("### 🏆 Ingreso Masivo de Cuotas (Toda la Jornada - Over y Under)")
st.info("Ingresa las cuotas reales de tu casino para el 1X2 y las líneas de Over/Under en Goles, Córners y Tarjetas.")

partidos_reales = obtener_proximos_partidos_espn("mex.1")

if not partidos_reales:
    st.warning("⚠️ No se encontraron partidos próximos en la cartelera de ESPN para la Liga MX.")
else:
    jornada_data = {}
    
    with st.form("form_jornada_completa"):
        for i, (nombre_llave, info) in enumerate(partidos_reales.items()):
            st.markdown(f"#### ⚽ {info['local']} vs {info['visita']}")
            
            # Fila 1: Resultado 1X2
            c1, c2, c3 = st.columns(3)
            val_1 = c1.number_input(f"Local (1)", value=2.00, step=0.05, format="%.2f", key=f"j_1_{i}")
            val_x = c2.number_input(f"Empate (X)", value=3.20, step=0.05, format="%.2f", key=f"j_x_{i}")
            val_2 = c3.number_input(f"Visita (2)", value=3.10, step=0.05, format="%.2f", key=f"j_2_{i}")
            
            # Fila 2: Goles (Línea + Over + Under)
            c4, c5, c6 = st.columns(3)
            val_lg = c4.number_input(f"Línea Goles", value=2.5, step=0.5, format="%.1f", key=f"j_lgoles_{i}")
            val_og = c5.number_input(f"Over Goles", value=1.90, step=0.05, format="%.2f", key=f"j_ogoles_{i}")
            val_ug = c6.number_input(f"Under Goles", value=1.90, step=0.05, format="%.2f", key=f"j_ugoles_{i}")

            # Fila 3: Córners (Línea + Over + Under)
            c7, c8, c9 = st.columns(3)
            val_lc = c7.number_input(f"Línea Córners", value=9.5, step=0.5, format="%.1f", key=f"j_lcorn_{i}")
            val_oc = c8.number_input(f"Over Córners", value=1.90, step=0.05, format="%.2f", key=f"j_ocorn_{i}")
            val_uc = c9.number_input(f"Under Córners", value=1.90, step=0.05, format="%.2f", key=f"j_ucorn_{i}")

            # Fila 4: Tarjetas (Línea + Over + Under)
            c10, c11, c12 = st.columns(3)
            val_lt = c10.number_input(f"Línea Tarjetas", value=4.5, step=0.5, format="%.1f", key=f"j_ltar_{i}")
            val_ot = c11.number_input(f"Over Tarjetas", value=1.90, step=0.05, format="%.2f", key=f"j_otar_{i}")
            val_ut = c12.number_input(f"Under Tarjetas", value=1.90, step=0.05, format="%.2f", key=f"j_utar_{i}")

            jornada_data[nombre_llave] = {
                "local": info['local'],
                "visita": info['visita'],
                "1": val_1, "X": val_x, "2": val_2,
                "Linea_Goles": val_lg, "Over_Goles": val_og, "Under_Goles": val_ug,
                "Linea_Corners": val_lc, "Over_Corners": val_oc, "Under_Corners": val_uc,
                "Linea_Tarjetas": val_lt, "Over_Tarjetas": val_ot, "Under_Tarjetas": val_ut
            }
            st.markdown("---")
            
        btn_calcular_jornada = st.form_submit_button("🚀 Calcular Valor (EV) de Toda la Jornada", type="primary")

    # --- PROCESAMIENTO DE LA JORNADA COMPLETA ---
    if btn_calcular_jornada:
        with st.spinner("Ejecutando simulaciones de Montecarlo para toda la jornada..."):
            df_hist_base = cargar_historico_seguro()
            motor_elo_temp = SistemaEloLigaMX()
            tabla_elo_temp = motor_elo_temp.calcular_historico(df_hist_base)
            
            consolizado_apuestas = []
            
            for nombre_llave, datos in jornada_data.items():
                loc = datos['local']
                vis = datos['visita']
                
                try:
                    e_loc = float(tabla_elo_temp.loc[tabla_elo_temp['Equipo'] == loc, 'ELO_Rating'].values[0])
                except:
                    e_loc = 1500.0
                try:
                    e_vis = float(tabla_elo_temp.loc[tabla_elo_temp['Equipo'] == vis, 'ELO_Rating'].values[0])
                except:
                    e_vis = 1500.0

                l_goles = datos['Linea_Goles']
                l_corners = datos['Linea_Corners']
                l_tarjetas = datos['Linea_Tarjetas']
                
                resultados = simular_partido_montecarlo(
                    loc, vis,
                    df_historico=df_hist_base,
                    elo_local=e_loc,
                    elo_visita=e_vis,
                    linea_goles=l_goles,
                    linea_corners=l_corners,
                    linea_tarjetas=l_tarjetas
                )
                
                if isinstance(resultados, str):
                    continue
                
                cuotas_dict = {
                    "1": datos['1'],
                    "X": datos['X'],
                    "2": datos['2'],
                    "Over_Goles": datos['Over_Goles'],
                    "Under_Goles": datos['Under_Goles'],
                    "Over_Corners": datos['Over_Corners'],
                    "Under_Corners": datos['Under_Corners'],
                    "Over_Tarjetas": datos['Over_Tarjetas'],
                    "Under_Tarjetas": datos['Under_Tarjetas']
                }
                
                lineas_default = {"Linea_Goles": l_goles, "Linea_Corners": l_corners, "Linea_Tarjetas": l_tarjetas}
                
                df_apuestas = analizar_apuestas(
                    resultados, loc, vis, 
                    cuotas_personalizadas=cuotas_dict, 
                    lineas_default=lineas_default
                )
                
                if not df_apuestas.empty:
                    df_apuestas.insert(0, "Partido", f"{loc} vs {vis}")
                    consolizado_apuestas.append(df_apuestas)
            
            if consolizado_apuestas:
                tabla_maestra = pd.concat(consolizado_apuestas, ignore_index=True)
                
                st.subheader("📊 Tabla Maestra de Oportunidades (Jornada Completa)")
                st.markdown("Resultados evaluados con tus cuotas manuales (incluyendo Unders).")
                
                def color_veredicto(val):
                    if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                    elif '✅' in str(val): return 'color: #adff2f'
                    elif '⚠️' in str(val): return 'color: #ffa500'
                    elif '❌' in str(val): return 'color: #ff4d4d'
                    return ''
                    
                st.dataframe(
                    tabla_maestra.style.map(color_veredicto, subset=['Veredicto']), 
                    width='stretch',
                    hide_index=True
                )
            else:
                st.warning("No se pudieron generar los análisis. Revisa los datos ingresados.")

st.markdown("---")

# --- SECCIÓN 2: ANÁLISIS DETALLADO INDIVIDUAL + MACHINE LEARNING ---
st.markdown("### 🔍 Análisis Detallado por Partido (Individual + Machine Learning)")
if partidos_reales:
    seleccion_ind = st.selectbox("Selecciona un partido para análisis profundo:", list(partidos_reales.keys()), key="select_individual")
    datos_partido = partidos_reales[seleccion_ind]

    cuotas_automaticas = obtener_cuotas_partido(datos_partido["local"], datos_partido["visita"])
    linea_goles_ind = cuotas_automaticas.get("Linea_Goles", 2.5) if cuotas_automaticas else 2.5
    linea_corners_ind = cuotas_automaticas.get("Linea_Corners", 9.5) if cuotas_automaticas else 9.5
    linea_tarjetas_ind = cuotas_automaticas.get("Linea_Tarjetas", 4.5) if cuotas_automaticas else 4.5

    if st.button("Ejecutar Análisis Profundo del Partido", type="primary", key="btn_ind"):
        with st.spinner('Calculando Montecarlo, ELO y Machine Learning...'):
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

                resultados = simular_partido_montecarlo(
                    datos_partido["local"], datos_partido["visita"],
                    df_historico=df_hist_base, elo_local=e_loc, elo_visita=e_vis,
                    linea_goles=linea_goles_ind, linea_corners=linea_corners_ind, linea_tarjetas=linea_tarjetas_ind
                )
                
                if isinstance(resultados, str):
                    st.error(f"🚨 Problema con los datos: {resultados}")
                else:
                    st.subheader("📊 Probabilidades Reales (Montecarlo)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"Victoria {datos_partido['local']}", f"{resultados['Resultado_1X2']['Gana Local']}%")
                    col2.metric("Empate", f"{resultados['Resultado_1X2']['Empate']}%")
                    col3.metric(f"Victoria {datos_partido['visita']}", f"{resultados['Resultado_1X2']['Gana Visita']}%")
                    
                    st.markdown("---")

                    ml_predictor = PredictorML()
                    if ml_predictor.entrenar(df_hist_base):
                        g_l_sim = resultados['Goles_Individuales'][datos_partido['local']]['goles']
                        g_v_sim = resultados['Goles_Individuales'][datos_partido['visita']]['goles']
                        
                        preds_ml = ml_predictor.predecir_mercados_completos(
                            df_hist_base, datos_partido['local'], datos_partido['visita'], 
                            g_l_sim, g_v_sim, e_loc, e_vis,
                            linea_goles=linea_goles_ind, linea_corners=linea_corners_ind, linea_tarjetas=linea_tarjetas_ind
                        )
                        
                        st.subheader("🤖 Predicciones Independientes de Machine Learning (Random Forest)")
                        if "Resultado_1X2" in preds_ml:
                            ml_c1, ml_c2, ml_c3 = st.columns(3)
                            ml_c1.metric("ML Local", f"{preds_ml['Resultado_1X2']['Gana Local']}%")
                            ml_c2.metric("ML Empate", f"{preds_ml['Resultado_1X2']['Empate']}%")
                            ml_c3.metric("ML Visita", f"{preds_ml['Resultado_1X2']['Gana Visita']}%")

                    st.subheader("🤖 Predicciones Híbridas (Poisson + ELO)")
                    df_predicciones = pd.DataFrame([{
                        'Local': datos_partido['local'],
                        'Visitante': datos_partido['visita'],
                        'Probabilidad_Local': resultados['Resultado_1X2']['Gana Local'] / 100.0
                    }])
                    df_predicciones = df_predicciones.merge(tabla_elo_temp, left_on='Local', right_on='Equipo', how='left').rename(columns={'ELO_Rating': 'ELO_Local'}).drop('Equipo', axis=1, errors='ignore')
                    df_predicciones = df_predicciones.merge(tabla_elo_temp, left_on='Visitante', right_on='Equipo', how='left').rename(columns={'ELO_Rating': 'ELO_Visita'}).drop('Equipo', axis=1, errors='ignore')

                    def evaluar_apuesta_hibrida(fila):
                        prob_poisson_local = fila['Probabilidad_Local']
                        el_loc = fila.get('ELO_Local', 1500)
                        el_vis = fila.get('ELO_Visita', 1500)
                        if prob_poisson_local > 0.55 and el_loc > el_vis: return "✅ Aprobada: Doble Validación"
                        elif prob_poisson_local > 0.55 and el_vis > el_loc: return "⚠️ Alerta: El visitante trae mejor racha"
                        else: return "Paso"

                    df_predicciones['Veredicto_Hibrido'] = df_predicciones.apply(evaluar_apuesta_hibrida, axis=1)
                    st.dataframe(df_predicciones[['Local', 'Visitante', 'Probabilidad_Local', 'ELO_Local', 'ELO_Visita', 'Veredicto_Hibrido']], width='stretch')

            except Exception as e:
                st.error(f"Ocurrió un error en el análisis detallado: {e}")

st.markdown("---")

# --- SECCIÓN 3: RANKING ELO ---
st.subheader("📊 Ranking de Poder ELO (Fuerza Actual)")
try:
    df_historico = cargar_historico_seguro()
    correccion_equipos = {
        "Atletico San Luis": "Atlético de San Luis", "Atlético San Luis": "Atlético de San Luis",
        "San Luis": "Atlético de San Luis", "Mazatlan": "Mazatlán", "Mazatlan FC": "Mazatlán",
        "Queretaro": "Querétaro", "Leon": "León"
    }
    df_historico['Local'] = df_historico['Local'].replace(correccion_equipos)
    df_historico['Visitante'] = df_historico['Visitante'].replace(correccion_equipos)

    motor_elo = SistemaEloLigaMX()
    tabla_posiciones_elo = motor_elo.calcular_historico(df_historico)
    st.dataframe(tabla_posiciones_elo, width='stretch')
except Exception as e:
    st.info("ℹ️ Cargando ranking ELO...")
