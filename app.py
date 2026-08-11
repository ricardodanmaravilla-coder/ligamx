import os
import sys
import streamlit as st
import pandas as pd
import requests
import numpy as np
import datetime

# Blindaje de rutas para la carpeta modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modules.elo_engine import SistemaEloLigaMX
from modules.ml_engine import PredictorML
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import analizar_apuestas

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
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{liga_espn}/scoreboard"
    hoy = datetime.date.today()
    futuro = hoy + datetime.timedelta(days=15)
    rango_fechas = f"{hoy.strftime('%Y%m%d')}-{futuro.strftime('%Y%m%d')}"
    
    params = {"limit": 10, "dates": rango_fechas}
    partidos_dict = {}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for event in data.get('events', [])[:10]:
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
                    partidos_dict[llave] = {"local": local, "visita": visita, "fixture_id": fix_id}
    except Exception as e:
        st.error(f"Error conectando a la API de ESPN: {e}")
        
    return partidos_dict

st.title("⚽ Liga MX Analytics & Value Betting (2026)")
st.write("Simulador Montecarlo (Goles, Corners, Tarjetas) + Criterio de Kelly")

st.markdown("### 1. Selecciona el Encuentro")
partidos_reales = obtener_proximos_partidos_espn("mex.1")

if not partidos_reales:
    st.warning("⚠️ No se encontraron partidos próximos en la cartelera de ESPN para la Liga MX.")
else:
    seleccion = st.selectbox("Próximos partidos de Liga MX:", list(partidos_reales.keys()))
    datos_partido = partidos_reales[seleccion]

    # Líneas por defecto
    linea_goles = 2.5
    linea_corners = 9.5
    linea_tarjetas = 4.5

    # --- CAMPOS DE INPUTS DE CUOTAS VISIBLES DE INMEDIATO ---
    with st.container():
        st.markdown("⚙️ **Ingreso de Cuotas Reales (Tu Casino / Playdoit)**")
        st.info("Ajusta los valores con las cuotas exactas que ves en tu plataforma antes de ejecutar la simulación.")
        
        c1, c2, c3 = st.columns(3)
        cuota_1 = c1.number_input("Gana Local (1)", min_value=0.0, value=2.10, step=0.05, format="%.2f")
        cuota_x = c2.number_input("Empate (X)", min_value=0.0, value=3.30, step=0.05, format="%.2f")
        cuota_2 = c3.number_input("Gana Visita (2)", min_value=0.0, value=3.20, step=0.05, format="%.2f")

        c4, c5, c6 = st.columns(3)
        linea_goles = c4.number_input("Línea de Goles", min_value=0.5, value=2.5, step=0.5, format="%.1f")
        cuota_over_goles = c5.number_input("Cuota Over Goles", min_value=0.0, value=1.90, step=0.05, format="%.2f")
        cuota_under_goles = c6.number_input("Cuota Under Goles", min_value=0.0, value=1.90, step=0.05, format="%.2f")

        c7, c8 = st.columns(2)
        linea_corners = c7.number_input("Línea Córners", min_value=0.5, value=9.5, step=0.5, format="%.1f")
        cuota_over_corners = c8.number_input("Cuota Over Córners", min_value=0.0, value=1.85, step=0.05, format="%.2f")

        c9, c10 = st.columns(2)
        linea_tarjetas = c9.number_input("Línea Tarjetas", min_value=0.5, value=4.5, step=0.5, format="%.1f")
        cuota_over_tarjetas = c10.number_input("Cuota Over Tarjetas", min_value=0.0, value=1.90, step=0.05, format="%.2f")

    cuotas_usuario = {
        "1": cuota_1, "X": cuota_x, "2": cuota_2,
        "Over_Goles": cuota_over_goles, "Under_Goles": cuota_under_goles,
        "Over_Corners": cuota_over_corners, "Under_Corners": 1.90,
        "Over_Tarjetas": cuota_over_tarjetas, "Under_Tarjetas": 1.90
    }

    if st.button("🚀 Ejecutar Simulación y Calcular Valor (EV)", type="primary"):
        with st.spinner('Procesando simulación y calculando rentabilidad...'):
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

                    lineas_default = {"Linea_Goles": linea_goles, "Linea_Corners": linea_corners, "Linea_Tarjetas": linea_tarjetas}
                    df_apuestas = analizar_apuestas(
                        resultados, 
                        datos_partido["local"], 
                        datos_partido["visita"], 
                        cuotas_personalizadas=cuotas_usuario,
                        lineas_default=lineas_default
                    )
                    
                    if not df_apuestas.empty:
                        st.subheader("💰 Tabla de Valor (EV) y Criterio de Kelly")
                        def color_veredicto(val):
                            if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                            elif '✅' in str(val): return 'color: #adff2f'
                            elif '⚠️' in str(val): return 'color: #ffa500'
                            elif '❌' in str(val): return 'color: #ff4d4d'
                            return ''
                            
                        st.dataframe(
                            df_apuestas.style.map(color_veredicto, subset=['Veredicto']), 
                            width='stretch',
                            hide_index=True
                        )
                    
            except Exception as e:
                st.error(f"Ocurrió un error inesperado durante la simulación: {e}")

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
    st.dataframe(tabla_posiciones_elo, width='stretch')

except Exception as e:
    st.info("ℹ️ Cargando ranking ELO...")
