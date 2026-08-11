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

# --- SECCIÓN DE INGRESO MASIVO DE TODA LA JORNADA ---
st.markdown("### 🏆 Ingreso Masivo de Cuotas (Toda la Jornada)")
st.info("Ingresa las cuotas principales (1, X, 2) que ves en tu plataforma (ej. Playdoit) para cada encuentro y presiona el botón para calcular el valor esperado (EV) de toda la jornada de golpe.")

partidos_reales = obtener_proximos_partidos_espn("mex.1")

if not partidos_reales:
    st.warning("⚠️ No se encontraron partidos próximos en la cartelera de ESPN para la Liga MX.")
else:
    jornada_data = {}
    
    with st.form("form_jornada_completa"):
        for i, (nombre_llave, info) in enumerate(partidos_reales.items()):
            st.markdown(f"#### ⚽ {info['local']} vs {info['visita']}")
            
            c1, c2, c3, c4 = st.columns(4)
            
            jornada_data[nombre_llave] = {
                "local": info['local'],
                "visita": info['visita'],
                "1": c1.number_input(f"Local (1)", value=2.00, step=0.05, format="%.2f", key=f"j_1_{i}"),
                "X": c2.number_input(f"Empate (X)", value=3.20, step=0.05, format="%.2f", key=f"j_x_{i}"),
                "2": c3.number_input(f"Visita (2)", value=3.10, step=0.05, format="%.2f", key=f"j_2_{i}"),
                "Linea_Goles": c4.number_input(f"Línea Goles", value=2.5, step=0.5, format="%.1f", key=f"j_goles_{i}")
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
                
                resultados = simular_partido_montecarlo(
                    loc, vis,
                    df_historico=df_hist_base,
                    elo_local=e_loc,
                    elo_visita=e_vis,
                    linea_goles=l_goles,
                    linea_corners=9.5,
                    linea_tarjetas=4.5
                )
                
                if isinstance(resultados, str):
                    continue
                
                over_goles_val = resultados['Goles_Over_Under'].get(f'Over {l_goles}', 50.0)
                # Estimación de cuota over goles por defecto segura si no hay input específico
                over_goles_cuota = 1.90 
                
                cuotas_dict = {
                    "1": datos['1'],
                    "X": datos['X'],
                    "2": datos['2'],
                    "Over_Goles": over_goles_cuota,
                    "Under_Goles": round(1.0 / (1.0 - (1.0 / over_goles_cuota)), 2) if over_goles_cuota > 1.0 else 1.90
                }
                
                lineas_default = {"Linea_Goles": l_goles, "Linea_Corners": 9.5, "Linea_Tarjetas": 4.5}
                
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
                st.markdown("Revisa las mejores oportunidades con EV positivo calculadas con tus cuotas manuales.")
                
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

# --- SECCIÓN SECUNDARIA: RANKING ELO Y MODELOS ---
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
