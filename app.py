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

# --- INYECCIÓN DE ESTILOS CSS DE ALTO IMPACTO VISUAL ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  :root{
    --bg:#09110D;
    --panel:#111B14;
    --panel-alt:#18261D;
    --panel-raise:#1F3025;
    --turf:#2E6F40;
    --turf-light:#4E9F64;
    --chalk:#F4F1EA;
    --chalk-dim:#B5B0A1;
    --amber:#FFB703;
    --red:#E05340;
    --steel:#7C8E81;
    --line:rgba(244,241,234,0.08);
    --line-strong:rgba(244,241,234,0.18);
  }

  /* Fondo general y tipografía principal */
  .stApp {
    background: 
      radial-gradient(1200px 600px at 15% -10%, rgba(46,111,64,0.20), transparent 60%),
      radial-gradient(900px 500px at 100% 0%, rgba(255,183,3,0.05), transparent 55%),
      var(--bg);
    font-family: 'Inter', sans-serif;
    color: var(--chalk);
  }

  /* Títulos con estilo industrial / deportivo */
  h1, h2, h3, .custom-title {
    font-family: 'Oswald', sans-serif !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase;
  }

  /* Tarjetas y contenedores de Streamlit con diseño de panel táctico */
  div[data-testid="stVerticalBlock"] > div[style*="background-color"], 
  div.stExpander, 
  div[data-testid="stHorizontalBlock"] {
    background-color: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.2rem;
  }

  /* Estilización de Métricas */
  div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel-raise), var(--panel));
    border: 1px solid var(--line-strong);
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  div[data-testid="stMetric"]:hover {
    border-color: var(--turf-light);
    transform: translateY(-2px);
  }
  [data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--steel) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--amber) !important;
  }

  /* Botones de alto impacto */
  .stButton button {
    background: linear-gradient(135deg, var(--turf), #235531) !important;
    color: var(--chalk) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 16px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(46,111,64,0.4);
    transition: all 0.2s ease;
  }
  .stButton button:hover {
    background: linear-gradient(135deg, var(--turf-light), var(--turf)) !important;
    border-color: var(--amber) !important;
    box-shadow: 0 6px 20px rgba(78,159,100,0.5);
  }

  /* Inputs y Selectboxes estilizados */
  .stSelectbox select, .stNumberInput input {
    background-color: var(--panel-alt) !important;
    border: 1px solid var(--line-strong) !important;
    color: var(--chalk) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stSelectbox select:focus, .stNumberInput input:focus {
    border-color: var(--turf-light) !important;
    box-shadow: 0 0 0 2px rgba(78,159,100,0.2);
  }

  /* Tablas de datos con estética de terminal táctica */
  dataframe, [data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid var(--line-strong);
    background-color: var(--panel);
  }

  /* Divisores personalizados tipo yardas de campo */
  hr {
    border: none;
    height: 1px;
    background: repeating-linear-gradient(90deg, var(--line-strong) 0 2px, transparent 2px 14px);
    margin: 2rem 0;
  }
</style>
""", unsafe_allow_html=True)

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
def obtener_proximos_partidos():
    url = f"{BASE_URL}/fixtures"
    
    hoy = datetime.date.today().strftime("%Y-%m-%d")
    futuro = (datetime.date.today() + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    
    querystring = {"league": str(LIGA_MX_ID), "from": hoy, "to": futuro} 
    
    response = requests.get(url, headers=HEADERS, params=querystring)
    datos = []
    
    if response.status_code == 200:
        datos = response.json().get("response", [])
        
    if not datos:
        querystring_fallback = {"league": str(LIGA_MX_ID), "next": "15"}
        response = requests.get(url, headers=HEADERS, params=querystring_fallback)
        if response.status_code == 200:
            datos = response.json().get("response", [])

    partidos_dict = {}
    for p in datos:
        estado = p.get("fixture", {}).get("status", {}).get("short", "")
        if estado in ["FT", "AET", "PEN", "CANC", "ABD"]: 
            continue
            
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
        
    # --- RESPALDO DE ESPN PARA PARTIDOS ---
    if not partidos_dict:
        url_espn = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
        try:
            res = requests.get(url_espn, timeout=5)
            if res.status_code == 200:
                data = res.json()
                for event in data.get("events", []):
                    estado = event.get("status", {}).get("type", {}).get("name", "")
                    if estado in ["STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FULL_TIME"]:
                        continue
                        
                    competitions = event.get("competitions", [])
                    if competitions:
                        comp = competitions[0]
                        competitors = comp.get("competitors", [])
                        local = "Local"
                        visita = "Visita"
                        for team in competitors:
                            if team.get("homeAway") == "home":
                                local = team.get("team", {}).get("name", "Local")
                            else:
                                visita = team.get("team", {}).get("name", "Visita")
                        
                        fix_id = event.get("id")
                        fecha = event.get("date", "")[:10]
                        
                        llave = f"📅 {fecha} | {local} vs {visita} (ESPN)"
                        partidos_dict[llave] = {
                            "local": local,
                            "visita": visita,
                            "fixture_id": fix_id
                        }
        except Exception as e:
            pass
            
    return partidos_dict

# --- ENCABEZADO PRINCIPAL ESTILIZADO ---
st.markdown("""
<div style="padding: 20px 0 10px 0; text-align: center;">
  <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.2em; color: var(--amber); text-transform: uppercase; margin-bottom: 8px;">
    MODELO ESTADÍSTICO · DATOS HISTÓRICOS Y EN TIEMPO REAL
  </div>
  <h1 style="font-size: clamp(32px, 5vw, 52px); margin: 0; font-weight: 700; color: var(--chalk);">
    LIGA MX <span style="color: var(--turf-light);">ANALYTICS</span>
  </h1>
  <p style="color: var(--chalk-dim); font-size: 14px; max-width: 600px; margin: 10px auto 0 auto; line-height: 1.5;">
    Simulador Montecarlo avanzado (Goles, Córners, Tarjetas) combinado con Value Betting y Criterio de Kelly.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 1. Selecciona el Encuentro")
partidos_reales = obtener_proximos_partidos()

if not partidos_reales:
    st.warning("⚠️ No se encontraron partidos próximos ni en API-Football ni en ESPN.")
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

                cuotas_automaticas = obtener_cuotas_partido(datos_partido["fixture_id"])
                
                l_goles = cuotas_automaticas.get("linea_goles_detectada", "2.5") if cuotas_automaticas else "2.5"
                l_corners = cuotas_automaticas.get("linea_corners_detectada", "9.5") if cuotas_automaticas else "9.5"
                l_tarjetas = cuotas_automaticas.get("linea_tarjetas_detectada", "4.5") if cuotas_automaticas else "4.5"

                resultados = simular_partido_montecarlo(
                    datos_partido["local"], 
                    datos_partido["visita"],
                    df_historico=df_hist_base,
                    elo_local=e_loc,
                    elo_visita=e_vis,
                    linea_goles=float(l_goles),
                    linea_corners=float(l_corners),
                    linea_tarjetas=float(l_tarjetas)
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
                    
                    # Corrección 1: Búsquedas con tolerancias robustas (.get)
                    over_goles = resultados['Goles_Over_Under'].get(f'Over {l_goles}', resultados['Goles_Over_Under'].get(f'Over {float(l_goles)}', 0))
                    col4.metric(f"Más de {l_goles} Goles", f"{over_goles}%", f"Under: {round(100-over_goles, 2)}%")
                    
                    over_corners = resultados['Corners_Totales'].get(f'Over {l_corners} Corners', resultados['Corners_Totales'].get(f'Over {float(l_corners)} Corners', resultados['Corners_Totales'].get(f'Over {l_corners}', 0)))
                    col5.metric(f"Más de {l_corners} Corners", f"{over_corners}%", f"Under: {round(100-over_corners, 2)}%")
                    
                    over_tarjetas = resultados['Tarjetas_Totales'].get(f'Over {l_tarjetas} Tarjetas', resultados['Tarjetas_Totales'].get(f'Over {float(l_tarjetas)} Tarjetas', resultados['Tarjetas_Totales'].get(f'Over {l_tarjetas}', 0)))
                    col6.metric(f"Más de {l_tarjetas} Tarjetas", f"{over_tarjetas}%", f"Under: {round(100-over_tarjetas, 2)}%")
                    
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
                            f"Over {l_goles} Goles": f"Over {l_goles}", 
                            f"Under {l_goles} Goles": f"Under {l_goles}",
                            f"Over {l_corners} Corners": f"Over {l_corners} Corners",
                            f"Under {l_corners} Corners": f"Under {l_corners} Corners",
                            f"Over {l_tarjetas} Tarjetas": f"Over {l_tarjetas} Tarjetas",
                            f"Under {l_tarjetas} Tarjetas": f"Under {l_tarjetas} Tarjetas"
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

                    cuotas_usuario["linea_goles_detectada"] = str(l_goles)
                    cuotas_usuario["linea_corners_detectada"] = str(l_corners)
                    cuotas_usuario["linea_tarjetas_detectada"] = str(l_tarjetas)

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
            
            # Corrección 2: Inyectar variables dinámicas al modelo predictivo
            preds_ml = ml_predictor.predecir_mercados_completos(
                df_historico, 
                datos_partido['local'], 
                datos_partido['visita'], 
                g_l_sim, 
                g_v_sim,
                puntos_local,  
                puntos_visita,
                linea_goles=float(l_goles),
                linea_corners=float(l_corners),
                linea_tarjetas=float(l_tarjetas)
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
                
                # Corrección 3: Búsquedas con tolerancias robustas (.get)
                over_g_ml = preds_ml['Goles_Over_Under'].get(f'Over {l_goles}', preds_ml['Goles_Over_Under'].get(f'Over {float(l_goles)}', 0))
                under_g_ml = preds_ml['Goles_Over_Under'].get(f'Under {l_goles}', preds_ml['Goles_Over_Under'].get(f'Under {float(l_goles)}', 0))
                
                over_c_ml = preds_ml['Corners_Totales'].get(f'Over {l_corners} Corners', preds_ml['Corners_Totales'].get(f'Over {float(l_corners)} Corners', preds_ml['Corners_Totales'].get(f'Over {l_corners}', 0)))
                under_c_ml = preds_ml['Corners_Totales'].get(f'Under {l_corners} Corners', preds_ml['Corners_Totales'].get(f'Under {float(l_corners)} Corners', preds_ml['Corners_Totales'].get(f'Under {l_corners}', 0)))
                
                over_t_ml = preds_ml['Tarjetas_Totales'].get(f'Over {l_tarjetas} Tarjetas', preds_ml['Tarjetas_Totales'].get(f'Over {float(l_tarjetas)} Tarjetas', preds_ml['Tarjetas_Totales'].get(f'Over {l_tarjetas}', 0)))
                under_t_ml = preds_ml['Tarjetas_Totales'].get(f'Under {l_tarjetas} Tarjetas', preds_ml['Tarjetas_Totales'].get(f'Under {float(l_tarjetas)} Tarjetas', preds_ml['Tarjetas_Totales'].get(f'Under {l_tarjetas}', 0)))
                
                with ml_col4:
                    st.metric(f"Over {l_goles} Goles", f"{over_g_ml}%")
                    st.metric(f"Under {l_goles} Goles", f"{under_g_ml}%")
                    
                with ml_col5:
                    st.metric(f"Over {l_corners} Córners", f"{over_c_ml}%")
                    st.metric(f"Under {l_corners} Córners", f"{under_c_ml}%")
                    
                with ml_col6:
                    st.metric(f"Over {l_tarjetas} Tarjetas", f"{over_t_ml}%")
                    st.metric(f"Under {l_tarjetas} Tarjetas", f"{under_t_ml}%")
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
        ).rename(columns={'ELO_Rating': 'ELO_Visual'}).drop('Equipo', axis=1, errors='ignore')

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
