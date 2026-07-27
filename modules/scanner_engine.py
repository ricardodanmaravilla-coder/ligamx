import os
import requests
import pandas as pd
from datetime import datetime
from github import Github
import streamlit as st
import io
import time  # <--- IMPORTANTE: La librería que evitará el bloqueo de la API

from modules.stats_engine import calcular_expectativa_partido
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido, evaluar_mercado_avanzado
from modules.ml_engine import PredictorML

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262

def registrar_apuesta_github(partido, mercado, prob_modelo, cuota, kelly_pct, bankroll_inicial=5000):
    if kelly_pct <= 0:
        return

    token = st.secrets["MI_GITHUB_TOKEN"]
    g = Github(token)
    nombre_repo = "ricardodanmaravilla-coder/ligamx" 
    repo = g.get_repo(nombre_repo)
    ruta_archivo_github = 'data/registro_apuestas.csv'
    
    porcentaje_seguro = min(kelly_pct, 10.0) 
    inversion_mxn = round(bankroll_inicial * (porcentaje_seguro / 100), 2)
    ganancia_potencial = round((inversion_mxn * cuota) - inversion_mxn, 2)
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    nueva_fila_df = pd.DataFrame([{
        "Fecha_Registro": fecha_actual,
        "Partido": partido,
        "Mercado": mercado,
        "Prob_Modelo_%": prob_modelo,
        "Cuota_Apostada": cuota,
        "Kelly_%": round(porcentaje_seguro, 2),
        "Inversion_MXN": inversion_mxn,
        "Ganancia_Potencial_MXN": ganancia_potencial,
        "Estado": "Pendiente",
        "Retorno_Real_MXN": 0.0
    }])

    try:
        archivo_en_repo = repo.get_contents(ruta_archivo_github)
        contenido_actual = archivo_en_repo.decoded_content.decode('utf-8')
        df_existente = pd.read_csv(io.StringIO(contenido_actual))
        
        duplicado = df_existente[(df_existente['Partido'] == partido) & (df_existente['Mercado'] == mercado) & (df_existente['Estado'] == 'Pendiente')]
        if not duplicado.empty:
            return

        df_final = pd.concat([df_existente, nueva_fila_df], ignore_index=True)
        nuevo_contenido_csv = df_final.to_csv(index=False)
        
        repo.update_file(
            path=archivo_en_repo.path,
            message=f"🤖 Registro automático de apuesta: {mercado}",
            content=nuevo_contenido_csv,
            sha=archivo_en_repo.sha
        )
    except Exception as e:
        nuevo_contenido_csv = nueva_fila_df.to_csv(index=False)
        repo.create_file(
            path=ruta_archivo_github,
            message="🤖 Creación de bitácora inicial",
            content=nuevo_contenido_csv
        )

def obtener_ultimo_elo(df, equipo):
    try:
        if df is not None and not df.empty:
            equipo_norm = str(equipo).strip().lower()
            
            mask = (df['Local'].astype(str).str.strip().str.lower() == equipo_norm) | \
                   (df['Visitante'].astype(str).str.strip().str.lower() == equipo_norm)
            df_eq = df[mask]
            
            if not df_eq.empty:
                ultima_fila = df_eq.iloc[-1]
                local_csv_norm = str(ultima_fila['Local']).strip().lower()
                
                if local_csv_norm == equipo_norm:
                    for col in ['ELO_Local', 'ELO_L', 'Elo_Local', 'Elo_L']:
                        if col in df.columns and not pd.isna(ultima_fila[col]): 
                            return float(ultima_fila[col])
                else:
                    for col in ['ELO_Visita', 'ELO_V', 'Elo_Visita', 'Elo_V', 'ELO_Visitante']:
                        if col in df.columns and not pd.isna(ultima_fila[col]): 
                            return float(ultima_fila[col])
    except Exception as e:
        pass
    return 1500.0
        
def escanear_jornada_actual(temporada_actual=2026):
    ml_escanner = PredictorML()
    df_historico_ml = None
    try:
        df_historico_ml = pd.read_csv("data/historico_ligamx_completo.csv")
        df_historico_ml['Local'] = df_historico_ml['Local'].str.strip()
        df_historico_ml['Visitante'] = df_historico_ml['Visitante'].str.strip()
        ml_escanner.entrenar(df_historico_ml)
    except Exception:
        pass

    url_rounds = f"{BASE_URL}/fixtures/rounds"
    res_rounds = requests.get(url_rounds, headers=HEADERS, params={"league": LIGA_MX_ID, "season": temporada_actual, "current": "true"})
    
    jornada_actual = None
    if res_rounds.status_code == 200:
        rounds_data = res_rounds.json().get("response", [])
        if rounds_data:
            jornada_actual = rounds_data[0]
            
    url = f"{BASE_URL}/fixtures"
    params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
    if jornada_actual:
        params["round"] = jornada_actual
    
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        return []
        
    fixtures = res.json().get("response", [])
    if not fixtures:
        params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            fixtures = res.json().get("response", [])

    oportunidades_oro = []

    st.write("---")
    st.write("🔎 **Analizando partidos de la jornada...** *(Esto tomará unos segundos para no saturar la API)*")

    for p in fixtures:
        try:
            fix_id = p["fixture"]["id"]
            local = p["teams"]["home"]["name"]
            visita = p["teams"]["away"]["name"]
            fecha = p["fixture"]["date"][:16].replace("T", " ")
            
            st.write(f"⚙️ Revisando: **{local} vs {visita}**")
            
            # --- LA PAUSA MÁGICA QUE EVITA QUE LA API NOS BLOQUEE ---
            time.sleep(1.5) 
            
            cuotas = obtener_cuotas_partido(fix_id)
            if not cuotas: 
                st.write(f"⚠️ *Saltado: La API bloqueó la petición o no hay momios aún.*")
                continue
                
            elo_loc = obtener_ultimo_elo(df_historico_ml, local)
            elo_vis = obtener_ultimo_elo(df_historico_ml, visita)
            
            resultados = simular_partido_montecarlo(
                local, visita, 
                df_historico=df_historico_ml, 
                elo_local=elo_loc, 
                elo_visita=elo_vis
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
                
                # Chivato ELO para asegurarnos de que ya no es 1500
                if "America" in local or "Santos" in local:
                    st.write(f"🔬 DIAGNÓSTICO ELO: Loc {elo_loc} | Vis {elo_vis}")
                
                preds_ml = ml_escanner.predecir_mercados_completos(
                    df_historico_ml, local, visita, g_l_sim, g_v_sim, elo_loc, elo_vis
                )
                
                if "Resultado_1X2" in preds_ml:
                    prob_ml_local = preds_ml['Resultado_1X2']['Gana Local']
                    prob_ml_empate = preds_ml['Resultado_1X2']['Empate']
                    prob_ml_visita = preds_ml['Resultado_1X2']['Gana Visita']
                    
                    prob_ml_over_g = preds_ml['Goles_Over_Under']['Over 2.5']
                    prob_ml_under_g = preds_ml['Goles_Over_Under']['Under 2.5']
                    
                    prob_ml_over_c = preds_ml['Corners_Totales']['Over 9.5 Corners']
                    prob_ml_under_c = preds_ml['Corners_Totales']['Under 9.5 Corners']
                    
                    prob_ml_over_t = preds_ml['Tarjetas_Totales']['Over 4.5 Tarjetas']
                    prob_ml_under_t = preds_ml['Tarjetas_Totales']['Under 4.5 Tarjetas']

            prob_mc_dict = {
                "Gana Local": resultados.get('Resultado_1X2', {}).get('Gana Local', 0.0),
                "Empate": resultados.get('Resultado_1X2', {}).get('Empate', 0.0),
                "Gana Visita": resultados.get('Resultado_1X2', {}).get('Gana Visita', 0.0),
                "Over 2.5 Goles": resultados.get('Goles_Over_Under', {}).get('Over 2.5', 0.0),
                "Under 2.5 Goles": resultados.get('Goles_Over_Under', {}).get('Under 2.5', 0.0),
                "Over 9.5 Corners": resultados.get('Corners_Totales', {}).get('Over 9.5 Corners', 0.0),
                "Under 9.5 Corners": resultados.get('Corners_Totales', {}).get('Under 9.5 Corners', 0.0),
                "Over 4.5 Tarjetas": resultados.get('Tarjetas_Totales', {}).get('Over 4.5 Tarjetas', 0.0),
                "Under 4.5 Tarjetas": resultados.get('Tarjetas_Totales', {}).get('Under 4.5 Tarjetas', 0.0)
            }

            prob_ml_dict = {
                "Gana Local": prob_ml_local,
                "Empate": prob_ml_empate,
                "Gana Visita": prob_ml_visita,
                "Over 2.5 Goles": prob_ml_over_g,
                "Under 2.5 Goles": prob_ml_under_g,
                "Over 9.5 Corners": prob_ml_over_c,
                "Under 9.5 Corners": prob_ml_under_c,
                "Over 4.5 Tarjetas": prob_ml_over_t,
                "Under 4.5 Tarjetas": prob_ml_under_t
            }

            mercados_a_mapear = [
                ("Gana Local", "Home"),
                ("Empate", "Draw"),
                ("Gana Visita", "Away"),
                ("Over 2.5 Goles", "Over 2.5"),
                ("Under 2.5 Goles", "Under 2.5"),
                ("Over 9.5 Corners", "Over 9.5"),
                ("Under 9.5 Corners", "Under 9.5"),
                ("Over 4.5 Tarjetas", "Over 4.5"),
                ("Under 4.5 Tarjetas", "Under 4.5")
            ]

            for nombre_m, llave_api in mercados_a_mapear:
                try:
                    p_mc = float(prob_mc_dict.get(nombre_m, 0.0))
                except:
                    p_mc = 0.0
                    
                try:
                    p_ml = float(prob_ml_dict.get(nombre_m, 0.0))
                except:
                    p_ml = 0.0
                
                raw_cuota = cuotas.get(llave_api)
                if not raw_cuota:
                    if "Corners" in nombre_m:
                        raw_cuota = cuotas.get("Over 9.5") if "Over" in nombre_m else cuotas.get("Under 9.5")
                    elif "Tarjetas" in nombre_m:
                        raw_cuota = cuotas.get("Over 4.5") if "Over" in nombre_m else cuotas.get("Under 4.5")
                
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
                            "Veredicto": "✅ CONSENSO BLINDADO",
                            "Fixture_ID": fix_id
                        })

        except Exception as e:
            continue
            
    st.write("✅ **Escaneo completado.**")
    return oportunidades_oro
